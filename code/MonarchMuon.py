from __future__ import annotations

import argparse
import gc
import json
import math
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from functools import partial
import torch.nn.init as init

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from collections import defaultdict
if TYPE_CHECKING:
    from comet_ml import Experiment


OUTPUT_ROOT = Path("data")
DATA_ROOT = OUTPUT_ROOT 

DATASET_NAME = "wikitext"
DATASET_CONFIG = "wikitext-103-raw-v1"
DATASET_SPLIT = "train"
ENCODING_NAME = "gpt2"

TOTAL_TOKENS = 25_000_000
TEST_TOKENS = 1_000_000
TRAIN_TOKENS = TOTAL_TOKENS - TEST_TOKENS

BATCH_SIZE = 64
BLOCK_SIZE = 256
GRAD_ACCUM = 2
NUM_EPOCHS = 50
EVAL_BATCHES = 25
LOG_EVERY_STEPS = 50
WARMUP_STEPS = 100
MIN_LR_RATIO = 0.1
SCHEDULER_EPOCHS = 10
ADAM_LR = 1e-3
MUON_LR = 5e-3
MUON_MOMENTUM = 0.95
WEIGHT_DECAY = 0.1
GRAD_CLIP = 1.0

SEED = 42
DEFAULT_DEVICE = "cpu"

TRAIN_BIN_PATH = DATA_ROOT / "train.bin"
TEST_BIN_PATH = DATA_ROOT / "test.bin"
META_PATH = DATA_ROOT / "meta.json"


@dataclass(frozen=True)
class ModelSpec:
    name: str
    n_layer: int
    n_head: int
    n_embd: int


MODEL_SPECS = [
     ModelSpec(name="gpt_l8_h8_d512Monarch", n_layer=8, n_head=8, n_embd=512),
]


def set_seed(seed: int, device: str) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested_device: str) -> str:
    if requested_device not in {"cpu", "cuda"}:
        raise ValueError(f"Unsupported device: {requested_device}")
    if requested_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Device 'cuda' was requested, but CUDA is not available.")
    return requested_device

def load_local_wikitext_metadata() -> dict:
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    missing = [path for path in (TRAIN_BIN_PATH, TEST_BIN_PATH, META_PATH) if not path.exists()]
    if missing:
        missing_str = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(
            "Expected prebuilt local Wikitext token files, but some are missing: "
            f"{missing_str}"
        )

    with open(META_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)

    train_expected_bytes = TRAIN_TOKENS * np.dtype(np.uint16).itemsize
    test_expected_bytes = TEST_TOKENS * np.dtype(np.uint16).itemsize
    train_actual_bytes = TRAIN_BIN_PATH.stat().st_size
    test_actual_bytes = TEST_BIN_PATH.stat().st_size

    if train_actual_bytes != train_expected_bytes or test_actual_bytes != test_expected_bytes:
        raise RuntimeError(
            "Local token files have unexpected size: "
            f"train={train_actual_bytes} bytes (expected {train_expected_bytes}), "
            f"test={test_actual_bytes} bytes (expected {test_expected_bytes})"
        )

    print("Using existing local Wikitext-103 token files.")
    print(f"Train tokens: {meta.get('train_tokens', TRAIN_TOKENS):,}")
    print(f"Test tokens:  {meta.get('test_tokens', TEST_TOKENS):,}")
    print(f"Data dir:     {DATA_ROOT.resolve()}")
    return meta


class SequentialTokenLoader:
    def __init__(self, path: Path, batch_size: int, block_size: int, device: str):
        self.path = Path(path)
        self.data = np.memmap(self.path, dtype=np.uint16, mode="r")
        self.batch_size = batch_size
        self.block_size = block_size
        self.device = device
        self.tokens_per_batch = batch_size * block_size
        self.max_cursor = len(self.data) - (self.tokens_per_batch + 1)
        if self.max_cursor <= 0:
            raise ValueError(
                f"{self.path} is too small for batch_size={batch_size}, block_size={block_size}"
            )
        self.cursor = 0

    def reset(self, cursor: int = 0) -> None:
        self.cursor = int(cursor)

    def __len__(self) -> int:
        return max(1, (len(self.data) - 1) // self.tokens_per_batch)

    def next_batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self.cursor > self.max_cursor:
            self.cursor = 0

        chunk = np.asarray(
            self.data[self.cursor:self.cursor + self.tokens_per_batch + 1],
            dtype=np.int64,
        )
        if chunk.shape[0] != self.tokens_per_batch + 1:
            self.cursor = 0
            chunk = np.asarray(
                self.data[self.cursor:self.cursor + self.tokens_per_batch + 1],
                dtype=np.int64,
            )

        x = torch.from_numpy(chunk[:-1].reshape(self.batch_size, self.block_size))
        y = torch.from_numpy(chunk[1:].reshape(self.batch_size, self.block_size))
        self.cursor += self.tokens_per_batch
        return x.to(self.device), y.to(self.device)


def build_loaders(device: str) -> tuple[SequentialTokenLoader, SequentialTokenLoader]:
    train_loader = SequentialTokenLoader(TRAIN_BIN_PATH, BATCH_SIZE, BLOCK_SIZE, device)
    test_loader = SequentialTokenLoader(TEST_BIN_PATH, BATCH_SIZE, BLOCK_SIZE, device)
    return train_loader, test_loader


def Newton_Schulz(matrix: torch.Tensor) -> torch.Tensor:
    matrix_norm = matrix / (matrix.norm(p="fro") + 1e-7)
    transposed = False
    if matrix_norm.shape[0] > matrix_norm.shape[1]:
        matrix_norm = matrix_norm.T
        transposed = True
    for _ in range(5):
        mm_t = matrix_norm @ matrix_norm.T
        matrix_norm = (
            3.4445 * matrix_norm
            - 4.7750 * mm_t @ matrix_norm
            + 2.0315 * (mm_t @ mm_t) @ matrix_norm
        )
    if transposed:
        matrix_norm = matrix_norm.T
    return matrix_norm


def Newton_Schulz_batched(B):
    # B: (nblocks, d1, d2)
    norms = B.norm(p='fro', dim=(-2, -1), keepdim=True)
    B_norm = B / (norms + 1e-7)

    need_transpose = B_norm.shape[-2] > B_norm.shape[-1]
    if need_transpose:
        B_norm = B_norm.transpose(-2, -1)

    for _ in range(5):
        BBt = B_norm @ B_norm.transpose(-2, -1)
        B_norm = 3.4445 * B_norm - 4.7750 * BBt @ B_norm + 2.0315 * (BBt @ BBt) @ B_norm

    if need_transpose:
        B_norm = B_norm.transpose(-2, -1)
    return B_norm


class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr: float = MUON_LR, momentum: float = MUON_MOMENTUM):
        defaults = {"lr": lr, "momentum": momentum}
        super().__init__(params, defaults)

    @torch.no_grad()
    def _get_momentum_buffer(self, param: torch.nn.Parameter) -> torch.Tensor:
        state = self.state[param]
        if "momentum_buffer" not in state:
            state["momentum_buffer"] = torch.zeros_like(param.grad)
        return state["momentum_buffer"]

    @torch.no_grad()
    def _update_non_batched_param(self, param: torch.nn.Parameter, lr: float, momentum: float) -> None:
        grad = param.grad
        if grad is None:
            return

        buffer = self._get_momentum_buffer(param)
        buffer.mul_(momentum).add_(grad)

        reshaped = buffer.reshape(buffer.shape[0], -1) if buffer.ndim > 2 else buffer
        ortho = Newton_Schulz(reshaped).reshape_as(buffer)
        scale = (reshaped.shape[0] / reshaped.shape[1]) ** 0.5

        param.data.add_(ortho, alpha=-lr * scale)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]

            # Группируем только 3D тензоры по (device, dtype, a, b),
            # где shape == [nblocks, a, b]
            grouped_3d = defaultdict(list)

            for param in group["params"]:
                if param.grad is None:
                    continue

                grad = param.grad

                if grad.ndim == 3:
                    buffer = self._get_momentum_buffer(param)
                    buffer.mul_(momentum).add_(grad)

                    key = (
                        buffer.device,
                        buffer.dtype,
                        buffer.shape[1],
                        buffer.shape[2],
                    )
                    grouped_3d[key].append((param, buffer))
                else:
                    self._update_non_batched_param(param, lr, momentum)

            # Для каждой группы одинаковых [*, a, b]
            # делаем один большой batched Newton-Schulz
            for (_, _, a, b), items in grouped_3d.items():
                merged = torch.cat([buf for _, buf in items], dim=0)  # [sum_nblocks, a, b]
                ortho_merged = Newton_Schulz_batched(merged)
                scale = (a / b) ** 0.5

                offset = 0
                for param, buffer in items:
                    nblocks = buffer.shape[0]
                    ortho = ortho_merged[offset:offset + nblocks]
                    param.data.add_(ortho, alpha=-lr * scale)
                    offset += nblocks

        return loss
    
class StructuredLinear(nn.Module):

    def __init__(self, in_features, out_features, bias=True, device=None, dtype=None):
        """Subclasses should call reset_parameters
        """
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        # Subclasses may override {in,out}_features_extended
        if not hasattr(self, 'in_features_extended'):
            self.in_features_extended = in_features
        if not hasattr(self, 'out_features_extended'):
            self.out_features_extended = out_features
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)

    def reset_parameters(self) -> None:
        self.set_weights_from_dense_init(dense_init_fn_=partial(init.kaiming_uniform_, a=math.sqrt(5)))
        self.reset_parameters_bias()

    def set_weights_from_dense_init(self, dense_init_fn_):
        raise NotImplementedError

    def reset_parameters_bias(self):
        if self.bias is not None:
            fan_in = self.bias.shape[-1]
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)

    @property
    def saving(self):
        raise NotImplementedError

    def convert_to_dense_weight(self):
        factory_kwargs = {'device': self.weight.device, 'dtype': self.weight.dtype}
        dense_weight = self.forward_matmul(torch.eye(self.in_features, **factory_kwargs)).T
        return dense_weight

    def preprocess(self, x):
        in_features = x.shape[-1]
        if in_features < self.in_features_extended:
            x = F.pad(x, (0, self.in_features_extended - in_features))
        return x

    def postprocess(self, output):
        out_features_extended = output.shape[-1]
        if out_features_extended > self.out_features:
            output = output[..., :self.out_features]
        return output

    def forward_matmul(self, x):
        raise NotImplementedError

    def forward(self, x):
        output = self.forward_matmul(x)
        # Convert bias to output.dtype in case of AMP, otherwise bias and activation will be in FP32
        return (output + self.bias.to(dtype=output.dtype)) if self.bias is not None else output    

def blockdiag_butterfly_multiply_reference(x, w1_bfly, w2_bfly, version=2):
    """
    This implementation is slow but more likely to be correct.
    There are 3 implementations, which should all yield the same answer
    Arguments:
        x: (batch, n)
        w1_bfly: (k, q, p), where k = n / p
        w2_bfly: (l, s, r), where l = k * q / r = n * q / (p * r)
    Outputs:
        out: (batch, m), where m = l * s = n * s * q / (p * r)
    """
    if version not in [1, 2, 3]:
        raise NotImplementedError('version must be either 1, 2, or 3')
    batch, n = x.shape
    k, q, p = w1_bfly.shape
    l, s, r = w2_bfly.shape
    assert k * p == n
    assert l * r == k * q

    x_reshaped = rearrange(x, 'b (k p) -> b k p', k=k)
    if version == 1:  # Implementation 1 (only works for when k = q = p = l = s = r = sqrt(n))
        assert k == q == p == l == s == r == int(math.sqrt(n))
        return torch.einsum('bkp,kqp,qlk->blq', x_reshaped, w1_bfly, w2_bfly).reshape(batch, n)
    elif version == 2:  # Implementation 2
        out1 = torch.einsum('kqp,bkp->bkq', w1_bfly, x_reshaped)
        out1 = rearrange(rearrange(out1, 'b k q -> b (k q)'), 'b (r l) -> b l r', l=l)
        return torch.einsum('lsr,blr->bsl', w2_bfly, out1).reshape(batch, s * l)
    # Implementation 3: most likely to be correct, but it's the slowest
    elif version == 3:
        w1_dense = torch.block_diag(*torch.unbind(w1_bfly, dim=0))
        out1 = F.linear(x, w1_dense)
        out1 = rearrange(out1, 'b (r l) -> b (l r)', l=l)
        w2_dense = torch.block_diag(*torch.unbind(w2_bfly, dim=0))
        out2 = F.linear(out1, w2_dense)
        out2 = rearrange(out2, 'b (l s) -> b (s l)', l=l)
        return out2


class BlockdiagButterflyMultiply(torch.autograd.Function):

    """This is a faster implementation, with careful memory copies for the fastest
    bmm performance.
    The backward pass is also written manually with careful memory copies.
    Arguments:
        x: (batch, n)
        w1_bfly: (k, q, p), where k = n / p
        w2_bfly: (l, s, r), where l = k * q / r = n * q / (p * r)
    Outputs:
        out: (batch, m), where m = l * s = n * s * q / (p * r)
    """

    @staticmethod
    @torch.cuda.amp.custom_fwd(cast_inputs=torch.bfloat16)
    def forward(ctx, x, w1_bfly, w2_bfly):
        batch_shape, n = x.shape[:-1], x.shape[-1]
        batch_dim = np.prod(batch_shape)
        k, q, p = w1_bfly.shape
        l, s, r = w2_bfly.shape
        assert k * p == n
        assert l * r == k * q
        x_reshaped = x.reshape(batch_dim, k, p).transpose(0, 1)
        out1 = torch.empty(batch_dim, k, q, device=x.device, dtype=x.dtype).transpose(0, 1)
        out1 = torch.bmm(x_reshaped, w1_bfly.transpose(-1, -2), out=out1)
        out1 = out1.transpose(0, 1).reshape(batch_dim, r, l).transpose(-1, -2).contiguous().transpose(0, 1)
        out2 = torch.empty(batch_dim, l, s, device=x.device, dtype=x.dtype).transpose(0, 1)
        out2 = torch.bmm(out1, w2_bfly.transpose(-1, -2), out=out2)
        out2 = out2.permute(1, 2, 0).reshape(*batch_shape, s * l)
        ctx.save_for_backward(x, w1_bfly, w2_bfly, out1)
        return out2

    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(ctx, dout):
        x, w1_bfly, w2_bfly, out1 = ctx.saved_tensors
        batch_shape, n = x.shape[:-1], x.shape[-1]
        batch_dim = np.prod(batch_shape)
        k, q, p = w1_bfly.shape
        l, s, r = w2_bfly.shape
        # assert k * p == n
        # assert l * r == k * q
        dx, dw1_bfly, dw2_bfly = None, None, None
        # dout_reshaped = dout.reshape(batch_dim, sqrtn, sqrtn).permute(2, 1, 0).contiguous()
        dout_reshaped = dout.reshape(batch_dim, s, l).transpose(-1, -2).contiguous()
        dout_reshaped = dout_reshaped.transpose(0, 1)
        if ctx.needs_input_grad[2]:
            # dw2_bfly = torch.empty(l, s, r, device=w2_bfly.device, dtype=w2_bfly.dtype)
            # dw2_bfly = torch.bmm(dout_reshaped.transpose(-1, -2), out1, out=dw2_bfly)
            dw2_bfly = torch.bmm(dout_reshaped.transpose(-1, -2), out1.conj())
        if ctx.needs_input_grad[1] or ctx.needs_input_grad[0]:
            dout1 = torch.empty(batch_dim, l, r, device=x.device, dtype=x.dtype).transpose(0, 1)
            dout1 = torch.bmm(dout_reshaped, w2_bfly.conj(), out=dout1)
            dout1 = dout1.transpose(0, 1).transpose(-1, -2).contiguous().reshape(batch_dim, k, q).transpose(0, 1)
            # dout1 = dout1.permute(1, 2, 0).contiguous().transpose(0, 1)
            if ctx.needs_input_grad[0]:
                dx = torch.empty(batch_dim, k, p, device=x.device, dtype=x.dtype)
                dx = torch.bmm(dout1, w1_bfly.conj(), out=dx.transpose(0, 1)).transpose(0, 1).reshape(*batch_shape, n)
            if ctx.needs_input_grad[1]:
                x_reshaped = x.reshape(batch_dim, k, p).transpose(0, 1)
                dw1_bfly = torch.bmm(dout1.transpose(-1, -2), x_reshaped.conj())
        return dx, dw1_bfly, dw2_bfly

blockdiag_butterfly_multiply = BlockdiagButterflyMultiply.apply
        
class MonarchLinear(StructuredLinear):

    def __init__(self, *args, nblocks=2, **kwargs):
        super().__init__(*args, **kwargs)
        in_blksz = int(math.ceil(self.in_features / nblocks))
        out_blksz = int(math.ceil(self.out_features / nblocks))
        self.in_features_extended = in_blksz * nblocks
        self.out_features_extended = out_blksz * nblocks

        if self.in_features_extended < self.out_features_extended:
            self.blkdiag1 = nn.Parameter(torch.empty(nblocks, in_blksz, in_blksz))
            self.blkdiag2 = nn.Parameter(torch.empty(nblocks, out_blksz, in_blksz))
        else:
            self.blkdiag1 = nn.Parameter(torch.empty(nblocks, out_blksz, in_blksz))
            self.blkdiag2 = nn.Parameter(torch.empty(nblocks, out_blksz, out_blksz))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Mimic init.kaiming_uniform: https://github.com/pytorch/pytorch/blob/24087d07ca7ffa244575d259711dd7c99245a67a/torch/nn/init.py#L360
        for blkdiag in [self.blkdiag1, self.blkdiag2]:
            fan_in = blkdiag.shape[-1]
            gain = init.calculate_gain(nonlinearity='leaky_relu', param=math.sqrt(5))
            std = gain / math.sqrt(fan_in)
            bound = math.sqrt(3.0) * std  # Calculate uniform bounds from standard deviation
            with torch.no_grad():
                blkdiag.uniform_(-bound, bound)
        self.reset_parameters_bias()

    @property
    def saving(self):
        return ((self.blkdiag1.numel() + self.blkdiag2.numel())
                / (self.in_features * self.out_features))

    def forward_matmul(self, x):
        output = blockdiag_butterfly_multiply(self.preprocess(x), self.blkdiag1, self.blkdiag2)
        return self.postprocess(output)
    
@dataclass
class GPTConfig:
    vocab_size: int = 50257
    block_size: int = BLOCK_SIZE
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 672
    dropout: float = 0.1

class CausalSelfAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        # ── MonarchLinear вместо nn.Linear ──
        self.c_attn  = MonarchLinear(cfg.n_embd, 3 * cfg.n_embd, bias=True)
        self.c_proj  = MonarchLinear(cfg.n_embd, cfg.n_embd,     bias=True)
        # ────────────────────────────────────
        self.attn_drop  = nn.Dropout(cfg.dropout)
        self.resid_drop = nn.Dropout(cfg.dropout)
        self.n_head = cfg.n_head
        self.n_embd = cfg.n_embd
        self.register_buffer('bias', torch.tril(torch.ones(cfg.block_size, cfg.block_size))
                             .view(1, 1, cfg.block_size, cfg.block_size))

    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_drop(self.c_proj(y))


class MLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # ── MonarchLinear вместо nn.Linear ──
        self.c_fc   = MonarchLinear(cfg.n_embd,     4 * cfg.n_embd, bias=True)
        self.c_proj = MonarchLinear(4 * cfg.n_embd, cfg.n_embd,     bias=True)
        # ────────────────────────────────────
        self.drop = nn.Dropout(cfg.dropout)
        self.act  = nn.GELU()

    def forward(self, x):
        return self.drop(self.c_proj(self.act(self.c_fc(x))))


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.ln_1 = nn.LayerNorm(cfg.n_embd)
        self.attn = CausalSelfAttention(cfg)
        self.ln_2 = nn.LayerNorm(cfg.n_embd)
        self.mlp  = MLP(cfg)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.transformer = nn.ModuleDict(dict(
            wte  = nn.Embedding(cfg.vocab_size, cfg.n_embd),
            wpe  = nn.Embedding(cfg.block_size, cfg.n_embd),
            drop = nn.Dropout(cfg.dropout),
            h    = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)]),
            ln_f = nn.LayerNorm(cfg.n_embd),
        ))
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight

        self.apply(self._init_weights)

        total   = sum(p.numel() for p in self.parameters())
        monarch = sum(p.numel() for n, p in self.named_parameters()
                      if isinstance(p, nn.Parameter)
                      and any(x in n for x in ['.R', '.L'])
                      and 'bias' not in n)
        print(f'Параметров всего:     {total:,}')
        print(f'Из них Monarch (L,R): {monarch:,}  ({monarch/total*100:.1f}%)')

    def _init_weights(self, module):
        # MonarchLinear инициализируется в своём __init__
        # Здесь обрабатываем только nn.Linear (lm_head) и Embedding
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.size()
        pos  = torch.arange(T, device=idx.device)
        x    = self.transformer.drop(
            self.transformer.wte(idx) + self.transformer.wpe(pos)
        )
        for block in self.transformer.h:
            x = block(x)
        x      = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        loss   = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss


@torch.no_grad()
def estimate_loss(
    model: GPT,
    train_eval_loader: SequentialTokenLoader,
    test_eval_loader: SequentialTokenLoader,
    eval_batches: int = EVAL_BATCHES,
) -> dict[str, float]:
    model.eval()
    losses = {"train": [], "test": []}

    for split, loader in (("train", train_eval_loader), ("test", test_eval_loader)):
        loader.reset(0)
        for _ in range(eval_batches):
            x, y = loader.next_batch()
            _, loss = model(x, y)
            losses[split].append(float(loss.detach().cpu()))

    model.train()
    return {name: float(np.mean(values)) for name, values in losses.items()}



def clone_checkpoint_to_cpu(checkpoint: dict) -> dict:
    result = {}
    for key, value in checkpoint.items():
        if isinstance(value, dict):
            result[key] = clone_checkpoint_to_cpu(value)
        elif torch.is_tensor(value):
            result[key] = value.detach().cpu()
        else:
            result[key] = value
    return result


def save_history(history: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "loss_history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)


def count_parameters(model: nn.Module) -> int:
    return sum(param.numel() for param in model.parameters())


def seconds_to_step(seconds: float) -> int:
    return max(0, int(round(seconds * 1000.0)))


def loss_to_perplexity(loss: float) -> float:
    try:
        return math.exp(loss)
    except OverflowError:
        return float("inf")


def get_lr_scale(
    step: int,
    warmup_steps: int,
    scheduler_steps: int,
    min_lr_ratio: float,
) -> float:
    """
    LR changes only during the first `scheduler_steps`.
    After that it stays constant at the last scheduled value.
    """
    if scheduler_steps <= 0:
        return 1.0

    clamped_step = min(step, scheduler_steps - 1)

    if warmup_steps > 0 and clamped_step < warmup_steps:
        return float(clamped_step + 1) / float(warmup_steps)

    decay_steps = max(1, scheduler_steps - warmup_steps)
    decay_progress = min(max(clamped_step - warmup_steps, 0), decay_steps) / decay_steps
    cosine = 0.5 * (1.0 + math.cos(math.pi * decay_progress))
    return min_lr_ratio + (1.0 - min_lr_ratio) * cosine


def build_model_and_optimizers(
    model_spec: ModelSpec,
    device: str,
) -> tuple[GPT, Muon, torch.optim.AdamW, GPTConfig]:
    cfg = GPTConfig(
        block_size=BLOCK_SIZE,
        n_layer=model_spec.n_layer,
        n_head=model_spec.n_head,
        n_embd=model_spec.n_embd,
    )
    model = GPT(cfg).to(device)

    muon_params = []
    adam_regular = []
    adam_unembed = []

    for name, param in model.named_parameters():
        if "lm_head" in name:
            adam_unembed.append(param)
        elif "wte" in name or "wpe" in name:
            adam_regular.append(param)
        elif param.ndim >= 2:
            muon_params.append(param)
        else:
            adam_regular.append(param)

    unembed_lr = ADAM_LR / math.sqrt(cfg.n_embd)
    optimizer_muon = Muon(muon_params, lr=MUON_LR, momentum=MUON_MOMENTUM)
    optimizer_adam = torch.optim.AdamW(
        [
            {"params": adam_regular, "lr": ADAM_LR},
            {"params": adam_unembed, "lr": unembed_lr},
        ],
        weight_decay=WEIGHT_DECAY,
    )
    return model, optimizer_muon, optimizer_adam, cfg


def create_comet_experiment(
    args: argparse.Namespace,
    model_spec: ModelSpec,
    model_cfg: GPTConfig,
    device: str,
) -> Any:
    from comet_ml import Experiment

    api_key = args.comet_api_key or os.environ.get("COMET_API_KEY")
    workspace = args.comet_workspace or os.environ.get("COMET_WORKSPACE")
    if not api_key:
        raise RuntimeError(
            "Comet API key is missing. Pass --comet-api-key or set COMET_API_KEY."
        )

    exp = Experiment(
        api_key=api_key,
        project_name=args.comet_project,
        workspace=workspace,
        auto_output_logging=False,
        auto_metric_logging=False,
        auto_param_logging=False,
        auto_histogram_weight_logging=False,
        auto_histogram_gradient_logging=False,
        auto_histogram_activation_logging=False,
        parse_args=False,
    )
    exp.set_name(model_spec.name)
    exp.add_tag("Monarch")
    exp.add_tag("wikitext103")
    exp.add_tag(model_spec.name)
    exp.log_parameters(
        {
            "dataset_name": DATASET_NAME,
            "dataset_config": DATASET_CONFIG,
            "dataset_split": DATASET_SPLIT,
            "total_tokens": TOTAL_TOKENS,
            "train_tokens": TRAIN_TOKENS,
            "test_tokens": TEST_TOKENS,
            "batch_size": BATCH_SIZE,
            "block_size": BLOCK_SIZE,
            "grad_accum": GRAD_ACCUM,
            "num_epochs": args.epochs,
            "eval_batches": args.eval_batches,
            "warmup_steps": args.warmup_steps,
            "min_lr_ratio": args.min_lr_ratio,
            "adam_lr": ADAM_LR,
            "muon_lr": MUON_LR,
            "muon_momentum": MUON_MOMENTUM,
            "weight_decay": WEIGHT_DECAY,
            "grad_clip": GRAD_CLIP,
            "device": device,
            "precision_mode": "float32",
            **asdict(model_cfg),
        }
    )
    return exp


def run_single_experiment(
    args: argparse.Namespace,
    model_spec: ModelSpec,
    metadata: dict,
) -> dict:
    device = args.device
    run_dir = OUTPUT_ROOT / model_spec.name
    run_dir.mkdir(parents=True, exist_ok=True)

    train_loader, _ = build_loaders(device)
    train_eval_loader, test_eval_loader = build_loaders(device)
    micro_batches_per_epoch = len(train_loader)
    steps_per_epoch = max(1, micro_batches_per_epoch // GRAD_ACCUM)
    total_training_steps = args.epochs * steps_per_epoch
    scheduler_epochs = min(SCHEDULER_EPOCHS, args.epochs)
    scheduler_steps = max(1, scheduler_epochs * steps_per_epoch)    

    set_seed(SEED, device)
    model, optimizer_muon, optimizer_adam, cfg = build_model_and_optimizers(model_spec, device)
    exp = create_comet_experiment(args, model_spec, cfg, device)
    total_params = count_parameters(model)
    print(f"[{model_spec.name}] total parameters: {total_params:,}")
    exp.log_parameter("total_parameters", total_params)
    exp.log_metric("total_parameters", total_params, step=0, epoch=0)

    history = {
        "model_name": model_spec.name,
        "dataset_metadata": metadata,
        "epochs": [],
        "train_loss": [],
        "test_loss": [],
        "mean_train_step_loss": [],
        "epoch_time_sec": [],
        "wallclock_time_sec": [],
        "epoch_optimizer_time_sec": [],
        "optimizer_time_sec": [],
        "global_step": [],
        "total_parameters": total_params,
        "steps_per_epoch": steps_per_epoch,
        "micro_batches_per_epoch": micro_batches_per_epoch,
        "config": {
            "batch_size": BATCH_SIZE,
            "block_size": BLOCK_SIZE,
            "grad_accum": GRAD_ACCUM,
            "num_epochs": args.epochs,
            "eval_batches": args.eval_batches,
            "warmup_steps": args.warmup_steps,
            "min_lr_ratio": args.min_lr_ratio,
            "adam_lr": ADAM_LR,
            "muon_lr": MUON_LR,
            "muon_momentum": MUON_MOMENTUM,
            "weight_decay": WEIGHT_DECAY,
            "grad_clip": GRAD_CLIP,
            **asdict(cfg),
        },
    }

    best_test_loss = float("inf")
    global_step = 0
    experiment_started_at = time.time()
    cumulative_optimizer_time_sec = 0.0

    try:
        for epoch in range(1, args.epochs + 1):
            model.train()
            train_loader.reset(0)

            epoch_loss_sum = 0.0
            epoch_steps = 0
            torch.cuda.synchronize()
            epoch_started_at = time.time()
            epoch_optimizer_time_sec = 0.0

            for step_in_epoch in range(1, steps_per_epoch + 1):
                optimizer_muon.zero_grad()
                optimizer_adam.zero_grad()

                step_loss = 0.0
                for _ in range(GRAD_ACCUM):
                    x, y = train_loader.next_batch()
                    _, loss = model(x, y)
                    loss = loss / GRAD_ACCUM
                    loss.backward()

                    step_loss += float(loss.detach().cpu())

                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                lr_scale = get_lr_scale(
                    step=global_step,
                    warmup_steps=args.warmup_steps,
                    scheduler_steps=scheduler_steps,
                    min_lr_ratio=args.min_lr_ratio,
                )
                optimizer_adam.param_groups[0]["lr"] = ADAM_LR * lr_scale
                optimizer_adam.param_groups[1]["lr"] = (ADAM_LR / math.sqrt(cfg.n_embd)) * lr_scale
                optimizer_muon.param_groups[0]["lr"] = MUON_LR * lr_scale
                torch.cuda.synchronize()
                optimizer_started_at = time.time()
                optimizer_adam.step()
                optimizer_muon.step()
                torch.cuda.synchronize()
                optimizer_elapsed_sec = time.time() - optimizer_started_at
                epoch_optimizer_time_sec += optimizer_elapsed_sec
                cumulative_optimizer_time_sec += optimizer_elapsed_sec

                global_step += 1
                epoch_steps += 1
                epoch_loss_sum += step_loss

                should_log = (
                    global_step % args.log_every_steps == 0
                    or step_in_epoch == steps_per_epoch
                )
                if should_log:
                    metrics = estimate_loss(
                        model,
                        train_eval_loader=train_eval_loader,
                        test_eval_loader=test_eval_loader,
                        eval_batches=args.eval_batches,
                    )
                    mean_train_step_loss = epoch_loss_sum / max(1, epoch_steps)
                    torch.cuda.synchronize()
                    epoch_time = time.time() - epoch_started_at
                    wallclock_time_sec = time.time() - experiment_started_at
                    train_perplexity = loss_to_perplexity(metrics["train"])
                    test_perplexity = loss_to_perplexity(metrics["test"])

                    history["epochs"].append(epoch)
                    history["train_loss"].append(metrics["train"])
                    history["test_loss"].append(metrics["test"])
                    history["mean_train_step_loss"].append(mean_train_step_loss)
                    history["epoch_time_sec"].append(epoch_time)
                    history["wallclock_time_sec"].append(wallclock_time_sec)
                    history["epoch_optimizer_time_sec"].append(epoch_optimizer_time_sec)
                    history["optimizer_time_sec"].append(cumulative_optimizer_time_sec)
                    history["global_step"].append(global_step)

                    exp.log_metric("train_loss", metrics["train"], step=global_step, epoch=epoch)
                    exp.log_metric("test_loss", metrics["test"], step=global_step, epoch=epoch)
                    exp.log_metric(
                        "mean_train_step_loss",
                        mean_train_step_loss,
                        step=global_step,
                        epoch=epoch,
                    )
                    exp.log_metric("epoch_time_sec", epoch_time, step=global_step, epoch=epoch)
                    exp.log_metric("wallclock_time_sec", wallclock_time_sec, step=global_step, epoch=epoch)
                    exp.log_metric(
                        "optimizer_time_sec",
                        cumulative_optimizer_time_sec,
                        step=global_step,
                        epoch=epoch,
                    )
                    exp.log_metric(
                        "adam_lr_regular",
                        optimizer_adam.param_groups[0]["lr"],
                        step=global_step,
                        epoch=epoch,
                    )
                    exp.log_metric(
                        "adam_lr_unembed",
                        optimizer_adam.param_groups[1]["lr"],
                        step=global_step,
                        epoch=epoch,
                    )
                    exp.log_metric("train_perplexity", train_perplexity, step=global_step, epoch=epoch)
                    exp.log_metric("test_perplexity", test_perplexity, step=global_step, epoch=epoch)
                    exp.log_metric(
                        "train_loss_vs_wallclock_time",
                        metrics["train"],
                        step=seconds_to_step(wallclock_time_sec),
                        epoch=epoch,
                    )
                    exp.log_metric(
                        "test_loss_vs_wallclock_time",
                        metrics["test"],
                        step=seconds_to_step(wallclock_time_sec),
                        epoch=epoch,
                    )
                    exp.log_metric(
                        "train_loss_vs_optimizer_time",
                        metrics["train"],
                        step=seconds_to_step(cumulative_optimizer_time_sec),
                        epoch=epoch,
                    )
                    exp.log_metric(
                        "test_loss_vs_optimizer_time",
                        metrics["test"],
                        step=seconds_to_step(cumulative_optimizer_time_sec),
                        epoch=epoch,
                    )
                    exp.log_metric(
                        "train_perplexity_vs_optimizer_time",
                        train_perplexity,
                        step=seconds_to_step(cumulative_optimizer_time_sec),
                        epoch=epoch,
                    )
                    exp.log_metric(
                        "test_perplexity_vs_optimizer_time",
                        test_perplexity,
                        step=seconds_to_step(cumulative_optimizer_time_sec),
                        epoch=epoch,
                    )

                    print(
                        f"[{model_spec.name}] "
                        f"epoch {epoch:03d}/{args.epochs:03d} | "
                        f"iter {step_in_epoch:05d}/{steps_per_epoch:05d} | "
                        f"global_step {global_step:06d} | "
                        f"train_loss {metrics['train']:.4f} | "
                        f"test_loss {metrics['test']:.4f} | "
                        f"train_ppl {train_perplexity:.2f} | "
                        f"test_ppl {test_perplexity:.2f} | "
                        f"lr {optimizer_adam.param_groups[0]['lr']:.6e} | "
                        f"mean_train_step_loss {mean_train_step_loss:.4f} | "
                        f"epoch_time {epoch_time / 60:.2f} min | "
                        f"wallclock {wallclock_time_sec / 60:.2f} min | "
                        f"optimizer_time {cumulative_optimizer_time_sec:.2f} s"
                    )

                    if metrics["test"] < best_test_loss:
                        best_test_loss = metrics["test"]
                        checkpoint = clone_checkpoint_to_cpu(
                            {
                                "model_name": model_spec.name,
                                "epoch": epoch,
                                "global_step": global_step,
                                "best_test_loss": best_test_loss,
                                "history": history,
                                "model_state_dict": model.state_dict(),
                                "optimizer_muon_state_dict": optimizer_muon.state_dict(),
                                "optimizer_adam_state_dict": optimizer_adam.state_dict(),
                                "gpt_config": asdict(cfg),
                            }
                        )
                        torch.save(checkpoint, run_dir / "best_model.pt")

                    save_history(history, run_dir)

        return history
    finally:
        exp.end()
        del model
        gc.collect()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run AdamW GPT experiments and log to Comet.")
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--eval-batches", type=int, default=EVAL_BATCHES)
    parser.add_argument("--log-every-steps", type=int, default=LOG_EVERY_STEPS)
    parser.add_argument("--warmup-steps", type=int, default=WARMUP_STEPS)
    parser.add_argument("--min-lr-ratio", type=float, default=MIN_LR_RATIO)
    parser.add_argument("--device", choices=["cpu", "cuda"], default=DEFAULT_DEVICE)
    parser.add_argument("--comet-api-key", type=str, default=None)
    parser.add_argument("--comet-workspace", type=str, default=None)
    parser.add_argument("--comet-project", type=str, default="adamwexperiments")
    args = parser.parse_args()
    if args.log_every_steps < 1:
        parser.error("--log-every-steps must be >= 1")
    if args.warmup_steps < 0:
        parser.error("--warmup-steps must be >= 0")
    if not 0.0 <= args.min_lr_ratio <= 1.0:
        parser.error("--min-lr-ratio must be in [0, 1]")
    return args


def main() -> None:
    args = parse_args()
    args.device = resolve_device(args.device)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    print(f"Device: {args.device}")
    print("Precision mode: float32")
    if args.device == "cuda":
        props = torch.cuda.get_device_properties(0)
        print(f"GPU: {props.name}")
        print(f"VRAM: {props.total_memory / 1024**3:.2f} GB")

    set_seed(SEED, args.device)
    metadata = load_local_wikitext_metadata()

    all_results = {}
    for model_spec in MODEL_SPECS:
        all_results[model_spec.name] = run_single_experiment(args, model_spec, metadata)

    summary_path = OUTPUT_ROOT / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"Finished all experiments. Summary saved to: {summary_path.resolve()}")


if __name__ == "__main__":
    main()
