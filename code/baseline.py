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

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

if TYPE_CHECKING:
    from comet_ml import Experiment


OUTPUT_ROOT = Path("MUON")
DATA_ROOT =  Path("data") / "fineweb"

DATASET_NAME = "fineweb"
DATASET_CONFIG = "fineweb"
DATASET_SPLIT = "train"
ENCODING_NAME = "gpt2"

TOTAL_TOKENS = 310_000_000
TEST_TOKENS = 10_000_000
TRAIN_TOKENS = TOTAL_TOKENS - TEST_TOKENS

BATCH_SIZE = 32
BLOCK_SIZE = 256
GRAD_ACCUM = 2
NUM_EPOCHS = 20
EVAL_BATCHES = 25

ADAM_LR = 1e-3
MUON_LR = 2e-2
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
    ModelSpec(name="gpt_l24_h16_d1024_MUON", n_layer=24, n_head=16, n_embd=1024),
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


def Newton_Schulz_batched(batch: torch.Tensor) -> torch.Tensor:
    out = []
    for index in range(batch.shape[0]):
        out.append(Newton_Schulz(batch[index]))
    return torch.stack(out, dim=0)


class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr: float = MUON_LR, momentum: float = MUON_MOMENTUM):
        defaults = {"lr": lr, "momentum": momentum}
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            for param in group["params"]:
                if param.grad is None:
                    continue

                grad = param.grad
                state = self.state[param]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(grad)

                buffer = state["momentum_buffer"]
                buffer.mul_(momentum).add_(grad)

                if grad.ndim == 3:
                    ortho = Newton_Schulz_batched(buffer)
                    slice_shape = grad[0].shape
                    scale = (slice_shape[0] / slice_shape[1]) ** 0.5
                else:
                    reshaped = buffer.reshape(buffer.shape[0], -1) if buffer.ndim > 2 else buffer
                    ortho = Newton_Schulz(reshaped).reshape_as(buffer)
                    scale = (reshaped.shape[0] / reshaped.shape[1]) ** 0.5

                param.data.add_(ortho, alpha=-lr * scale)
        return loss


@dataclass
class GPTConfig:
    vocab_size: int = 50257
    block_size: int = BLOCK_SIZE
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 672
    dropout: float = 0.1


class CausalSelfAttentionDense(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        self.c_attn = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=True)
        self.c_proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=True)
        self.attn_drop = nn.Dropout(cfg.dropout)
        self.resid_drop = nn.Dropout(cfg.dropout)
        self.n_head = cfg.n_head
        self.n_embd = cfg.n_embd
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(cfg.block_size, cfg.block_size)).view(
                1, 1, cfg.block_size, cfg.block_size
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, channels = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(batch_size, seq_len, self.n_head, channels // self.n_head).transpose(1, 2)
        q = q.view(batch_size, seq_len, self.n_head, channels // self.n_head).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_head, channels // self.n_head).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:, :, :seq_len, :seq_len] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(batch_size, seq_len, channels)
        return self.resid_drop(self.c_proj(y))


class MLPDense(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.c_fc = nn.Linear(cfg.n_embd, 4 * cfg.n_embd, bias=True)
        self.c_proj = nn.Linear(4 * cfg.n_embd, cfg.n_embd, bias=True)
        self.drop = nn.Dropout(cfg.dropout)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.c_proj(self.act(self.c_fc(x))))


class BlockDense(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(cfg.n_embd)
        self.attn = CausalSelfAttentionDense(cfg)
        self.ln_2 = nn.LayerNorm(cfg.n_embd)
        self.mlp = MLPDense(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.transformer = nn.ModuleDict(
            {
                "wte": nn.Embedding(cfg.vocab_size, cfg.n_embd),
                "wpe": nn.Embedding(cfg.block_size, cfg.n_embd),
                "drop": nn.Dropout(cfg.dropout),
                "h": nn.ModuleList([BlockDense(cfg) for _ in range(cfg.n_layer)]),
                "ln_f": nn.LayerNorm(cfg.n_embd),
            }
        )
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.transformer["wte"].weight = self.lm_head.weight
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        _, seq_len = idx.size()
        pos = torch.arange(seq_len, device=idx.device)
        x = self.transformer["drop"](
            self.transformer["wte"](idx) + self.transformer["wpe"](pos)
        )
        for block in self.transformer["h"]:
            x = block(x)
        x = self.transformer["ln_f"](x)
        logits = self.lm_head(x)
        loss = None
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
    exp.add_tag("muon")
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
            epoch_started_at = time.time()
            epoch_optimizer_time_sec = 0.0

            for _ in range(steps_per_epoch):
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

                optimizer_started_at = time.time()
                optimizer_adam.step()
                optimizer_muon.step()
                optimizer_elapsed_sec = time.time() - optimizer_started_at
                epoch_optimizer_time_sec += optimizer_elapsed_sec
                cumulative_optimizer_time_sec += optimizer_elapsed_sec

                global_step += 1
                epoch_steps += 1
                epoch_loss_sum += step_loss

            metrics = estimate_loss(
                model,
                train_eval_loader=train_eval_loader,
                test_eval_loader=test_eval_loader,
                eval_batches=args.eval_batches,
            )
            mean_train_step_loss = epoch_loss_sum / max(1, epoch_steps)
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
                f"train_loss {metrics['train']:.4f} | "
                f"test_loss {metrics['test']:.4f} | "
                f"train_ppl {train_perplexity:.2f} | "
                f"test_ppl {test_perplexity:.2f} | "
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
    parser = argparse.ArgumentParser(description="Run Muon GPT experiments and log to Comet.")
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--eval-batches", type=int, default=EVAL_BATCHES)
    parser.add_argument("--device", choices=["cpu", "cuda"], default=DEFAULT_DEVICE)
    parser.add_argument("--comet-api-key", type=str, default=None)
    parser.add_argument("--comet-workspace", type=str, default=None)
    parser.add_argument("--comet-project", type=str, default="muonexperiments")
    return parser.parse_args()


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
