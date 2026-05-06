"""Muon-style optimizers used in the computational experiment.

Generated-by: ChatGPT, based on the paper draft and refactored user scripts.
The Newton--Schulz constants match the constants used in the original scripts.
"""
from __future__ import annotations

import math
from collections import defaultdict
from typing import Iterable

import torch


def newton_schulz(matrix: torch.Tensor, steps: int = 5, eps: float = 1e-7) -> torch.Tensor:
    """Approximate the polar factor of a matrix with Newton--Schulz iterations."""
    x = matrix / (matrix.norm(p="fro") + eps)
    transposed = False
    if x.shape[0] > x.shape[1]:
        x = x.T
        transposed = True
    for _ in range(steps):
        xx_t = x @ x.T
        x = 3.4445 * x - 4.7750 * xx_t @ x + 2.0315 * (xx_t @ xx_t) @ x
    return x.T if transposed else x


def newton_schulz_batched(batch: torch.Tensor, steps: int = 5, eps: float = 1e-7) -> torch.Tensor:
    """Batched Newton--Schulz for tensors shaped [nblocks, rows, cols]."""
    x = batch / (batch.norm(p="fro", dim=(-2, -1), keepdim=True) + eps)
    transposed = x.shape[-2] > x.shape[-1]
    if transposed:
        x = x.transpose(-2, -1)
    for _ in range(steps):
        xx_t = x @ x.transpose(-2, -1)
        x = 3.4445 * x - 4.7750 * xx_t @ x + 2.0315 * (xx_t @ xx_t) @ x
    return x.transpose(-2, -1) if transposed else x


class Muon(torch.optim.Optimizer):
    """Muon optimizer for matrix-valued parameters.

    3D tensors are treated as a stack of independent block matrices. Blocks with
    the same shape are merged into one batched Newton--Schulz call to reduce
    Python overhead.
    """

    def __init__(self, params: Iterable[torch.nn.Parameter], lr: float, momentum: float = 0.95):
        super().__init__(params, defaults={"lr": lr, "momentum": momentum})

    @torch.no_grad()
    def _momentum_buffer(self, param: torch.nn.Parameter) -> torch.Tensor:
        state = self.state[param]
        if "momentum_buffer" not in state:
            state["momentum_buffer"] = torch.zeros_like(param.grad)
        return state["momentum_buffer"]

    @torch.no_grad()
    def _step_single(self, param: torch.nn.Parameter, lr: float, momentum: float) -> None:
        grad = param.grad
        if grad is None:
            return
        buffer = self._momentum_buffer(param)
        buffer.mul_(momentum).add_(grad)
        matrix = buffer.reshape(buffer.shape[0], -1) if buffer.ndim > 2 else buffer
        ortho = newton_schulz(matrix).reshape_as(buffer)
        scale = math.sqrt(matrix.shape[0] / matrix.shape[1])
        param.add_(ortho, alpha=-lr * scale)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            grouped_3d: dict[tuple[torch.device, torch.dtype, int, int], list[tuple[torch.nn.Parameter, torch.Tensor]]] = defaultdict(list)

            for param in group["params"]:
                if param.grad is None:
                    continue
                if param.grad.ndim == 3:
                    buffer = self._momentum_buffer(param)
                    buffer.mul_(momentum).add_(param.grad)
                    key = (buffer.device, buffer.dtype, buffer.shape[1], buffer.shape[2])
                    grouped_3d[key].append((param, buffer))
                else:
                    self._step_single(param, lr, momentum)

            for (_, _, rows, cols), items in grouped_3d.items():
                merged = torch.cat([buf for _, buf in items], dim=0)
                ortho_merged = newton_schulz_batched(merged)
                scale = math.sqrt(rows / cols)
                offset = 0
                for param, buffer in items:
                    nblocks = buffer.shape[0]
                    param.add_(ortho_merged[offset : offset + nblocks], alpha=-lr * scale)
                    offset += nblocks
        return loss


class RandomSubmatrixMuon(Muon):
    """Dion2-inspired ablation.

    For 3D block tensors, this optimizer orthogonalizes a random fraction of the
    flattened block rows and scatters the result back. It is intentionally kept
    as an ablation because the paper experiments show no clear convergence gain
    over the simpler block-wise Muon update.
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float,
        momentum: float = 0.95,
        row_alpha: float = 0.5,
    ):
        if not 0.0 < row_alpha <= 1.0:
            raise ValueError("row_alpha must be in (0, 1]")
        super().__init__(params, lr=lr, momentum=momentum)
        for group in self.param_groups:
            group["row_alpha"] = row_alpha

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            row_alpha = group["row_alpha"]
            for param in group["params"]:
                if param.grad is None:
                    continue
                buffer = self._momentum_buffer(param)
                buffer.mul_(momentum).add_(param.grad)

                if buffer.ndim == 3:
                    nblocks, rows, cols = buffer.shape
                    flat = buffer.reshape(nblocks * rows, cols)
                    n_selected = max(1, math.ceil(row_alpha * flat.shape[0]))
                    idx = torch.randperm(flat.shape[0], device=flat.device)[:n_selected]
                    selected = flat[idx]
                    update = torch.zeros_like(flat)
                    update[idx] = newton_schulz(selected)
                    scale = math.sqrt(selected.shape[0] / selected.shape[1])
                    ortho = update.reshape_as(buffer)
                else:
                    matrix = buffer.reshape(buffer.shape[0], -1) if buffer.ndim > 2 else buffer
                    ortho = newton_schulz(matrix).reshape_as(buffer)
                    scale = math.sqrt(matrix.shape[0] / matrix.shape[1])
                param.add_(ortho, alpha=-lr * scale)
        return loss
