"""Simple Monarch-parameterized linear layer.

Generated-by: ChatGPT, based on the paper draft and refactored user scripts.
The layer replaces one dense weight W by two trainable block-diagonal factors
with a fixed shuffle permutation between them.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def blockdiag_linear(x: torch.Tensor, blocks: torch.Tensor) -> torch.Tensor:
    """Apply a block-diagonal linear map stored as [nblocks, out_block, in_block]."""
    nblocks, out_block, in_block = blocks.shape
    x_blocks = x.reshape(*x.shape[:-1], nblocks, in_block)
    y = torch.einsum("...bi,boi->...bo", x_blocks, blocks)
    return y.reshape(*x.shape[:-1], nblocks * out_block)


def monarch_permutation(size: int, nblocks: int, device: torch.device) -> torch.Tensor:
    block = size // nblocks
    return torch.arange(size, device=device).reshape(nblocks, block).T.reshape(-1)


class MonarchLinear(nn.Module):
    """Two-factor Monarch-style linear layer.

    The effective map is approximately P L P^T R. The implementation uses a
    block-diagonal R, a fixed shuffle permutation, and a block-diagonal L.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True, nblocks: int = 2):
        super().__init__()
        if nblocks < 1:
            raise ValueError("nblocks must be >= 1")
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.nblocks = int(nblocks)
        self.in_block = math.ceil(self.in_features / nblocks)
        self.out_block = math.ceil(self.out_features / nblocks)
        self.in_extended = self.in_block * nblocks
        self.out_extended = self.out_block * nblocks
        self.hidden_extended = max(self.in_extended, self.out_extended)
        self.hidden_block = math.ceil(self.hidden_extended / nblocks)
        self.hidden_extended = self.hidden_block * nblocks

        self.R = nn.Parameter(torch.empty(nblocks, self.hidden_block, self.in_block))
        self.L = nn.Parameter(torch.empty(nblocks, self.out_block, self.hidden_block))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for param in (self.R, self.L):
            nn.init.kaiming_uniform_(param, a=math.sqrt(5))
        if self.bias is not None:
            bound = 1 / math.sqrt(self.in_features)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] < self.in_extended:
            x = F.pad(x, (0, self.in_extended - x.shape[-1]))
        y = blockdiag_linear(x, self.R)
        perm = monarch_permutation(self.hidden_extended, self.nblocks, y.device)
        y = y.index_select(-1, perm)
        y = blockdiag_linear(y, self.L)
        y = y[..., : self.out_features]
        if self.bias is not None:
            y = y + self.bias.to(dtype=y.dtype)
        return y
