"""Token loaders for reproducible language-model experiments.

Generated-by: ChatGPT, based on the paper draft and refactored user scripts.
The default synthetic dataset keeps the repository runnable on any machine.
For real experiments, pass --data-kind memmap and provide train.bin/test.bin.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


@dataclass(slots=True)
class TokenArrays:
    train: np.ndarray
    val: np.ndarray
    vocab_size: int


class SequentialTokenLoader:
    """Deterministic sequential token loader.

    This mirrors the sequential memmap loader used in the original scripts, but
    also accepts in-memory arrays for quick smoke tests.
    """

    def __init__(self, tokens: np.ndarray, batch_size: int, block_size: int, device: str):
        self.tokens = tokens.astype(np.uint16, copy=False)
        self.batch_size = int(batch_size)
        self.block_size = int(block_size)
        self.device = device
        self.tokens_per_batch = self.batch_size * self.block_size
        self.max_cursor = len(self.tokens) - (self.tokens_per_batch + 1)
        if self.max_cursor <= 0:
            raise ValueError(
                "Token array is too small for the requested batch_size/block_size."
            )
        self.cursor = 0

    def reset(self, cursor: int = 0) -> None:
        self.cursor = int(cursor)

    def __len__(self) -> int:
        return max(1, (len(self.tokens) - 1) // self.tokens_per_batch)

    def next_batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self.cursor > self.max_cursor:
            self.cursor = 0
        chunk = np.asarray(
            self.tokens[self.cursor : self.cursor + self.tokens_per_batch + 1],
            dtype=np.int64,
        )
        if chunk.shape[0] != self.tokens_per_batch + 1:
            self.cursor = 0
            chunk = np.asarray(
                self.tokens[self.cursor : self.cursor + self.tokens_per_batch + 1],
                dtype=np.int64,
            )
        x = torch.from_numpy(chunk[:-1].reshape(self.batch_size, self.block_size))
        y = torch.from_numpy(chunk[1:].reshape(self.batch_size, self.block_size))
        self.cursor += self.tokens_per_batch
        return x.to(self.device), y.to(self.device)


def make_synthetic_tokens(
    train_tokens: int,
    val_tokens: int,
    vocab_size: int,
    seed: int,
) -> TokenArrays:
    """Create a small deterministic token dataset with local structure.

    The sequence is not meaningful language. It is deliberately simple so that
    the whole computational experiment runs without downloading data.
    """
    rng = np.random.default_rng(seed)
    total = train_tokens + val_tokens + 1
    base = np.arange(total, dtype=np.int64) % min(vocab_size, 997)
    noise = rng.integers(0, min(vocab_size, 127), size=total, dtype=np.int64)
    tokens = ((base * 17 + noise * 3) % vocab_size).astype(np.uint16)
    return TokenArrays(
        train=tokens[:train_tokens],
        val=tokens[train_tokens : train_tokens + val_tokens],
        vocab_size=vocab_size,
    )


def load_memmap_tokens(data_dir: Path, vocab_size: int) -> TokenArrays:
    train_path = data_dir / "train.bin"
    val_path = data_dir / "test.bin"
    if not train_path.exists() or not val_path.exists():
        raise FileNotFoundError(
            f"Expected {train_path} and {val_path}. Use --data-kind synthetic for a smoke run."
        )
    return TokenArrays(
        train=np.memmap(train_path, dtype=np.uint16, mode="r"),
        val=np.memmap(val_path, dtype=np.uint16, mode="r"),
        vocab_size=vocab_size,
    )


def build_loaders(
    tokens: TokenArrays,
    batch_size: int,
    block_size: int,
    device: str,
) -> tuple[SequentialTokenLoader, SequentialTokenLoader]:
    return (
        SequentialTokenLoader(tokens.train, batch_size, block_size, device),
        SequentialTokenLoader(tokens.val, batch_size, block_size, device),
    )
