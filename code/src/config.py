"""Experiment parameters for the Monarch-Muon computational experiment.

Generated-by: ChatGPT, based on the paper draft and refactored user scripts.
This file intentionally centralizes the experimental parameters so that runs are
reproducible and easy to modify.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal


OptimizerName = Literal["adamw", "muon", "monarch_muon", "monarch_dion"]
DataKind = Literal["synthetic", "memmap"]


@dataclass(slots=True)
class ExperimentConfig:
    # ---------------------------------------------------------------------
    # Data parameters
    # ---------------------------------------------------------------------
    data_kind: DataKind = "synthetic"
    data_dir: Path = Path("data")
    vocab_size: int = 50257
    synthetic_train_tokens: int = 32768
    synthetic_val_tokens: int = 8192

    # ---------------------------------------------------------------------
    # Model parameters: compact GPT-2 style model by default
    # ---------------------------------------------------------------------
    block_size: int = 64
    n_layer: int = 2
    n_head: int = 2
    n_embd: int = 128
    dropout: float = 0.1
    monarch_blocks: int = 2

    # ---------------------------------------------------------------------
    # Training parameters
    # ---------------------------------------------------------------------
    batch_size: int = 8
    grad_accum: int = 1
    epochs: int = 2
    max_steps: int | None = None
    eval_batches: int = 5
    log_every_steps: int = 10
    seed: int = 42
    device: str = "cpu"

    # ---------------------------------------------------------------------
    # Optimizer parameters
    # ---------------------------------------------------------------------
    adam_lr: float = 1e-3
    muon_lr: float = 5e-3
    muon_momentum: float = 0.95
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    dion_row_alpha: float = 0.5

    # ---------------------------------------------------------------------
    # Output parameters
    # ---------------------------------------------------------------------
    output_dir: Path = Path("outputs")
    write_tex: bool = True

    def to_jsonable(self) -> dict:
        result = asdict(self)
        for key, value in list(result.items()):
            if isinstance(value, Path):
                result[key] = str(value)
        return result
