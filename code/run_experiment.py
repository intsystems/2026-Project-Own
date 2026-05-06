#!/usr/bin/env python3
"""Main entry point for the Monarch-Muon computational experiment.

Generated-by: ChatGPT, based on the paper draft and refactored user scripts.
This is the only file that needs to be run for the experiment.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch

from monarch_muon.config import ExperimentConfig, OptimizerName
from monarch_muon.data import build_loaders, load_memmap_tokens, make_synthetic_tokens
from monarch_muon.plotting import make_all_plots
from monarch_muon.report import write_tex_report
from monarch_muon.train import run_one_method
from monarch_muon.utils import resolve_device, save_json, set_seed


ALL_METHODS: tuple[OptimizerName, ...] = ("adamw", "muon", "monarch_muon", "monarch_dion")


def parse_methods(value: str) -> list[OptimizerName]:
    items = [item.strip() for item in value.split(",") if item.strip()]
    unknown = sorted(set(items) - set(ALL_METHODS))
    if unknown:
        raise argparse.ArgumentTypeError(f"Unknown methods: {unknown}. Allowed: {ALL_METHODS}")
    return items  # type: ignore[return-value]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Monarch-Muon computational experiment.")

    # ------------------------------------------------------------------
    # Special-purpose parameter section required by the assignment.
    # ------------------------------------------------------------------
    parser.add_argument("--methods", type=parse_methods, default=list(ALL_METHODS), help="Comma-separated methods: adamw,muon,monarch_muon,monarch_dion")
    parser.add_argument("--data-kind", choices=["synthetic", "memmap"], default="synthetic")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=None, help="Optional cap for quick smoke runs.")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--eval-batches", type=int, default=5)
    parser.add_argument("--log-every-steps", type=int, default=10)

    parser.add_argument("--vocab-size", type=int, default=50257)
    parser.add_argument("--synthetic-train-tokens", type=int, default=32768)
    parser.add_argument("--synthetic-val-tokens", type=int, default=8192)
    parser.add_argument("--n-layer", type=int, default=2)
    parser.add_argument("--n-head", type=int, default=2)
    parser.add_argument("--n-embd", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--monarch-blocks", type=int, default=2)

    parser.add_argument("--adam-lr", type=float, default=1e-3)
    parser.add_argument("--muon-lr", type=float, default=5e-3)
    parser.add_argument("--muon-momentum", type=float, default=0.95)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--dion-row-alpha", type=float, default=0.5)

    parser.add_argument("--no-tex", action="store_true", help="Do not write the generated LaTeX report.")
    parser.add_argument("--compile-tex", action="store_true", help="Try to compile the generated report with pdflatex.")
    parser.add_argument("--quick", action="store_true", help="Override settings with a very small smoke-test run.")
    return parser.parse_args()


def config_from_args(args: argparse.Namespace) -> ExperimentConfig:
    if args.quick:
        args.epochs = 1
        args.max_steps = 1
        args.batch_size = 2
        args.block_size = 8
        args.eval_batches = 1
        args.log_every_steps = 1
        args.vocab_size = min(args.vocab_size, 256)
        args.synthetic_train_tokens = 1024
        args.synthetic_val_tokens = 512
        args.n_layer = 1
        args.n_head = 1
        args.n_embd = 16

    return ExperimentConfig(
        data_kind=args.data_kind,
        data_dir=args.data_dir,
        vocab_size=args.vocab_size,
        synthetic_train_tokens=args.synthetic_train_tokens,
        synthetic_val_tokens=args.synthetic_val_tokens,
        block_size=args.block_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        dropout=args.dropout,
        monarch_blocks=args.monarch_blocks,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        epochs=args.epochs,
        max_steps=args.max_steps,
        eval_batches=args.eval_batches,
        log_every_steps=args.log_every_steps,
        seed=args.seed,
        device=resolve_device(args.device),
        adam_lr=args.adam_lr,
        muon_lr=args.muon_lr,
        muon_momentum=args.muon_momentum,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        dion_row_alpha=args.dion_row_alpha,
        output_dir=args.output_dir,
        write_tex=not args.no_tex,
    )


def main() -> None:
    # Small matrices are faster and more stable on CPU with one thread.
    torch.set_num_threads(int(os.environ.get("TORCH_NUM_THREADS", "1")))
    args = parse_args()
    cfg = config_from_args(args)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(cfg.seed, cfg.device)
    save_json(cfg.to_jsonable(), cfg.output_dir / "experiment_config.json")

    if cfg.data_kind == "synthetic":
        tokens = make_synthetic_tokens(
            cfg.synthetic_train_tokens,
            cfg.synthetic_val_tokens,
            cfg.vocab_size,
            cfg.seed,
        )
    else:
        tokens = load_memmap_tokens(cfg.data_dir, cfg.vocab_size)

    for method in args.methods:
        # Build fresh deterministic loaders so every method sees the same batches.
        train_loader, val_loader = build_loaders(tokens, cfg.batch_size, cfg.block_size, cfg.device)
        run_one_method(method, cfg, train_loader, val_loader)

    plot_paths = make_all_plots(cfg.output_dir)
    save_json({key: str(value) for key, value in plot_paths.items()}, cfg.output_dir / "figures.json")

    if cfg.write_tex:
        tex_path = write_tex_report(cfg.output_dir, compile_pdf=args.compile_tex)
        print(f"Generated LaTeX report: {tex_path.resolve()}")

    print(f"Experiment finished. Results are stored in: {cfg.output_dir.resolve()}")


if __name__ == "__main__":
    main()
