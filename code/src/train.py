"""Training and evaluation loop for the computational experiment.

Generated-by: ChatGPT, based on the paper draft and refactored user scripts.
The loop stores all metrics locally: history.json, history.csv, checkpoints,
named plots, and a generated LaTeX report.
"""
from __future__ import annotations

import csv
import os
import time
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn

torch.set_num_threads(int(os.environ.get("TORCH_NUM_THREADS", "1")))

from monarch_muon.config import ExperimentConfig, OptimizerName
from monarch_muon.data import SequentialTokenLoader
from monarch_muon.models.gpt import GPT, GPTConfig
from monarch_muon.optim.muon import Muon, RandomSubmatrixMuon
from monarch_muon.utils import count_parameters, loss_to_perplexity, save_json, set_seed


@torch.no_grad()
def estimate_loss(
    model: GPT,
    train_loader: SequentialTokenLoader,
    val_loader: SequentialTokenLoader,
    eval_batches: int,
) -> dict[str, float]:
    model.eval()
    losses: dict[str, list[float]] = {"train": [], "val": []}
    for split, loader in (("train", train_loader), ("val", val_loader)):
        loader.reset(0)
        for _ in range(eval_batches):
            x, y = loader.next_batch()
            _, loss = model(x, y)
            losses[split].append(float(loss.detach().cpu()))
    model.train()
    return {key: float(np.mean(value)) for key, value in losses.items()}


def split_params_for_muon(model: nn.Module) -> tuple[list[nn.Parameter], list[nn.Parameter]]:
    """Return matrix-like parameters for Muon and the rest for AdamW."""
    muon_params: list[nn.Parameter] = []
    adam_params: list[nn.Parameter] = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "wte" in name or "wpe" in name or "lm_head" in name or param.ndim < 2:
            adam_params.append(param)
        else:
            muon_params.append(param)
    return muon_params, adam_params


def build_model_and_optimizers(method: OptimizerName, cfg: ExperimentConfig) -> tuple[GPT, list[torch.optim.Optimizer], GPTConfig]:
    use_monarch = method in {"monarch_muon", "monarch_dion"}
    model_cfg = GPTConfig(
        vocab_size=cfg.vocab_size,
        block_size=cfg.block_size,
        n_layer=cfg.n_layer,
        n_head=cfg.n_head,
        n_embd=cfg.n_embd,
        dropout=cfg.dropout,
        monarch_blocks=cfg.monarch_blocks,
        use_monarch=use_monarch,
    )
    model = GPT(model_cfg).to(cfg.device)

    if method == "adamw":
        optimizers = [torch.optim.AdamW(model.parameters(), lr=cfg.adam_lr, weight_decay=cfg.weight_decay)]
    else:
        muon_params, adam_params = split_params_for_muon(model)
        if method == "monarch_dion":
            muon = RandomSubmatrixMuon(
                muon_params,
                lr=cfg.muon_lr,
                momentum=cfg.muon_momentum,
                row_alpha=cfg.dion_row_alpha,
            )
        else:
            muon = Muon(muon_params, lr=cfg.muon_lr, momentum=cfg.muon_momentum)
        adam = torch.optim.AdamW(adam_params, lr=cfg.adam_lr, weight_decay=cfg.weight_decay)
        optimizers = [muon, adam]
    return model, optimizers, model_cfg


def zero_grad(optimizers: Iterable[torch.optim.Optimizer]) -> None:
    for optimizer in optimizers:
        optimizer.zero_grad(set_to_none=True)


def optimizer_step(optimizers: Iterable[torch.optim.Optimizer]) -> float:
    start = time.perf_counter()
    for optimizer in optimizers:
        optimizer.step()
    return time.perf_counter() - start


def save_history_files(history: list[dict], run_dir: Path) -> None:
    save_json(history, run_dir / "history.json")
    if not history:
        return
    with (run_dir / "history.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)


def run_one_method(
    method: OptimizerName,
    cfg: ExperimentConfig,
    train_loader: SequentialTokenLoader,
    val_loader: SequentialTokenLoader,
) -> list[dict]:
    """Run one optimizer/model variant and store all raw results."""
    run_dir = cfg.output_dir / method
    run_dir.mkdir(parents=True, exist_ok=True)
    set_seed(cfg.seed, cfg.device)

    model, optimizers, model_cfg = build_model_and_optimizers(method, cfg)
    save_json(cfg.to_jsonable(), run_dir / "config.json")
    save_json(asdict(model_cfg), run_dir / "model_config.json")

    micro_batches_per_epoch = len(train_loader)
    steps_per_epoch = max(1, micro_batches_per_epoch // cfg.grad_accum)
    total_steps = cfg.epochs * steps_per_epoch
    if cfg.max_steps is not None:
        total_steps = min(total_steps, cfg.max_steps)

    history: list[dict] = []
    global_step = 0
    optimizer_time = 0.0
    wall_start = time.perf_counter()
    best_val_loss = float("inf")

    train_loader.reset(0)
    zero_grad(optimizers)
    while global_step < total_steps:
        running_loss = 0.0
        for _ in range(cfg.grad_accum):
            x, y = train_loader.next_batch()
            _, loss = model(x, y)
            (loss / cfg.grad_accum).backward()
            running_loss += float(loss.detach().cpu())

        if cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer_time += optimizer_step(optimizers)
        zero_grad(optimizers)
        global_step += 1

        should_log = (
            global_step == 1
            or global_step == total_steps
            or global_step % cfg.log_every_steps == 0
        )
        if should_log:
            metrics = estimate_loss(model, train_loader, val_loader, cfg.eval_batches)
            row = {
                "method": method,
                "global_step": global_step,
                "epoch_float": global_step / max(1, steps_per_epoch),
                "train_loss": metrics["train"],
                "val_loss": metrics["val"],
                "train_perplexity": loss_to_perplexity(metrics["train"]),
                "val_perplexity": loss_to_perplexity(metrics["val"]),
                "mean_step_loss": running_loss / cfg.grad_accum,
                "optimizer_time_sec": optimizer_time,
                "wallclock_sec": time.perf_counter() - wall_start,
                "n_params": count_parameters(model),
            }
            history.append(row)
            save_history_files(history, run_dir)
            print(
                f"[{method}] step {global_step:05d}/{total_steps:05d} | "
                f"train {row['train_loss']:.4f} | val {row['val_loss']:.4f} | "
                f"val_ppl {row['val_perplexity']:.2f} | opt_time {optimizer_time:.3f}s"
            )
            if metrics["val"] < best_val_loss:
                best_val_loss = metrics["val"]
                torch.save(
                    {
                        "method": method,
                        "global_step": global_step,
                        "best_val_loss": best_val_loss,
                        "model_state_dict": model.state_dict(),
                        "model_config": asdict(model_cfg),
                    },
                    run_dir / "best_model.pt",
                )

    save_history_files(history, run_dir)
    return history
