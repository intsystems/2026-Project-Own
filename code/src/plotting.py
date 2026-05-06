"""Named plots for the paper/report.

Generated-by: ChatGPT, based on the paper draft and refactored user scripts.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from monarch_muon.utils import load_json


METHOD_LABELS = {
    "adamw": "AdamW",
    "muon": "Muon",
    "monarch_muon": "Monarch Muon",
    "monarch_dion": "Dion2-wise Monarch Muon",
}

# Fixed colors match the paper text: light blue = Monarch Muon,
# gray = standard Muon, purple = AdamW.
METHOD_COLORS = {
    "adamw": "purple",
    "muon": "gray",
    "monarch_muon": "lightskyblue",
    "monarch_dion": "darkorange",
}


def load_histories(output_dir: Path) -> dict[str, list[dict]]:
    histories = {}
    for path in sorted(output_dir.glob("*/history.json")):
        histories[path.parent.name] = load_json(path)
    return histories


def _plot_metric(
    histories: dict[str, list[dict]],
    x_key: str,
    y_key: str,
    output_path: Path,
    xlabel: str,
    ylabel: str,
    methods: list[str] | None = None,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6.6, 4.2), dpi=160)
    methods_to_plot = methods or list(histories.keys())
    for method in methods_to_plot:
        rows = histories.get(method, [])
        if not rows:
            continue
        x = [row[x_key] for row in rows]
        y = [row[y_key] for row in rows]
        plt.plot(
            x,
            y,
            label=METHOD_LABELS.get(method, method),
            color=METHOD_COLORS.get(method),
            linewidth=2.0,
        )
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.25)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def make_all_plots(output_dir: Path) -> dict[str, Path]:
    histories = load_histories(output_dir)
    fig_dir = output_dir / "figures"
    paths = {
        "train_loss_vs_opt_time": fig_dir / "train_loss_vs_opt_time.png",
        "perplexity_vs_time": fig_dir / "perplexity_vs_time.png",
        "loss_vs_iteration": fig_dir / "loss_vs_iteration.png",
        "dion2_monarch_vs_monarch": fig_dir / "dion2_monarch_vs_monarch.png",
    }
    main_methods = ["adamw", "muon", "monarch_muon"]
    _plot_metric(
        histories,
        "optimizer_time_sec",
        "train_loss",
        paths["train_loss_vs_opt_time"],
        "Optimizer-step time, seconds",
        "Training loss",
        main_methods,
    )
    _plot_metric(
        histories,
        "optimizer_time_sec",
        "val_perplexity",
        paths["perplexity_vs_time"],
        "Optimizer-step time, seconds",
        "Validation perplexity",
        main_methods,
    )
    _plot_metric(
        histories,
        "global_step",
        "train_loss",
        paths["loss_vs_iteration"],
        "Optimization iteration",
        "Training loss",
        list(histories.keys()),
    )
    _plot_metric(
        histories,
        "global_step",
        "train_loss",
        paths["dion2_monarch_vs_monarch"],
        "Optimization iteration",
        "Training loss",
        ["monarch_muon", "monarch_dion"],
    )
    return paths
