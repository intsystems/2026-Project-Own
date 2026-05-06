"""Generate a small LaTeX report from stored experiment results.

Generated-by: ChatGPT, based on the course requirement to write results to a
.tex file and compile it if desired.
"""
from __future__ import annotations

from pathlib import Path
import subprocess

from monarch_muon.plotting import METHOD_LABELS, load_histories


def _last(rows: list[dict], key: str) -> str:
    if not rows:
        return "--"
    value = rows[-1][key]
    return f"{value:.4f}" if isinstance(value, float) else str(value)


def write_tex_report(output_dir: Path, compile_pdf: bool = False) -> Path:
    histories = load_histories(output_dir)
    report_dir = output_dir / "report"
    report_dir.mkdir(parents=True, exist_ok=True)
    tex_path = report_dir / "results_report.tex"
    fig_rel = Path("..") / "figures"

    rows = []
    for method, history in histories.items():
        rows.append(
            f"{METHOD_LABELS.get(method, method)} & {_last(history, 'train_loss')} & "
            f"{_last(history, 'val_loss')} & {_last(history, 'val_perplexity')} & "
            f"{_last(history, 'optimizer_time_sec')} \\\\"
        )

    tex = r"""\documentclass{article}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage[margin=1in]{geometry}
\title{Monarch Muon Computational Experiment}
\author{Generated experiment report}
\date{\today}
\begin{document}
\maketitle

\section{Experiment flow}
The experiment trains the same compact GPT-style model setup under several optimizer variants:
AdamW, dense Muon, Monarch Muon, and a Dion2-wise Monarch ablation. The code records
validation loss, perplexity, wall-clock time, and optimizer-step time.

\section{Final metrics}
\begin{center}
\begin{tabular}{lrrrr}
\toprule
Method & Train loss & Val. loss & Val. PPL & Optimizer time (s) \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{center}

\section{Plots}
\begin{figure}[h]
\centering
\includegraphics[width=0.48\linewidth]{""" + str(fig_rel / "train_loss_vs_opt_time.png") + r"""}
\includegraphics[width=0.48\linewidth]{""" + str(fig_rel / "perplexity_vs_time.png") + r"""}
\caption{Training loss and validation perplexity as functions of optimizer-step time.}
\end{figure}

\begin{figure}[h]
\centering
\includegraphics[width=0.70\linewidth]{""" + str(fig_rel / "dion2_monarch_vs_monarch.png") + r"""}
\caption{Ablation: Dion2-wise Monarch Muon versus block-wise Monarch Muon by iteration.}
\end{figure}

\end{document}
"""
    tex_path.write_text(tex, encoding="utf-8")

    if compile_pdf:
        subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", tex_path.name],
            cwd=report_dir,
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    return tex_path
