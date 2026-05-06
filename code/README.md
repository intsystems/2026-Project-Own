# Reducing the Cost of Muon with Monarch-Parameterized Layers

This repository contains a cleaned and reproducible computational experiment for the paper draft **"Reducing the Cost of Muon with Monarch-Parameterized Layers"**.

The code is organized so that the experiment can be run from a single main file, while the implementation is decomposed into modules for data loading, model construction, Monarch layers, Muon-style optimizers, training, plotting, and report generation.

## What is implemented

The repository compares four optimizer/model variants:

1. `adamw` — dense GPT-style model trained with AdamW.
2. `muon` — dense GPT-style model with Muon on matrix parameters and AdamW on the remaining parameters.
3. `monarch_muon` — Monarch-parameterized GPT-style model with block-wise Muon updates.
4. `monarch_dion` — Dion2-wise / random-submatrix Monarch Muon ablation.

The main paper result should focus on `monarch_muon`. The `monarch_dion` variant is kept as an ablation because in the reported experiments it did not noticeably improve iteration-wise convergence over ordinary block-wise Monarch Muon.

## Repository structure

```text
.
├── run_experiment.py              # single main entry point
├── src/
│   ├── config.py                  # special-purpose experiment parameter section
│   ├── data.py                    # synthetic and memmap token loaders
│   ├── train.py                   # training loop and local metric storage
│   ├── plotting.py                # named paper plots
│   ├── report.py                  # generated .tex report
│   ├── layers/monarch.py          # MonarchLinear layer
│   ├── models/gpt.py              # compact GPT-2 style model
│   └── optim/muon.py              # Muon and Dion2-wise ablation optimizers
├── requirements.txt
├── pyproject.toml
```

## Quick start

The default mode uses a deterministic synthetic token dataset. This makes the repository runnable without downloading Wikitext/FineWeb or any other large dataset.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
PYTHONPATH=src python run_experiment.py --quick --methods adamw,muon,monarch_muon,monarch_dion --output-dir outputs/quick
```

## Running a larger local experiment

```bash
PYTHONPATH=src python run_experiment.py \
  --methods adamw,muon,monarch_muon,monarch_dion \
  --device cuda \
  --epochs 10 \
  --batch-size 64 \
  --block-size 256 \
  --n-layer 4 \
  --n-head 4 \
  --n-embd 384 \
  --output-dir outputs/gpt_l4_h4_d384
```

## Using pre-tokenized data

For a real dataset, prepare two uint16 token files:

```text
data/train.bin
data/test.bin
```

Then run:

```bash
PYTHONPATH=src python run_experiment.py \
  --data-kind memmap \
  --data-dir data \
  --methods adamw,muon,monarch_muon,monarch_dion \
  --epochs 10 \
  --output-dir outputs/wikitext_run
```

## Stored outputs

Each run writes results locally:

```text
outputs/<run_name>/
├── experiment_config.json
├── figures.json
├── figures/
│   ├── train_loss_vs_opt_time.png
│   ├── perplexity_vs_time.png
│   ├── loss_vs_iteration.png
│   └── dion2_monarch_vs_monarch.png
├── report/results_report.tex
└── <method>/
    ├── config.json
    ├── model_config.json
    ├── history.json
    ├── history.csv
    └── best_model.pt
```

The plot names are chosen to match the paper draft:

- `train_loss_vs_opt_time.png`
- `perplexity_vs_time.png`
- `dion2_monarch_vs_monarch.png`

The generated report is written to `outputs/<run_name>/report/results_report.tex`. You can request PDF compilation with:

```bash
PYTHONPATH=src python run_experiment.py --quick --compile-tex
```

This requires `pdflatex` to be installed.

The tests are intentionally small and verify that Newton--Schulz preserves tensor shapes and that one Monarch-Muon smoke training step stores results.
