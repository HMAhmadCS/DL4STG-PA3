# AI651 Assignment 3 – Spatio-temporal traffic forecasting on LargeST-SD

## Contents

| File | Purpose |
|---|---|
| `q1_aggregation.py` | Q1: sum/mean/max aggregation, injective scalar messages (NumPy checks) |
| `q2_direction.py` | Q2: ring-road direction experiments and heatmaps |
| `prepare_data.py` | Q3(a): extract the 716 San Diego sensors (2020–2021), 5→15 min resampling, missing values |
| `graph_analysis.py` | Q3(b,c): adjacency statistics, heatmap, degree distribution, 1/2-hop neighbourhood |
| `data_utils.py` | Q4(a): chronological splits, train-only normalisation, 8→4 sliding windows |
| `models.py` | Q4/Q5: temporal encoder, graph convolution, forecast head, fixed/adaptive/hybrid adjacency |
| `train.py` | Training and evaluation of one model, checkpointed every epoch (resumable) |
| `run_all.sh` | Trains the four models, then two control models (`adaptive_nomiss`, `temporal_tod`) |
| `analyze.py` | Q6: comparison tables, prediction plots, learned-graph analysis |
| `probe_adjacency.py` | Q6: swaps the trained adaptive model's adjacency at test time |

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv/bin/pip install numpy pandas matplotlib h5py
```

Download the LargeST archive from <https://www.kaggle.com/datasets/liuxu77/largest> (`archive.zip`)
and put `ca_meta.csv` and `ca_rn_adj.npy` in `data/raw/`. `prepare_data.py` extracts
`ca_his_raw_2020.h5` and `ca_his_raw_2021.h5` from the archive one at a time
(`--archive path/to/archive.zip`, default `~/Downloads/archive.zip`), keeps the SD columns and
deletes the 7 GB files again (pass `--keep` to keep them).

## Reproduce

```bash
.venv/bin/python q1_aggregation.py
.venv/bin/python q2_direction.py
.venv/bin/python prepare_data.py
.venv/bin/python graph_analysis.py
./run_all.sh            # temporal, fixed, adaptive, hybrid + 2 controls
.venv/bin/python analyze.py
.venv/bin/python probe_adjacency.py
```

Figures go to `outputs/figures/`, metrics/tables to `outputs/results/`, logs to `outputs/logs/`.

**Resuming.** `train.py` saves `checkpoints/<model>/last.pt` (model, optimiser, epoch counter,
early-stopping state, history, accumulated training time) after every epoch and the best
validation model to `best.pt`. If training is interrupted, run the same command (or
`./run_all.sh`) again: finished models are skipped and the interrupted one continues from the
last completed epoch. The shuffling order of epoch `e` is seeded with `seed + e`, so a resumed
run follows the same batch order as an uninterrupted one.

## Configuration

| Setting | Value |
|---|---|
| Input / output | 8 steps (2 h) → 4 steps (1 h), 15-min flow, 716 sensors |
| Splits | train 2020, val Jan–Mar 2021, test Apr–Dec 2021 (windows never cross splits) |
| Normalisation | global z-score from observed 2020 entries |
| Temporal encoder | Conv1d(1→32,k3) → ReLU → Conv1d(32→32,k3,dil2) → ReLU → Conv1d(32→32,k2) |
| Graph layer | H = ReLU(Â Z W_g) + Z, one layer |
| Head | Linear(32→32) → ReLU → Linear(32→4) |
| Adaptive graph | E1, E2 ∈ R^{716×10}, Softmax(ReLU(E1E2ᵀ)) with top-20 per row before softmax |
| Hybrid | α·Â_fixed + (1−α)·A_adaptive, α = sigmoid(θ) learned, init 0.5 |
| Training | masked MAE loss, Adam lr 1e-3, batch 64, ≤30 epochs, early stopping patience 5, grad clip 5, seed 42, 6 CPU threads |
