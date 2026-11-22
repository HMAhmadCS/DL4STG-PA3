"""Q6(b)/(c): test-time adjacency swaps for the trained adaptive model.

The trained encoder, graph layer and head are kept fixed and only the matrix used in the
graph convolution is replaced. This shows which property of the learned graph the model relies on.
"""
import json
import os

import numpy as np
import torch

from data_utils import ADJ_PATH, build_datasets
from models import STModel, row_normalize
from train import evaluate

RES = "outputs/results"


def main(key="adaptive", seed=0):
    torch.set_num_threads(6)
    ds, scaler, _, _ = build_datasets()
    fixed = np.load(ADJ_PATH)
    model = STModel(716, graph=key, fixed_adj=fixed)
    model.load_state_dict(torch.load(os.path.join("checkpoints", key, "best.pt")))
    with torch.no_grad():
        learned = model.adjacency().clone()
    n = learned.shape[0]
    g = torch.Generator().manual_seed(seed)

    rand = torch.zeros(n, n)
    for i in range(n):
        rand[i, torch.randperm(n, generator=g)[:20]] = 1 / 20
    perm = torch.randperm(n, generator=g)
    variants = {
        "learned": learned,
        "learned, rows permuted (wrong node gets each row)": learned[perm],
        "random top-20 graph": rand,
        "uniform 1/N (global mean)": torch.full((n, n), 1 / n),
        "fixed road graph": torch.tensor(row_normalize(fixed), dtype=torch.float32),
        "identity (self only)": torch.eye(n),
    }
    out = {}
    for name, A in variants.items():
        model.adjacency = lambda A=A: A
        r = evaluate(model, ds["test"], scaler)
        out[name] = {"mae": r["mae"], "rmse": r["rmse"]}
        print(f"{name:52s} test MAE {r['mae']:.2f} RMSE {r['rmse']:.2f}", flush=True)
    with open(os.path.join(RES, f"probe_{key}.json"), "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
