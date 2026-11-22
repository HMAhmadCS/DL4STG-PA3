"""Windowing, chronological splits and normalisation for the SD traffic data (Q4a)."""
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

PROCESSED = "data/processed/sd_flow_15min.npz"
ADJ_PATH = "data/processed/sd_adj.npy"
META_PATH = "data/processed/sd_meta.csv"

IN_LEN, OUT_LEN = 8, 4
SPLITS = {
    "train": ("2020-01-01", "2021-01-01"),
    "val": ("2021-01-01", "2021-04-01"),
    "test": ("2021-04-01", "2022-01-01"),
}


def load_processed():
    d = np.load(PROCESSED, allow_pickle=True)
    times = pd.DatetimeIndex(d["times"])
    return times, d["flow"], d["mask"]


def split_bounds(times):
    """Return [start, end) index ranges of each split in the 15-min time axis."""
    out = {}
    for name, (a, b) in SPLITS.items():
        idx = np.where((times >= pd.Timestamp(a)) & (times < pd.Timestamp(b)))[0]
        out[name] = (idx[0], idx[-1] + 1)
    return out


class Scaler:
    """Global z-score fitted on observed training entries only."""

    def __init__(self, flow, mask, train_range):
        s, e = train_range
        obs = flow[s:e][mask[s:e]]
        self.mean, self.std = float(obs.mean()), float(obs.std())

    def transform(self, x):
        return (x - self.mean) / self.std

    def inverse(self, x):
        return x * self.std + self.mean


class WindowDataset(Dataset):
    """Sliding windows that stay inside one split.

    For a prediction time t the input is x_{t-7..t} and the target x_{t+1..t+4};
    every window index lies in [start, end) so no window crosses a split boundary.
    """

    def __init__(self, x_norm, flow, mask, rng):
        s, e = rng
        self.x = torch.from_numpy(x_norm[s:e]).float()
        self.y = torch.from_numpy(flow[s:e]).float()
        self.m = torch.from_numpy(mask[s:e])
        self.n = (e - s) - IN_LEN - OUT_LEN + 1

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        x = self.x[i:i + IN_LEN]                              # (8, N)
        y = self.y[i + IN_LEN:i + IN_LEN + OUT_LEN]           # (4, N), original units
        m = self.m[i + IN_LEN:i + IN_LEN + OUT_LEN]           # observed targets
        return x.unsqueeze(-1), y.unsqueeze(-1), m.unsqueeze(-1)

    def batch(self, idx):
        """Vectorised __getitem__ for a tensor of window indices -> (B, T, N, 1) tensors."""
        ti = idx[:, None] + torch.arange(IN_LEN)
        to = idx[:, None] + IN_LEN + torch.arange(OUT_LEN)
        return self.x[ti].unsqueeze(-1), self.y[to].unsqueeze(-1), self.m[to].unsqueeze(-1)


def build_datasets():
    times, flow, mask = load_processed()
    bounds = split_bounds(times)
    scaler = Scaler(flow, mask, bounds["train"])
    x_norm = scaler.transform(flow).astype(np.float32)
    ds = {k: WindowDataset(x_norm, flow, mask, r) for k, r in bounds.items()}
    return ds, scaler, times, bounds
