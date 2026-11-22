"""Q3(a): extract the 716 San Diego sensors for 2020-2021, resample to 15 min, handle gaps.

The raw yearly files are ~7 GB each. They are extracted from the Kaggle archive one at a
time, only the SD columns are kept, and the big file is deleted again (unless --keep).
"""
import argparse
import os
import subprocess
import warnings

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RAW = "data/raw"
INTERIM = "data/interim"
OUT = "data/processed"
FIG_DIR = "outputs/figures"
YEARS = (2020, 2021)
SHORT_GAP = 4  # gaps up to one hour (4 x 15 min) are forward-filled


def sd_meta():
    meta = pd.read_csv(os.path.join(RAW, "ca_meta.csv"))
    sd = meta[meta.District == 11].reset_index(drop=True)
    assert len(sd) == 716
    return meta, sd


def extract_year(year, sd, archive, keep):
    out = os.path.join(INTERIM, f"sd_5min_{year}.npz")
    if os.path.exists(out):
        return out
    h5 = os.path.join(RAW, f"ca_his_raw_{year}.h5")
    if not os.path.exists(h5):
        subprocess.run(["unzip", "-o", "-q", archive, os.path.basename(h5), "-d", RAW], check=True)
    with h5py.File(h5, "r") as f:
        cols = f["t/axis0"][:].astype(str)
        times = pd.to_datetime(f["t/axis1"][:])
        lo, hi = sd.ID2.min(), sd.ID2.max()
        assert hi - lo + 1 == len(sd)  # SD sensors are a contiguous block of columns
        vals = f["t/block0_values"][:, lo:hi + 1].astype(np.float32)
    # Column order must match the metadata (and hence the adjacency) order.
    assert (cols[lo:hi + 1] == sd.ID.astype(str).values).all()
    os.makedirs(INTERIM, exist_ok=True)
    np.savez(out, flow=vals, times=times.values)
    if not keep:
        os.remove(h5)
    return out


def resample_15min(flow5, times5):
    """Sum of three 5-min vehicle counts -> vehicles per 15 min.

    With 1-2 of 3 readings missing the sum is estimated as 3 x mean of the observed ones;
    with all three missing the bin stays NaN.
    """
    t, n = flow5.shape
    assert t % 3 == 0 and times5[0].minute % 15 == 0
    blocks = flow5.reshape(t // 3, 3, n)
    cnt = np.isfinite(blocks).sum(1)
    tot = np.nansum(blocks, 1)
    flow = np.where(cnt > 0, 3.0 * tot / np.maximum(cnt, 1), np.nan).astype(np.float32)
    return flow, times5[::3]


def fill_missing(flow, times, train_end):
    """Leakage-free imputation of model inputs (targets stay masked).

    1. gaps of <= 1 hour: forward fill (uses only the past);
    2. longer gaps: the sensor's average time-of-day profile from the 2020 training data;
    3. sensors with no 2020 data: the network-wide 2020 time-of-day profile.
    """
    df = pd.DataFrame(flow)
    ffilled = df.ffill(limit=SHORT_GAP).values
    # Only gaps whose total length is <= SHORT_GAP take the forward-filled values.
    isn = np.isnan(flow)
    out = flow.copy()
    for j in range(flow.shape[1]):
        col = isn[:, j]
        if not col.any():
            continue
        starts = np.flatnonzero(col & ~np.r_[False, col[:-1]])
        ends = np.flatnonzero(col & ~np.r_[col[1:], False]) + 1
        for s, e in zip(starts, ends):
            if e - s <= SHORT_GAP and s > 0:
                out[s:e, j] = ffilled[s:e, j]

    tod = (times.hour * 4 + times.minute // 15).values
    train = slice(0, train_end)
    prof = np.full((96, flow.shape[1]), np.nan, dtype=np.float32)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN sensors
        for k in range(96):
            prof[k] = np.nanmean(flow[np.flatnonzero(tod[train] == k)], 0)
        global_prof = np.nanmean(prof, 1, keepdims=True)
    prof = np.where(np.isnan(prof), global_prof, prof)
    rows, cols = np.nonzero(np.isnan(out))
    out[rows, cols] = prof[tod[rows], cols]
    return out


def plot_week(flow, mask, times, sd, sensors, start="2020-03-02"):
    s = times.get_loc(pd.Timestamp(start))
    sl = slice(s, s + 7 * 96)
    fig, axes = plt.subplots(len(sensors), 1, figsize=(11, 2.4 * len(sensors)), sharex=True)
    for ax, j in zip(axes, sensors):
        y = np.where(mask[sl, j], flow[sl, j], np.nan)
        ax.plot(times[sl], y, lw=0.9)
        ax.set_title(f"Sensor {sd.ID[j]} ({sd.Fwy[j]}, idx {j})", fontsize=9)
        ax.set_ylabel("veh / 15 min")
    axes[-1].set_xlabel("time (2-8 Mar 2020)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "q3a_week_three_sensors.pdf"))
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive", default=os.path.expanduser("~/Downloads/archive.zip"))
    ap.add_argument("--keep", action="store_true", help="keep the 7 GB raw .h5 files")
    args = ap.parse_args()

    meta, sd = sd_meta()
    flows, times = [], []
    for y in YEARS:
        d = np.load(extract_year(y, sd, args.archive, args.keep))
        f15, t15 = resample_15min(d["flow"], pd.DatetimeIndex(d["times"]))
        flows.append(f15)
        times.append(t15)
        raw = d["flow"]
        print(f"{y}: 5-min shape {raw.shape}, missing {np.isnan(raw).mean():.2%}, "
              f"sensors with no data {np.isnan(raw).all(0).sum()}")
    flow = np.concatenate(flows)
    times = pd.DatetimeIndex(np.concatenate([t.values for t in times]))
    assert (np.diff(times.values) == np.timedelta64(15, "m")).all()
    mask = np.isfinite(flow)
    train_end = int((times < pd.Timestamp("2021-01-01")).sum())
    print(f"15-min tensor {flow.shape}, missing after resampling {1 - mask.mean():.2%}")
    filled = fill_missing(flow, times, train_end)
    assert np.isfinite(filled).all()

    os.makedirs(OUT, exist_ok=True)
    np.savez_compressed(os.path.join(OUT, "sd_flow_15min.npz"), flow=filled, mask=mask,
                        times=times.values)
    adj = np.load(os.path.join(RAW, "ca_rn_adj.npy"), mmap_mode="r")
    idx = sd.ID2.values
    np.save(os.path.join(OUT, "sd_adj.npy"), np.asarray(adj[np.ix_(idx, idx)], dtype=np.float32))
    sd.to_csv(os.path.join(OUT, "sd_meta.csv"), index=False)

    # Missing-value summary
    per_sensor = 1 - mask.mean(0)
    print(f"sensors >50% missing: {(per_sensor > 0.5).sum()}, fully missing: {(per_sensor == 1).sum()}")
    for name, sl in [("2020", slice(0, train_end)), ("2021", slice(train_end, None))]:
        print(f"  missing {name}: {1 - mask[sl].mean():.2%}")

    os.makedirs(FIG_DIR, exist_ok=True)
    plot_week(filled, mask, times, sd, sensors=[0, 300, 600])


if __name__ == "__main__":
    main()
