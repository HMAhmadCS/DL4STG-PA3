"""Q6: comparison table, predicted-vs-actual plots and learned-graph analysis."""
import json
import os
from collections import deque

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm

from data_utils import ADJ_PATH, META_PATH, load_processed, split_bounds
from models import row_normalize

RES = "outputs/results"
FIG = "outputs/figures"
MODELS = [("temporal", "Temporal-only", "None"), ("fixed", "STGNN", "Fixed"),
          ("adaptive", "STGNN", "Adaptive"), ("hybrid", "STGNN", "Hybrid")]
COLORS = {"temporal": "tab:gray", "fixed": "tab:blue", "adaptive": "tab:orange", "hybrid": "tab:green"}


def load_metrics():
    return {k: json.load(open(os.path.join(RES, f"metrics_{k}.json"))) for k, *_ in MODELS
            if os.path.exists(os.path.join(RES, f"metrics_{k}.json"))}


def tables(met):
    rows = []
    for k, name, adj in MODELS:
        if k not in met:
            continue
        m = met[k]
        rows.append(f"{name} & {adj} & {m['val']['mae']:.2f} & {m['val']['rmse']:.2f} & "
                    f"{m['test']['mae']:.2f} & {m['test']['rmse']:.2f} & {m['epochs_run']} ({m['best_epoch']}) & "
                    f"{m['train_time_s'] / 60:.1f} & {m['time_per_epoch_s']:.0f} \\\\")
    with open(os.path.join(RES, "table_main.tex"), "w") as f:
        f.write("\n".join(rows) + "\n")
    rows = []
    for k, name, adj in MODELS:
        if k not in met:
            continue
        t = met[k]["test"]
        cells = " & ".join(f"{a:.2f} / {r:.2f}" for a, r in zip(t["mae_h"], t["rmse_h"]))
        rows.append(f"{name} ({adj}) & {cells} \\\\")
    with open(os.path.join(RES, "table_horizon.tex"), "w") as f:
        f.write("\n".join(rows) + "\n")
    for k in met:
        m = met[k]
        print(f"{k:9s} val {m['val']['mae']:.2f}/{m['val']['rmse']:.2f} test {m['test']['mae']:.2f}/"
              f"{m['test']['rmse']:.2f} epochs {m['epochs_run']} best {m['best_epoch']} "
              f"time {m['train_time_s']:.0f}s" + (f" alpha {m['alpha']:.3f}" if "alpha" in m else ""))


def curves(met):
    fig, ax = plt.subplots(figsize=(6, 3.4))
    for k, name, adj in MODELS:
        if k in met:
            h = met[k]["history"]
            ax.plot([r["epoch"] for r in h], [r["val_mae"] for r in h], marker="o", ms=3,
                    color=COLORS[k], label=f"{name} ({adj})")
    ax.set_xlabel("epoch")
    ax.set_ylabel("validation MAE (veh / 15 min)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "q6_val_curves.pdf"))
    plt.close(fig)


def pred_plots(met, start="2021-04-13", days=3):
    meta = pd.read_csv(META_PATH)
    preds = {k: np.load(os.path.join(RES, f"pred_{k}.npz")) for k in met}
    any_p = next(iter(preds.values()))
    t0 = pd.Timestamp(str(any_p["first_target_time"]))
    sensors = any_p["sensors"]
    for h in (0, 3):
        # window i predicts time t0 + i*15min + h*15min at horizon h
        s = int((pd.Timestamp(start) - t0) / pd.Timedelta("15min")) - h
        sl = slice(s, s + days * 96)
        tt = pd.date_range(pd.Timestamp(start), periods=days * 96, freq="15min")
        fig, axes = plt.subplots(len(sensors), 1, figsize=(11, 2.5 * len(sensors)), sharex=True)
        for a, (ax, j) in enumerate(zip(axes, sensors)):
            true = any_p["true"][sl, h, a]
            mask = any_p["mask"][sl, h, a]
            ax.plot(tt, np.where(mask, true, np.nan), color="black", lw=1.2, label="actual")
            for k, name, adj in MODELS:
                if k in preds:
                    ax.plot(tt, preds[k]["pred"][sl, h, a], color=COLORS[k], lw=0.8, alpha=0.85,
                            label=f"{name} ({adj})")
            ax.set_title(f"Sensor {meta.ID[j]} ({meta.Fwy[j]}, idx {j})", fontsize=9)
            ax.set_ylabel("veh / 15 min")
        axes[0].legend(fontsize=7, ncol=5, loc="upper left")
        axes[-1].set_xlabel(f"test period, {(h + 1) * 15}-minute-ahead forecast")
        fig.tight_layout()
        fig.savefig(os.path.join(FIG, f"q6_pred_vs_actual_h{h + 1}.pdf"))
        plt.close(fig)


def haversine(lat1, lng1, lat2, lng2):
    lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lng2 - lng1) / 2) ** 2
    return 2 * 6371.0 * np.arcsin(np.sqrt(a))


def hop_distances(nz):
    """All-pairs hop counts on the undirected version of the physical graph (BFS)."""
    und = nz | nz.T
    nbrs = [np.flatnonzero(r) for r in und]
    n = len(nbrs)
    d = np.full((n, n), -1, dtype=np.int32)
    for s in range(n):
        d[s, s] = 0
        q = deque([s])
        while q:
            u = q.popleft()
            for v in nbrs[u]:
                if d[s, v] < 0:
                    d[s, v] = d[s, u] + 1
                    q.append(v)
    return d


def graph_analysis(key):
    A = np.load(ADJ_PATH)
    meta = pd.read_csv(META_PATH)
    L = np.load(os.path.join(RES, f"adaptive_adj_{key}.npy"))
    n = len(A)
    phys = A != 0
    np.fill_diagonal(phys, False)
    off = ~np.eye(n, dtype=bool)
    learned = (L > 0) & off  # top-k support (k=20 per row), self excluded
    dist = haversine(meta.Lat.values[:, None], meta.Lng.values[:, None], meta.Lat.values[None], meta.Lng.values[None])
    hops = hop_distances(phys)

    times, flow, mask = load_processed()
    tr = split_bounds(times)["train"]
    x = np.where(mask[tr[0]:tr[1]], flow[tr[0]:tr[1]], np.nan)
    x = pd.DataFrame(x)
    corr = x.corr(min_periods=96 * 30).values  # training data only
    # Residual correlation: remove each sensor's time-of-day mean so the shared daily cycle
    # does not make every pair look correlated.
    tod = (times[tr[0]:tr[1]].hour * 4 + times[tr[0]:tr[1]].minute // 15).values
    resid = x - x.groupby(tod).transform("mean")
    rcorr = resid.corr(min_periods=96 * 30).values

    rng = np.random.default_rng(0)
    rand = np.zeros_like(phys)
    for i in range(n):
        rand[i, rng.choice(np.flatnonzero(off[i]), 20, replace=False)] = True

    # strongest learned connection of every node (self excluded) and global top 1%
    Lo = np.where(off, L, -1)
    top1 = np.zeros_like(phys)
    top1[np.arange(n), Lo.argmax(1)] = True
    thr = np.quantile(L[learned], 0.95)
    strongest = learned & (L >= thr)

    def summ(sel):
        h = hops[sel]
        c = corr[sel]
        return {
            "edges": int(sel.sum()),
            "in_physical_frac": float(phys[sel].mean()),
            "median_km": float(np.median(dist[sel])),
            "mean_km": float(dist[sel].mean()),
            "frac_within_2_hops": float(((h > 0) & (h <= 2)).mean()),
            "frac_unreachable": float((h < 0).mean()),
            "median_hops_reachable": float(np.median(h[h > 0])) if (h > 0).any() else None,
            "mean_corr": float(np.nanmean(c)),
            "mean_resid_corr": float(np.nanmean(rcorr[sel])),
        }

    res = {
        "key": key,
        "self_weight_mean": float(np.diag(L).mean()),
        "self_in_topk_frac": float((np.diag(L) > 0).mean()),
        "physical": summ(phys),
        "learned_topk": summ(learned),
        "learned_top1_per_node": summ(top1),
        "learned_strongest_5pct": summ(strongest),
        "learned_not_physical": summ(learned & ~phys),
        "random_20_per_node": summ(rand),
        "physical_density": float(phys.sum() / off.sum()),
        "spearman_weight_vs_resid_corr": float(pd.Series(L[learned]).corr(pd.Series(rcorr[learned]), method="spearman")),
    }
    # Weight mass the learned graph puts on physical edges
    res["learned_mass_on_physical"] = float((L * phys).sum() / (L * off).sum())
    with open(os.path.join(RES, f"learned_graph_{key}.json"), "w") as f:
        json.dump(res, f, indent=2)

    # Figures: side-by-side heatmaps
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    fixed_n = row_normalize(A)
    for ax, M, title in [(axes[0], fixed_n, "fixed (row-normalised)"),
                         (axes[1], L, f"learned ({key})")]:
        im = ax.imshow(np.ma.masked_equal(M, 0), cmap="viridis", norm=LogNorm(1e-3, 1), interpolation="nearest")
        ax.set_title(title)
        ax.set_xlabel("sensor $j$")
    axes[0].set_ylabel("sensor $i$")
    fig.colorbar(im, ax=axes, label="weight (log scale)")
    fig.savefig(os.path.join(FIG, f"q5_adjacency_{key}.pdf"), dpi=200, bbox_inches="tight")
    plt.close(fig)

    # zoom into a block for readability
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.2))
    blk = slice(500, 600)
    for ax, M, title in [(axes[0], fixed_n, "fixed"), (axes[1], L, f"learned ({key})")]:
        ax.imshow(np.ma.masked_equal(M[blk, blk], 0), cmap="viridis", norm=LogNorm(1e-3, 1),
                  extent=[500, 600, 600, 500], interpolation="nearest")
        ax.set_title(f"{title}, sensors 500-599")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, f"q5_adjacency_zoom_{key}.pdf"), dpi=200)
    plt.close(fig)

    # Distance distribution
    fig, ax = plt.subplots(figsize=(6, 3.4))
    bins = np.linspace(0, 80, 41)
    for sel, lab, c in [(phys, "physical edges", "tab:blue"), (learned, "learned top-20 edges", "tab:orange"),
                        (rand, "random pairs", "tab:gray")]:
        ax.hist(np.clip(dist[sel], 0, 80), bins=bins, density=True, histtype="step", lw=1.6, color=c, label=lab)
    ax.set_xlabel("straight-line distance between sensors (km)")
    ax.set_ylabel("density")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, f"q6_edge_distance_{key}.pdf"))
    plt.close(fig)

    # Map of the strongest learned edges
    fig, ax = plt.subplots(figsize=(6, 6.5))
    ax.scatter(meta.Lng, meta.Lat, s=3, c="lightgray", zorder=1)
    ii, jj = np.nonzero(strongest)
    for i, j in zip(ii, jj):
        c = "tab:blue" if phys[i, j] else "tab:red"
        ax.plot([meta.Lng[i], meta.Lng[j]], [meta.Lat[i], meta.Lat[j]], color=c, lw=0.6, alpha=0.6, zorder=2)
    ax.plot([], [], color="tab:blue", label="also in road graph")
    ax.plot([], [], color="tab:red", label="not in road graph")
    ax.set_title(f"Strongest 5% learned edges ({key})")
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")
    ax.legend(loc="lower left", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, f"q6_learned_edges_map_{key}.pdf"), dpi=200)
    plt.close(fig)

    print(json.dumps(res, indent=1))
    return res


def main():
    os.makedirs(FIG, exist_ok=True)
    met = load_metrics()
    tables(met)
    curves(met)
    pred_plots(met)
    for k in ("adaptive", "hybrid"):
        if os.path.exists(os.path.join(RES, f"adaptive_adj_{k}.npy")):
            graph_analysis(k)


if __name__ == "__main__":
    main()
