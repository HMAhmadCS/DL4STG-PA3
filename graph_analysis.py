"""Q3(b), Q3(c): properties of the SD road-distance adjacency and one sensor's neighbourhood."""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm

from data_utils import ADJ_PATH, META_PATH

FIG_DIR = "outputs/figures"
RES_DIR = "outputs/results"


def main(sensor=None):
    A = np.load(ADJ_PATH)
    meta = pd.read_csv(META_PATH)
    n = A.shape[0]
    off = A.copy()
    np.fill_diagonal(off, 0)
    nz = off != 0
    sym_pattern = (nz == nz.T)
    stats = {
        "nodes": n,
        "nonzero_entries_incl_diag": int((A != 0).sum()),
        "self_loops": int((np.diag(A) != 0).sum()),
        "directed_edges_excl_diag": int(nz.sum()),
        "symmetric_values": bool(np.allclose(A, A.T)),
        "symmetric_pattern": bool(sym_pattern.all()),
        "reciprocated_edge_fraction": float((nz & nz.T).sum() / nz.sum()),
        "weight_min": float(off[nz].min()),
        "weight_max": float(off[nz].max()),
        "weight_mean": float(off[nz].mean()),
    }
    out_deg = nz.sum(1)
    in_deg = nz.sum(0)
    stats.update({
        "out_degree_mean": float(out_deg.mean()), "out_degree_median": float(np.median(out_deg)),
        "out_degree_max": int(out_deg.max()), "isolated_out": int((out_deg == 0).sum()),
        "in_degree_mean": float(in_deg.mean()), "isolated_in": int((in_deg == 0).sum()),
        "isolated_both": int(((out_deg == 0) & (in_deg == 0)).sum()),
    })
    # Weakly connected components
    und = nz | nz.T
    comp = -np.ones(n, int)
    c = 0
    for s in range(n):
        if comp[s] >= 0:
            continue
        stack = [s]
        comp[s] = c
        while stack:
            u = stack.pop()
            for v in np.flatnonzero(und[u]):
                if comp[v] < 0:
                    comp[v] = c
                    stack.append(v)
        c += 1
    sizes = np.bincount(comp)
    stats["weak_components"] = int(c)
    stats["largest_component"] = int(sizes.max())

    os.makedirs(FIG_DIR, exist_ok=True)
    os.makedirs(RES_DIR, exist_ok=True)

    # Heatmap (log colour scale, zeros in white)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(np.ma.masked_equal(A, 0), cmap="viridis", norm=LogNorm(1e-2, 1), interpolation="nearest")
    ax.set_title("SD road-distance adjacency $A_{SD}$ (716 x 716)")
    ax.set_xlabel("sensor $j$")
    ax.set_ylabel("sensor $i$")
    fig.colorbar(im, ax=ax, label="$A_{ij}$ (log scale)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "q3b_adjacency_heatmap.pdf"), dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 3.4))
    bins = np.arange(0, max(out_deg.max(), in_deg.max()) + 2) - 0.5
    ax.hist(out_deg, bins=bins, alpha=0.7, label="out-degree (row nonzeros)")
    ax.hist(in_deg, bins=bins, alpha=0.5, label="in-degree (column nonzeros)")
    ax.set_xlabel("number of neighbours (self-loop excluded)")
    ax.set_ylabel("number of sensors")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "q3b_degree_distribution.pdf"))
    plt.close(fig)

    # Q3(c): one- and two-hop neighbourhoods (outgoing edges, N(i) = {j : A_ij != 0})
    if sensor is None:
        sensor = int(np.argsort(out_deg)[len(out_deg) // 2 + 200])  # a moderately connected sensor
    one = set(np.flatnonzero(nz[sensor]))
    two = set()
    for j in one:
        two |= set(np.flatnonzero(nz[j]))
    two -= one | {sensor}
    stats["q3c"] = {
        "sensor_index": sensor, "sensor_id": int(meta.ID[sensor]), "fwy": meta.Fwy[sensor],
        "one_hop": [int(meta.ID[j]) for j in sorted(one)],
        "one_hop_weights": [round(float(A[sensor, j]), 3) for j in sorted(one)],
        "one_hop_fwy": [meta.Fwy[j] for j in sorted(one)],
        "two_hop_only": [int(meta.ID[j]) for j in sorted(two)],
        "n_one_hop": len(one), "n_two_hop_only": len(two),
    }

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(meta.Lng, meta.Lat, s=4, c="lightgray", label="other sensors")
    t = sorted(two)
    o = sorted(one)
    ax.scatter(meta.Lng[t], meta.Lat[t], s=18, c="tab:orange", label=f"2-hop only ({len(t)})")
    ax.scatter(meta.Lng[o], meta.Lat[o], s=22, c="tab:blue", label=f"1-hop ({len(o)})")
    ax.scatter(meta.Lng[sensor], meta.Lat[sensor], s=90, c="red", marker="*", label=f"sensor {meta.ID[sensor]}")
    lat, lng = meta.Lat[sensor], meta.Lng[sensor]
    ax.set_xlim(lng - 0.12, lng + 0.12)
    ax.set_ylim(lat - 0.12, lat + 0.12)
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")
    ax.legend(fontsize=8, loc="lower left")
    ax.set_title("One- and two-hop neighbourhood")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "q3c_neighbourhood.pdf"))
    plt.close(fig)

    with open(os.path.join(RES_DIR, "graph_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)
    print(json.dumps({k: v for k, v in stats.items() if k != "q3c"}, indent=2))
    q = stats["q3c"]
    print(f"Q3(c) sensor idx {sensor} (ID {q['sensor_id']}, {q['fwy']}): "
          f"{q['n_one_hop']} one-hop, {q['n_two_hop_only']} two-hop-only neighbours")
    print("  one-hop:", list(zip(q["one_hop"], q["one_hop_fwy"], q["one_hop_weights"])))


if __name__ == "__main__":
    main()
