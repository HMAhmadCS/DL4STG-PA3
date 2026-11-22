"""Train / evaluate one model (Q4, Q5). Resumable: rerun the same command after an interruption.

    python train.py --model temporal|fixed|adaptive|hybrid
"""
import argparse
import json
import os
import platform
import random
import time

import numpy as np
import torch

from data_utils import ADJ_PATH, build_datasets
from models import STModel

PLOT_SENSORS = [0, 300, 600]  # same three sensors as the Q3(a) plot


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def masked_mae(pred, y, m):
    m = m.float()
    return (torch.abs(pred - y) * m).sum() / m.sum().clamp(min=1)


@torch.no_grad()
def evaluate(model, ds, scaler, batch_size=256, keep=None):
    """MAE / RMSE in original units over observed targets, overall and per horizon."""
    model.eval()
    abs_err = torch.zeros(4)
    sq_err = torch.zeros(4)
    cnt = torch.zeros(4)
    kept_pred, kept_true, kept_mask = [], [], []
    for s in range(0, len(ds), batch_size):
        idx = torch.arange(s, min(s + batch_size, len(ds)))
        x, y, m = ds.batch(idx)
        pred = scaler.inverse(model(x))
        mf = m.float()
        err = (pred - y) * mf
        abs_err += err.abs().sum((0, 2, 3))
        sq_err += (err ** 2).sum((0, 2, 3))
        cnt += mf.sum((0, 2, 3))
        if keep is not None:
            kept_pred.append(pred[:, :, keep, 0])
            kept_true.append(y[:, :, keep, 0])
            kept_mask.append(m[:, :, keep, 0])
    res = {
        "mae": float(abs_err.sum() / cnt.sum()),
        "rmse": float(torch.sqrt(sq_err.sum() / cnt.sum())),
        "mae_h": (abs_err / cnt).tolist(),
        "rmse_h": torch.sqrt(sq_err / cnt).tolist(),
    }
    if keep is not None:
        res["kept"] = (torch.cat(kept_pred).numpy(), torch.cat(kept_true).numpy(),
                       torch.cat(kept_mask).numpy())
    return res


def save_atomic(obj, path):
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["temporal", "fixed", "adaptive", "hybrid"])
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--emb", type=int, default=10)
    ap.add_argument("--topk", type=int, default=20)
    ap.add_argument("--alpha", type=float, default=0.5, help="initial hybrid alpha (learned)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--max-batches", type=int, default=0, help="debug: limit batches per epoch")
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    set_seed(args.seed)
    ckpt_dir = os.path.join("checkpoints", args.model)
    res_dir = "outputs/results"
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(res_dir, exist_ok=True)
    last_path = os.path.join(ckpt_dir, "last.pt")
    best_path = os.path.join(ckpt_dir, "best.pt")

    ds, scaler, times, bounds = build_datasets()
    train_ds, val_ds, test_ds = ds["train"], ds["val"], ds["test"]
    num_nodes = train_ds.x.shape[1]
    graph = None if args.model == "temporal" else args.model
    model = STModel(num_nodes, graph=graph, fixed_adj=np.load(ADJ_PATH), hidden=args.hidden,
                    emb_dim=args.emb, topk=args.topk, alpha_init=args.alpha)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model={args.model} params={n_params} train/val/test windows="
          f"{len(train_ds)}/{len(val_ds)}/{len(test_ds)} scaler mean={scaler.mean:.2f} std={scaler.std:.2f}")

    state = {"epoch": 0, "best_val": float("inf"), "best_epoch": -1, "bad_epochs": 0,
             "train_time": 0.0, "history": []}
    if os.path.exists(last_path):
        ck = torch.load(last_path, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        state = ck["state"]
        print(f"resumed from epoch {state['epoch']} (best val MAE {state['best_val']:.3f})")

    while state["epoch"] < args.epochs and state["bad_epochs"] < args.patience:
        ep = state["epoch"]
        model.train()
        g = torch.Generator().manual_seed(args.seed + ep)  # order independent of resumes
        perm = torch.randperm(len(train_ds), generator=g)
        nb = (len(perm) + args.batch - 1) // args.batch
        if args.max_batches:
            nb = min(nb, args.max_batches)
        t0 = time.time()
        tot = 0.0
        for b in range(nb):
            x, y, m = train_ds.batch(perm[b * args.batch:(b + 1) * args.batch])
            pred = model(x)
            loss = masked_mae(pred, scaler.transform(y), m)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tot += loss.item()
        ep_time = time.time() - t0
        val = evaluate(model, val_ds, scaler)
        state["train_time"] += ep_time
        state["epoch"] = ep + 1
        rec = {"epoch": ep + 1, "train_loss": tot / nb, "val_mae": val["mae"], "val_rmse": val["rmse"],
               "epoch_time": ep_time}
        if graph == "hybrid":
            rec["alpha"] = float(model.alpha())
        state["history"].append(rec)
        if val["mae"] < state["best_val"]:
            state.update(best_val=val["mae"], best_epoch=ep + 1, bad_epochs=0)
            save_atomic(model.state_dict(), best_path)
        else:
            state["bad_epochs"] += 1
        save_atomic({"model": model.state_dict(), "opt": opt.state_dict(), "state": state,
                     "args": vars(args)}, last_path)
        print(f"epoch {ep + 1:3d} loss {tot / nb:.4f} val MAE {val['mae']:.3f} RMSE {val['rmse']:.3f} "
              f"time {ep_time:.1f}s" + (f" alpha {rec['alpha']:.3f}" if "alpha" in rec else ""), flush=True)

    # Final evaluation with the best checkpoint
    model.load_state_dict(torch.load(best_path))
    val = evaluate(model, val_ds, scaler)
    test = evaluate(model, test_ds, scaler, keep=PLOT_SENSORS)
    pred, true, mask = test.pop("kept")
    test_start = bounds["test"][0] + 8  # time index of the first target (horizon 1)
    np.savez_compressed(os.path.join(res_dir, f"pred_{args.model}.npz"), pred=pred, true=true, mask=mask,
                        sensors=PLOT_SENSORS, first_target_time=str(times[test_start]))
    out = {
        "model": args.model, "params": n_params, "args": vars(args),
        "epochs_run": state["epoch"], "best_epoch": state["best_epoch"],
        "train_time_s": state["train_time"],
        "time_per_epoch_s": state["train_time"] / max(state["epoch"], 1),
        "val": val, "test": test, "history": state["history"],
        "hardware": {"cpu": platform.processor() or platform.machine(), "threads": args.threads,
                     "torch": torch.__version__},
    }
    if graph in ("adaptive", "hybrid"):
        with torch.no_grad():
            np.save(os.path.join(res_dir, f"adaptive_adj_{args.model}.npy"), model.adaptive().numpy())
        if graph == "hybrid":
            out["alpha"] = float(model.alpha())
    with open(os.path.join(res_dir, f"metrics_{args.model}.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"[{args.model}] best epoch {state['best_epoch']} | val MAE {val['mae']:.3f} RMSE {val['rmse']:.3f} | "
          f"test MAE {test['mae']:.3f} RMSE {test['rmse']:.3f} | train time {state['train_time']:.0f}s")


if __name__ == "__main__":
    main()
