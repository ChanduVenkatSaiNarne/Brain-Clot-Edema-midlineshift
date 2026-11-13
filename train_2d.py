"""Train the 2D clot (haemorrhage) segmentation model on prepared head-CT slices.

  python prepare_bhsd.py --src data/bhsd --out data/bhsd_256             # once
  python train_2d.py --data data/bhsd_256 --out models/clot               # train + test
  python train_2d.py --data data/bhsd_256 --out /tmp/t --epochs 2         # quick test

Split by SCAN (70/15/15), so no scan contributes slices to both training and test.
Each epoch uses every training slice with bleeding plus the same number of random slices without.
Test metrics (per scan, on the held-out scans):
  * 3D Dice of the bleed (all slices of a scan stacked)          - the main score
  * bleed volume in mL: predicted vs expert, mean absolute error and correlation
  * slice-level detection: sensitivity and specificity for "this slice shows bleeding"
"""

import argparse
import csv
import glob
import json
import math
import os
import random
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from data_ct import case_split
from model import SegModel, predict_slices


def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def load_cases(folder):
    cases = {}
    for p in sorted(glob.glob(os.path.join(folder, "*.npz"))):
        d = np.load(p)
        cases[os.path.basename(p)[:-4]] = {"images": d["images"], "masks": (d["masks"] > 0).astype(np.uint8),
                                           "voxel_ml": float(np.prod(d["pixel_mm"]) * d["spacing"][2] / 1000)}
    if not cases:
        raise SystemExit(f"No .npz files in {folder}. Run prepare_bhsd.py first.")
    return cases


def remove_small(prob, threshold=0.5, min_px=10):
    """Binary mask without specks smaller than min_px pixels."""
    b = (prob > threshold).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(b, connectivity=8)
    keep = np.zeros(n, bool)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= min_px
    return keep[lab].astype(np.uint8)


def augment(x, m):
    n, dev = x.shape[0], x.device
    flip = torch.rand(n, device=dev) < 0.5
    x = torch.where(flip[:, None, None, None], x.flip(-1), x)
    m = torch.where(flip[:, None, None, None], m.flip(-1), m)
    ang = (torch.rand(n, device=dev) - 0.5) * math.radians(30)
    scale = 0.85 + torch.rand(n, device=dev) * 0.30
    shift = (torch.rand(n, 2, device=dev) - 0.5) * 0.16
    cos, sin = torch.cos(ang) / scale, torch.sin(ang) / scale
    theta = torch.stack([torch.stack([cos, -sin, shift[:, 0]], 1), torch.stack([sin, cos, shift[:, 1]], 1)], 1)
    grid = F.affine_grid(theta, x.shape, align_corners=False)
    x = F.grid_sample(x, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
    m = (F.grid_sample(m, grid, mode="bilinear", padding_mode="zeros", align_corners=False) > 0.5).float()
    contrast = 0.9 + torch.rand(n, 1, 1, 1, device=dev) * 0.2
    brightness = (torch.rand(n, 1, 1, 1, device=dev) - 0.5) * 0.1
    return (x * contrast + brightness).clamp(0, 1), m


def loss_fn(logits, m):
    p = torch.sigmoid(logits).flatten(1)
    t = m.flatten(1)
    dice = (2 * (p * t).sum(1) + 1) / (p.sum(1) + t.sum(1) + 1)
    return F.binary_cross_entropy_with_logits(logits, m) + (1 - dice).mean()


def scan_metrics(model, cases, ids, device, batch, min_px):
    """Per-scan 3D Dice, volumes and slice detection for the given scans."""
    rows = []
    tp = fn = tn = fp = 0
    for cid in ids:
        c = cases[cid]
        prob = predict_slices(model, c["images"], device, batch)
        pred = np.stack([remove_small(p, min_px=min_px) for p in prob])
        gt = c["masks"]
        inter, s = (pred & gt).sum(), pred.sum() + gt.sum()
        dice = 1.0 if s == 0 else 2 * inter / s
        rows.append({"case": cid, "dice_3d": float(dice), "has_bleed": bool(gt.any()),
                     "true_ml": float(gt.sum() * c["voxel_ml"]), "pred_ml": float(pred.sum() * c["voxel_ml"])})
        g_any, p_any = gt.reshape(len(gt), -1).any(1), pred.reshape(len(pred), -1).any(1)
        tp += int((g_any & p_any).sum()); fn += int((g_any & ~p_any).sum())
        tn += int((~g_any & ~p_any).sum()); fp += int((~g_any & p_any).sum())
    bleed = [r for r in rows if r["has_bleed"]]
    t = np.array([r["true_ml"] for r in rows]); p = np.array([r["pred_ml"] for r in rows])
    return {
        "dice_3d_mean": float(np.mean([r["dice_3d"] for r in bleed])) if bleed else float("nan"),
        "dice_3d_median": float(np.median([r["dice_3d"] for r in bleed])) if bleed else float("nan"),
        "volume_mae_ml": float(np.abs(t - p).mean()),
        "volume_r": float(np.corrcoef(t, p)[0, 1]) if len(rows) > 2 and t.std() > 0 and p.std() > 0 else float("nan"),
        "slice_sensitivity": tp / max(tp + fn, 1), "slice_specificity": tn / max(tn + fp, 1),
        "n_scans": len(rows), "per_scan": rows,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Folder of .npz files from prepare_bhsd.py")
    ap.add_argument("--out", default="models/clot")
    ap.add_argument("--arch", default="unetplusplus")
    ap.add_argument("--encoder", default="efficientnet-b3")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--neg-ratio", type=float, default=1.0, help="Slices without bleeding per slice with bleeding")
    ap.add_argument("--min-px", type=int, default=10, help="Ignore predicted specks smaller than this")
    args = ap.parse_args()

    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = device.type == "cuda"
    print(f"Device: {'GPU - ' + torch.cuda.get_device_name(0) if amp else 'CPU (slow)'}")
    cases = load_cases(args.data)
    tr, va, te = case_split(list(cases))
    os.makedirs(args.out, exist_ok=True)
    json.dump({"train": tr, "val": va, "test": te}, open(os.path.join(args.out, "split.json"), "w"), indent=1)

    X = np.concatenate([cases[c]["images"] for c in tr])
    M = np.concatenate([cases[c]["masks"] for c in tr])
    pos = np.where(M.reshape(len(M), -1).any(1))[0]
    neg = np.where(~M.reshape(len(M), -1).any(1))[0]
    print(f"{len(cases)} scans -> train {len(tr)}, val {len(va)}, test {len(te)} | "
          f"train slices: {len(pos)} with bleeding, {len(neg)} without")

    model = SegModel(args.arch, args.encoder).to(device)
    print(f"Model: {args.arch} + {args.encoder} (ImageNet) | parameters: {sum(p.numel() for p in model.parameters()):,}")
    per_epoch = len(pos) + min(len(neg), int(args.neg_ratio * len(pos)))
    steps = args.epochs * math.ceil(per_epoch / args.batch)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps, pct_start=0.05)
    scaler = torch.amp.GradScaler(device.type, enabled=amp)

    best_path = os.path.join(args.out, "clot_best.pt")
    hist = os.path.join(args.out, "history.csv")
    with open(hist, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "loss", "val_dice_3d", "val_volume_mae_ml", "val_slice_sens", "val_slice_spec", "lr"])
    best, bad = -1.0, 0
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        idx = np.concatenate([pos, np.random.choice(neg, min(len(neg), int(args.neg_ratio * len(pos))), replace=False)])
        np.random.shuffle(idx)
        total, n = 0.0, 0
        for i in range(0, len(idx), args.batch):
            b = np.sort(idx[i:i + args.batch])
            if len(b) < 2:
                continue
            x = torch.from_numpy(X[b]).float().div(255).to(device)
            m = torch.from_numpy(M[b]).float().unsqueeze(1).to(device)
            x, m = augment(x, m)
            with torch.autocast(device.type, enabled=amp):
                loss = loss_fn(model(x).float(), m)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(opt); scaler.update(); sched.step()
            total += loss.item() * len(b); n += len(b)
        model.eval()
        v = scan_metrics(model, cases, va, device, args.batch, args.min_px)
        lr = opt.param_groups[0]["lr"]
        with open(hist, "a", newline="") as f:
            csv.writer(f).writerow([epoch, total / n, v["dice_3d_mean"], v["volume_mae_ml"],
                                    v["slice_sensitivity"], v["slice_specificity"], lr])
        mark = ""
        if v["dice_3d_mean"] > best:
            best, bad, mark = v["dice_3d_mean"], 0, "  * best"
            torch.save({"model": model.state_dict(), "arch": args.arch, "encoder": args.encoder,
                        "size": int(X.shape[-1]), "min_px": args.min_px, "target": "clot"}, best_path)
        else:
            bad += 1
        print(f"Epoch {epoch:3d} | loss {total / n:.3f} | val 3D Dice {v['dice_3d_mean']:.3f} "
              f"| vol err {v['volume_mae_ml']:.1f} mL | slice sens/spec {v['slice_sensitivity']:.2f}/{v['slice_specificity']:.2f} "
              f"| lr {lr:.1e} | {time.time() - t0:.0f}s{mark}", flush=True)
        if bad >= args.patience:
            print(f"Early stopping: no improvement for {args.patience} epochs")
            break

    model.load_state_dict(torch.load(best_path, map_location=device, weights_only=False)["model"])
    model.eval()
    res = scan_metrics(model, cases, te, device, args.batch, args.min_px)
    res.update({"model": args.arch, "encoder": args.encoder, "test_scans": te})
    json.dump(res, open(os.path.join(args.out, "test_results.json"), "w"), indent=2)
    print(f"TEST ({res['n_scans']} scans) | 3D Dice {res['dice_3d_mean']:.3f} (median {res['dice_3d_median']:.3f}) "
          f"| volume error {res['volume_mae_ml']:.1f} mL (r = {res['volume_r']:.2f}) "
          f"| slice sensitivity {res['slice_sensitivity']:.3f}, specificity {res['slice_specificity']:.3f}")


if __name__ == "__main__":
    main()
