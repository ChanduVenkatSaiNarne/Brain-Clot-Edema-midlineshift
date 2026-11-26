"""Tune the decision threshold and speck size of a trained clot model, then report test once.

  python tune_clot.py --data data/bhsd_256 --model models/clot

No retraining. The model's probabilities are computed once per scan, then every
threshold / min_px pair is scored on the VALIDATION scans only. The best pair is applied
to the test scans a single time, so the test set is not used for tuning.

Also prints an error analysis: test Dice by bleed size, and the worst scans.
Writes tuning.csv and tuned_test_results.json into the model folder.
"""

import argparse
import csv
import json
import os

import numpy as np
import torch

from model import load_checkpoint, predict_slices
from train_2d import load_cases, remove_small

THRESHOLDS = [0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6]
MIN_PX = [0, 5, 10, 20, 40]


def probs_for(model, cases, ids, device, batch):
    return {c: predict_slices(model, cases[c]["images"], device, batch) for c in ids}


def score(cases, probs, threshold, min_px):
    rows, tp, fn, tn, fp = [], 0, 0, 0, 0
    for cid, prob in probs.items():
        c = cases[cid]
        pred = np.stack([remove_small(p, threshold, min_px) for p in prob])
        gt = c["masks"]
        s = pred.sum() + gt.sum()
        dice = 1.0 if s == 0 else 2 * (pred & gt).sum() / s
        rows.append({"case": cid, "dice_3d": float(dice), "has_bleed": bool(gt.any()),
                     "true_ml": float(gt.sum() * c["voxel_ml"]), "pred_ml": float(pred.sum() * c["voxel_ml"])})
        g, p = gt.reshape(len(gt), -1).any(1), pred.reshape(len(pred), -1).any(1)
        tp += int((g & p).sum()); fn += int((g & ~p).sum()); tn += int((~g & ~p).sum()); fp += int((~g & p).sum())
    bleed = [r["dice_3d"] for r in rows if r["has_bleed"]]
    t = np.array([r["true_ml"] for r in rows]); q = np.array([r["pred_ml"] for r in rows])
    return {"threshold": threshold, "min_px": min_px,
            "dice_3d_mean": float(np.mean(bleed)), "dice_3d_median": float(np.median(bleed)),
            "volume_mae_ml": float(np.abs(t - q).mean()),
            "volume_r": float(np.corrcoef(t, q)[0, 1]) if t.std() > 0 and q.std() > 0 else float("nan"),
            "slice_sensitivity": tp / max(tp + fn, 1), "slice_specificity": tn / max(tn + fp, 1),
            "per_scan": rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", default="models/clot", help="Folder with clot_best.pt and split.json")
    ap.add_argument("--batch", type=int, default=16)
    a = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, ck = load_checkpoint(os.path.join(a.model, "clot_best.pt"), device)
    split = json.load(open(os.path.join(a.model, "split.json")))
    cases = load_cases(a.data)
    va, te = split["val"], split["test"]

    print("Computing validation probabilities ...")
    pv = probs_for(model, cases, va, device, a.batch)
    results = []
    for th in THRESHOLDS:
        for mp in MIN_PX:
            r = score(cases, pv, th, mp)
            results.append(r)
            print(f"val | threshold {th:.2f} min_px {mp:3d} | Dice {r['dice_3d_mean']:.3f} "
                  f"(median {r['dice_3d_median']:.3f}) | vol err {r['volume_mae_ml']:.1f} mL "
                  f"| sens/spec {r['slice_sensitivity']:.3f}/{r['slice_specificity']:.3f}")
    with open(os.path.join(a.model, "tuning.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[k for k in results[0] if k != "per_scan"])
        w.writeheader()
        for r in results:
            w.writerow({k: v for k, v in r.items() if k != "per_scan"})

    best = max(results, key=lambda r: r["dice_3d_mean"])
    th, mp = best["threshold"], best["min_px"]
    print(f"\nBest on validation: threshold {th}, min_px {mp} (val Dice {best['dice_3d_mean']:.3f})")

    print("Computing test probabilities ...")
    pt = probs_for(model, cases, te, device, a.batch)
    base = score(cases, pt, 0.5, ck.get("min_px", 10))
    tuned = score(cases, pt, th, mp)
    for name, r in (("default 0.5", base), ("tuned", tuned)):
        print(f"TEST {name:11s} | 3D Dice {r['dice_3d_mean']:.3f} (median {r['dice_3d_median']:.3f}) "
              f"| volume error {r['volume_mae_ml']:.1f} mL (r = {r['volume_r']:.2f}) "
              f"| slice sensitivity {r['slice_sensitivity']:.3f}, specificity {r['slice_specificity']:.3f}")

    print("\nError analysis (tuned, test scans with bleeding)")
    bleed = [r for r in tuned["per_scan"] if r["has_bleed"]]
    for lo, hi, name in ((0, 5, "< 5 mL"), (5, 30, "5-30 mL"), (30, 1e9, "> 30 mL")):
        g = [r["dice_3d"] for r in bleed if lo <= r["true_ml"] < hi]
        if g:
            print(f"  {name:8s} | {len(g):2d} scans | mean Dice {np.mean(g):.3f}")
    print("  Worst 5 scans:")
    for r in sorted(bleed, key=lambda r: r["dice_3d"])[:5]:
        print(f"    {r['case']:30s} Dice {r['dice_3d']:.3f} | true {r['true_ml']:.1f} mL, predicted {r['pred_ml']:.1f} mL")

    json.dump({"threshold": th, "min_px": mp, "default": base, "tuned": tuned},
              open(os.path.join(a.model, "tuned_test_results.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
