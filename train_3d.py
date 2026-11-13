"""Train a 3D segmentation model (tumour, ischaemic stroke or haemorrhage) with MONAI.

Input is a dataset converted by to_nnunet.py. Examples:
  python train_3d.py --dataset /content/nnUNet_raw/Dataset501_BraTS  --model segresnet --out runs/brats_segresnet
  python train_3d.py --dataset /content/nnUNet_raw/Dataset502_ISLES22 --model swinunetr --out runs/isles_swin
  python train_3d.py --dataset /content/nnUNet_raw/Dataset503_BHSD   --model dynunet   --out runs/bhsd_dyn
  add --resume to continue after a Colab disconnect; --max-cases 20 --epochs 2 for a quick test

Saves in --out: best.pt, last.pt, split.json, history.csv, test_results.json
"""

import argparse
import csv
import json
import os
import time

import numpy as np
import torch

from data_3d import DatasetInfo, get_transforms, split_cases
from models_3d import DEFAULT_CUBE, build_model


def dice_scores(model, loader, info, roi, device, amp, overlap=0.5):
    """Mean Dice per region/class over the cases in loader (background excluded)."""
    from monai.inferers import sliding_window_inference
    from monai.metrics import DiceMetric

    metric = DiceMetric(include_background=info.regions, reduction="mean_batch", ignore_empty=True)
    model.eval()
    with torch.no_grad():
        for b in loader:
            x, y = b["image"].to(device), b["label"].to(device)
            with torch.autocast(device.type, enabled=amp):
                logits = sliding_window_inference(x, roi, 2, model, overlap=overlap).float()
            if info.regions:
                pred = (torch.sigmoid(logits) > 0.5).float()
            else:
                n = info.out_channels
                pred = torch.nn.functional.one_hot(logits.argmax(1), n).permute(0, 4, 1, 2, 3).float()
                y = torch.nn.functional.one_hot(y[:, 0].long(), n).permute(0, 4, 1, 2, 3).float()
            metric(pred, y)
    d = metric.aggregate().cpu().numpy()
    metric.reset()
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="nnU-Net-format dataset folder (from to_nnunet.py)")
    ap.add_argument("--model", default="segresnet", choices=list(DEFAULT_CUBE))
    ap.add_argument("--out", default="runs/run1")
    ap.add_argument("--roi", type=int, nargs=3, help="Patch size, e.g. 128 128 128 (default: automatic)")
    ap.add_argument("--spacing", type=float, nargs=3, help="Voxel size in mm (default: dataset median)")
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--val-every", type=int, default=2)
    ap.add_argument("--patience", type=int, default=20, help="Validation checks without improvement")
    ap.add_argument("--batch", type=int, default=1, help="Volumes per step (each gives 2 patches)")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--max-cases", type=int)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    from monai.data import DataLoader, Dataset
    from monai.losses import DiceCELoss, DiceLoss
    from monai.utils import set_determinism

    set_determinism(42)
    info = DatasetInfo(args.dataset)
    cases = info.cases()[:args.max_cases] if args.max_cases else info.cases()
    if not cases:
        raise SystemExit(f"No cases found in {args.dataset}/imagesTr + labelsTr")
    os.makedirs(args.out, exist_ok=True)

    spacing = tuple(args.spacing) if args.spacing else info.median_spacing(cases)
    roi = tuple(args.roi) if args.roi else info.default_roi(spacing, DEFAULT_CUBE[args.model])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = device.type == "cuda"
    if amp:
        p = torch.cuda.get_device_properties(0)
        print(f"GPU: {p.name}, {p.total_memory / 1e9:.1f} GB")
    else:
        print("WARNING: no CUDA GPU - 3D training on CPU is not realistic")

    tr, va, te = split_cases(cases, os.path.join(args.out, "split.json"))
    print(f"{len(cases)} cases -> train {len(tr)}, val {len(va)}, test {len(te)}")
    print(f"Channels {info.channels} | {'regions' if info.regions else 'classes'} {info.names}")
    print(f"Model {args.model} | spacing {spacing} mm | patch {roi}")

    tr_loader = DataLoader(Dataset(tr, get_transforms(info, spacing, roi, "train")), batch_size=args.batch,
                           shuffle=True, num_workers=args.workers, pin_memory=True)
    va_loader = DataLoader(Dataset(va, get_transforms(info, spacing, roi, "val")), batch_size=1, num_workers=args.workers)
    te_loader = DataLoader(Dataset(te, get_transforms(info, spacing, roi, "val")), batch_size=1, num_workers=args.workers)

    model = build_model(args.model, len(info.channels), info.out_channels, roi).to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    if info.regions:
        loss_fn = DiceLoss(smooth_nr=0, smooth_dr=1e-5, squared_pred=True, sigmoid=True)
    else:
        loss_fn = DiceCELoss(to_onehot_y=True, softmax=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    scaler = torch.amp.GradScaler(device.type, enabled=amp)

    best_path, last_path = os.path.join(args.out, "best.pt"), os.path.join(args.out, "last.pt")
    hist_path = os.path.join(args.out, "history.csv")
    meta = {"model_name": args.model, "roi": list(roi), "spacing": list(spacing), "dataset": args.dataset}
    start, best, bad = 1, -1.0, 0
    if args.resume and os.path.exists(last_path):
        ck = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"])
        start, best, bad = ck["epoch"] + 1, ck["best"], ck["bad"]
        print(f"Resumed after epoch {ck['epoch']} (best val Dice {best:.3f})")
    else:
        with open(hist_path, "w", newline="") as f:
            csv.writer(f).writerow(["epoch", "loss", "val_dice_mean"] + [f"val_dice_{n}" for n in info.names] + ["lr"])

    for epoch in range(start, args.epochs + 1):
        t0, total, n = time.time(), 0.0, 0
        model.train()
        for b in tr_loader:
            x, y = b["image"].to(device), b["label"].to(device)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device.type, enabled=amp):
                loss = loss_fn(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            total += loss.item(); n += 1
        sched.step()
        row = [epoch, total / n, ""] + [""] * len(info.names) + [opt.param_groups[0]["lr"]]
        msg = f"Epoch {epoch:3d} | loss {total / n:.4f}"
        if epoch % args.val_every == 0:
            d = dice_scores(model, va_loader, info, roi, device, amp)
            row[2:3 + len(d)] = [float(np.nanmean(d))] + [float(v) for v in d]
            msg += f" | val Dice {np.nanmean(d):.3f} (" + ", ".join(f"{k} {v:.3f}" for k, v in zip(info.names, d)) + ")"
            if np.nanmean(d) > best:
                best, bad = float(np.nanmean(d)), 0
                torch.save({"model": model.state_dict(), **meta}, best_path)
                msg += "  * best"
            else:
                bad += 1
        print(msg + f" | {time.time() - t0:.0f}s", flush=True)
        with open(hist_path, "a", newline="") as f:
            csv.writer(f).writerow(row)
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "epoch": epoch, "best": best, "bad": bad, **meta}, last_path)
        if bad >= args.patience:
            print(f"Early stopping: no validation improvement for {args.patience} checks")
            break

    model.load_state_dict(torch.load(best_path, map_location=device, weights_only=False)["model"])
    d = dice_scores(model, te_loader, info, roi, device, amp)
    res = {"model": args.model, "dataset": os.path.basename(args.dataset.rstrip("/")), "n_test_cases": len(te),
           "test_dice": {k: float(v) for k, v in zip(info.names, d)}, "test_dice_mean": float(np.nanmean(d)),
           "spacing_mm": list(spacing), "patch": list(roi)}
    json.dump(res, open(os.path.join(args.out, "test_results.json"), "w"), indent=2)
    print("TEST Dice", {k: round(v, 3) for k, v in res["test_dice"].items()}, "| mean", round(res["test_dice_mean"], 3))


if __name__ == "__main__":
    main()
