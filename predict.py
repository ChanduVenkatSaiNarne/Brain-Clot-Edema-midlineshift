"""Find bleeding (clot) on a head-CT scan and measure it.

  python predict.py --ckpt models/clot/clot_best.pt --ct scan.nii.gz --out predictions/
  python predict.py --ckpt models/clot/clot_best.pt --ct scan.nii.gz --truth mask.nii.gz --out predictions/

Writes to --out:
  <case>_clot.nii.gz    predicted bleed mask (RAS orientation; opens over the CT in 3D Slicer / ITK-SNAP)
  <case>_overlay.png    the 4 slices with the most bleeding, prediction in red (truth in green if given)
  measurements.csv      bleed volume (mL), side (patient's right/left/both) and number of slices, one row per run

Midline shift and edema need their own trained models (see README: awaiting suitable data).
Research use only, not for diagnosis.
"""

import argparse
import csv
import os

import cv2
import numpy as np
import torch

import measurements as meas
from data_ct import load_nifti, radiological, stem, volume_to_slices
from model import load_checkpoint, predict_slices
from train_2d import remove_small


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--ct", required=True, help="Head CT volume (.nii / .nii.gz, values in HU)")
    ap.add_argument("--truth", help="Optional expert mask for comparison")
    ap.add_argument("--out", default="predictions")
    ap.add_argument("--threshold", type=float, default=0.5)
    a = ap.parse_args()

    import nibabel as nib
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, ck = load_checkpoint(a.ckpt, device)
    size = ck["size"]

    img = load_nifti(a.ct)
    hu = np.asanyarray(img.dataobj).astype(np.float32)
    hu = hu[..., 0] if hu.ndim == 4 else hu
    X, Y, Z = hu.shape
    images, _ = volume_to_slices(hu, np.zeros(hu.shape, np.uint8), size)
    prob = predict_slices(model, images, device)
    pred = np.stack([remove_small(p, a.threshold, ck.get("min_px", 10)) for p in prob])     # (Z, size, size)

    # measurements on the resized slices, converted with the real pixel size
    sx, sy, sz = img.header.get_zooms()[:3]
    px_mm2 = (sy * Y / size) * (sx * X / size)
    volume_ml = float(pred.sum() * px_mm2 * sz / 1000)
    right = left = 0
    for k in np.where(pred.reshape(Z, -1).any(1))[0]:
        head = meas.brain_mask(images[k, 2])                 # bone window shows the skull outline best
        mid = meas.ideal_midline_x(head)
        if mid is None:
            continue
        cols = np.where(pred[k])[1]
        right += int((cols < mid).sum())                     # radiological: image left = patient's right
        left += int((cols >= mid).sum())
    if right + left == 0:
        side = "none"
    elif min(right, left) / (right + left) > 0.2:
        side = "both sides"
    else:
        side = "patient's right" if right > left else "patient's left"

    # save the mask back in the scan's own grid (inverse of data_ct.radiological + resize)
    mask_vol = np.zeros((X, Y, Z), np.uint8)
    for k in range(Z):
        disp = cv2.resize(pred[k], (X, Y), interpolation=cv2.INTER_NEAREST)       # (rows=Y, cols=X)
        mask_vol[:, :, k] = np.fliplr(np.flipud(disp)).T
    os.makedirs(a.out, exist_ok=True)
    case = stem(a.ct)
    nib.save(nib.Nifti1Image(mask_vol, img.affine), os.path.join(a.out, f"{case}_clot.nii.gz"))

    truth_ml, dice = None, None
    gt_slices = None
    if a.truth:
        t = np.rint(np.asanyarray(load_nifti(a.truth).dataobj)) > 0
        _, gt_slices = volume_to_slices(hu, t.astype(np.uint8), size)
        gt_slices = (gt_slices > 0).astype(np.uint8)
        truth_ml = float(gt_slices.sum() * px_mm2 * sz / 1000)
        s = pred.sum() + gt_slices.sum()
        dice = 1.0 if s == 0 else float(2 * (pred & gt_slices).sum() / s)

    # overlay figure: 4 slices with the most predicted (or true) bleeding
    score = pred.reshape(Z, -1).sum(1) + (gt_slices.reshape(Z, -1).sum(1) if gt_slices is not None else 0)
    tiles = []
    for k in np.argsort(score)[::-1][:4]:
        rgb = cv2.cvtColor(images[k, 0], cv2.COLOR_GRAY2BGR)
        cnts, _ = cv2.findContours(pred[k], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(rgb, cnts, -1, (0, 0, 255), 1)
        if gt_slices is not None:
            cnts, _ = cv2.findContours(gt_slices[k], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(rgb, cnts, -1, (0, 255, 0), 1)
        cv2.putText(rgb, f"slice {k}", (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        tiles.append(rgb)
    cv2.imwrite(os.path.join(a.out, f"{case}_overlay.png"), np.hstack(tiles))

    row = {"case": case, "bleed_volume_ml": round(volume_ml, 2), "side": side,
           "slices_with_bleed": int(pred.reshape(Z, -1).any(1).sum()),
           "truth_volume_ml": None if truth_ml is None else round(truth_ml, 2),
           "dice_3d": None if dice is None else round(dice, 3)}
    path = os.path.join(a.out, "measurements.csv")
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        if new:
            w.writeheader()
        w.writerow(row)
    print(row, "- research use only")


if __name__ == "__main__":
    main()
