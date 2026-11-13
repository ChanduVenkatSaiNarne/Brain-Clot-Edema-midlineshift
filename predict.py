"""Run the clot, edema and midline models on images and report measurements.

Replaces midline_shift_h5_testing_model.py. Changes:
  * no made-up symptoms, clinical conditions or %-area "severity grades"
  * midline shift reported as a DISTANCE (estimate), not an area
  * mm / mm^2 only when the real pixel spacing is given (--pixel-spacing)
  * hemisphere uses radiological convention and the head's midline
  * clot/edema overlap is counted once
  * runs outside Colab; saves a figure and a CSV per run

Example:
  python predict.py --images data/slices \
      --clot-model models/unet_clot_best.keras --edema-model models/unet_edema_best.keras \
      --midline-model models/unet_midline_best.keras --pixel-spacing 0.45
"""

import argparse
import csv
import os

import cv2
import numpy as np

import measurements as m

IMG_SIZE = 256
EXTS = (".png", ".jpg", ".jpeg")


def load_model(path):
    import tensorflow as tf
    return tf.keras.models.load_model(path, compile=False)  # compile=False: no custom loss needed


def segment(model, gray):
    small = cv2.resize(gray, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA) / 255.0
    prob = model.predict(small[None, ..., None], verbose=0)[0, ..., 0]
    prob = cv2.resize(prob, (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_LINEAR)
    return (prob > 0.5).astype(np.uint8)


def overlay(bgr, mask, colour):
    out = bgr.copy()
    tint = np.zeros_like(bgr)
    tint[:] = colour
    blended = cv2.addWeighted(bgr, 0.6, tint, 0.4, 0)
    out[mask == 1] = blended[mask == 1]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True, help="Image file or folder")
    ap.add_argument("--clot-model", required=True)
    ap.add_argument("--edema-model", required=True)
    ap.add_argument("--midline-model", required=True)
    ap.add_argument("--pixel-spacing", type=float, default=None,
                    help="Real pixel spacing in mm from the DICOM/NIfTI header. Omit if unknown.")
    ap.add_argument("--neurological", action="store_true",
                    help="Images use neurological convention (image left = patient left)")
    ap.add_argument("--out", default="predictions")
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    files = ([os.path.join(args.images, f) for f in sorted(os.listdir(args.images))
              if f.lower().endswith(EXTS)] if os.path.isdir(args.images) else [args.images])
    models = {k: load_model(p) for k, p in
              [("clot", args.clot_model), ("edema", args.edema_model), ("midline", args.midline_model)]}
    os.makedirs(args.out, exist_ok=True)
    radiological = not args.neurological
    unit = "mm" if args.pixel_spacing else "px"
    rows = []

    for path in files:
        bgr = cv2.imread(path)
        if bgr is None:
            print(f"Skipping unreadable file: {path}")
            continue
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        masks = {k: segment(mdl, gray) for k, mdl in models.items()}
        head = m.brain_mask(gray)
        mid_x = m.ideal_midline_x(head)
        shift = m.midline_shift(masks["midline"], mid_x, args.pixel_spacing)
        clot_a = m.area(masks["clot"], args.pixel_spacing)
        edema_a = m.area(masks["edema"], args.pixel_spacing)
        row = {
            "image": os.path.basename(path),
            "clot_area_px": clot_a["pixels"], "clot_area_mm2": clot_a["mm2"],
            "clot_side": m.side(masks["clot"], mid_x, radiological),
            "edema_area_px": edema_a["pixels"], "edema_area_mm2": edema_a["mm2"],
            "edema_side": m.side(masks["edema"], mid_x, radiological),
            "lesion_percent_of_head": m.lesion_fraction(masks["clot"], masks["edema"], head),
            "midline_shift_px": shift["pixels"], "midline_shift_mm": shift["mm"],
            "midline_shift_direction": shift["direction"],
        }
        rows.append(row)

        shift_val = shift["mm"] if args.pixel_spacing else shift["pixels"]
        shift_txt = "not measurable" if shift_val is None else f"{shift_val:.1f} {unit} (estimate)"
        panels = [
            (bgr, "Original"),
            (overlay(bgr, masks["clot"], (0, 0, 255)), f"Clot – {row['clot_side']}"),
            (overlay(bgr, masks["edema"], (0, 255, 255)), f"Edema – {row['edema_side']}"),
            (overlay(bgr, masks["midline"], (0, 255, 0)), f"Midline shift: {shift_txt}"),
        ]
        fig, axs = plt.subplots(1, 4, figsize=(20, 5))
        for ax, (im, title) in zip(axs, panels):
            ax.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
            if mid_x is not None:
                ax.axvline(mid_x, color="white", lw=0.8, ls="--")
            ax.set_title(title, fontsize=10)
            ax.axis("off")
        fig.suptitle("Research output only – not for clinical use", fontsize=9)
        fig.tight_layout()
        fig.savefig(os.path.join(args.out, os.path.splitext(row["image"])[0] + "_pred.png"), dpi=120)
        plt.close(fig)
        print(f"{row['image']}: clot {clot_a['pixels']} px ({row['clot_side']}), "
              f"edema {edema_a['pixels']} px ({row['edema_side']}), midline shift {shift_txt}")

    if rows:
        with open(os.path.join(args.out, "measurements.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    if not args.pixel_spacing:
        print("Note: no --pixel-spacing given, so sizes are in pixels, not mm.")


if __name__ == "__main__":
    main()
