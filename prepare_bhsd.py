"""Convert BHSD head-CT volumes + masks into compact 2D slice files for training.

  python prepare_bhsd.py --src /content/data/bhsd --out /content/data/bhsd_256 --size 256
  python prepare_bhsd.py --src /content/data/bhsd --out /content/data/bhsd_256_crop --crop   # head cropped first

For each scan writes <out>/<case>.npz with
  images  (Z, 3, size, size) uint8    3 CT windows per slice (see data_ct.py)
  masks   (Z, size, size)    uint8    0 = none, 1..5 = bleed type as stored in BHSD
  spacing (3,)               float    original voxel size in mm (x, y, z) after RAS reorientation
  pixel_mm (2,)              float    size of one RESIZED pixel in mm (row, column), for areas and volumes
--src is searched for an image folder and a mask folder with matching file names
(or pass --images and --labels). Works for other head-CT haemorrhage sets with the same layout.
"""

import argparse
import json
import os

import numpy as np

from data_ct import find_pairs, head_box, load_nifti, radiological, stem, volume_to_slices


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=".")
    ap.add_argument("--images")
    ap.add_argument("--labels")
    ap.add_argument("--out", required=True)
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--crop", action="store_true", help="Crop each scan to the head before resizing")
    a = ap.parse_args()

    pairs, img_dir, lab_dir = find_pairs(a.src, a.images, a.labels)
    print(f"images: {img_dir}\nlabels: {lab_dir}\n{len(pairs)} scans")
    os.makedirs(a.out, exist_ok=True)
    json.dump({"size": a.size, "crop": bool(a.crop)}, open(os.path.join(a.out, "prep.json"), "w"))
    n_slices = n_bleed = 0
    for i, (ip, lp) in enumerate(pairs, 1):
        img, lab = load_nifti(ip), load_nifti(lp)
        hu = np.asanyarray(img.dataobj).astype(np.float32)
        seg = np.rint(np.asanyarray(lab.dataobj)).astype(np.int16)
        if hu.ndim == 4:
            hu = hu[..., 0]
        if seg.shape != hu.shape:
            print(f"skip {stem(ip)}: image {hu.shape} vs mask {seg.shape}")
            continue
        box = head_box(hu) if a.crop else (0, hu.shape[1], 0, hu.shape[0])
        images, masks = volume_to_slices(hu, seg, a.size, box)
        lost = int((seg > 0).sum()) - int(sum((radiological(seg[:, :, k])[box[0]:box[1], box[2]:box[3]] > 0).sum()
                                               for k in range(seg.shape[2])))
        if lost:
            print(f"warning {stem(ip)}: crop cut off {lost} bleed voxels")
        spacing = np.array(img.header.get_zooms()[:3], dtype=np.float32)
        # display rows come from the y axis and columns from the x axis (see data_ct.radiological)
        rows, cols = box[1] - box[0], box[3] - box[2]
        pixel_mm = np.array([spacing[1] * rows / a.size, spacing[0] * cols / a.size], np.float32)
        cid = stem(ip).replace(" ", "_")
        np.savez_compressed(os.path.join(a.out, f"{cid}.npz"), images=images, masks=masks,
                            spacing=spacing, pixel_mm=pixel_mm)
        n_slices += len(masks)
        n_bleed += int((masks.reshape(len(masks), -1).max(1) > 0).sum())
        if i % 20 == 0 or i == len(pairs):
            print(f"{i}/{len(pairs)} scans done")
    print(f"Saved {n_slices} slices ({n_bleed} with bleeding) to {a.out}")


if __name__ == "__main__":
    main()
