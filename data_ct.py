"""Head-CT data helpers: find image/mask pairs, read NIfTI volumes, window HU values, cut axial slices.

Conventions used everywhere in this repo
* Volumes are reoriented to RAS, then each axial slice is shown in RADIOLOGICAL convention
  (image left = patient's right, anterior at the top), matching measurements.py.
* Each slice becomes 3 channels = 3 standard CT windows (like a radiologist switching windows):
    brain     level 40,  width 80    grey/white matter, most bleeds
    subdural  level 80,  width 200   thin bleeds next to the skull
    bone      level 600, width 2800  skull, fractures, head outline
  Three channels also match ImageNet-pretrained encoders.
"""

import glob
import os
import re
import shutil
import tempfile

import cv2
import numpy as np

WINDOWS = [(40, 80), (80, 200), (600, 2800)]          # (level, width) in Hounsfield units


def window(hu, level, width):
    lo, hi = level - width / 2, level + width / 2
    return ((np.clip(hu, lo, hi) - lo) / (hi - lo) * 255).astype(np.uint8)


def to_channels(hu_slice):
    """(H, W) HU slice -> (3, H, W) uint8, one channel per CT window."""
    return np.stack([window(hu_slice, lv, wd) for lv, wd in WINDOWS])


def radiological(slice_xy):
    """RAS (x, y) slice -> display array (rows = anterior..posterior, cols = patient right..left)."""
    return np.flipud(np.fliplr(slice_xy.T))


def stem(path):
    b = os.path.basename(path)
    for ext in (".nii.gz", ".nii"):
        if b.endswith(ext):
            return b[: -len(ext)]
    return b


def load_nifti(path):
    """nibabel needs a .nii/.nii.gz ending; BHSD files may have none. Returns a RAS-oriented image."""
    import nibabel as nib
    if not path.endswith((".nii", ".nii.gz")):
        with open(path, "rb") as f:
            gz = f.read(2) == b"\x1f\x8b"
        tmp = os.path.join(tempfile.mkdtemp(), "x.nii.gz" if gz else "x.nii")
        shutil.copy(path, tmp)
        img = nib.load(tmp)
        img = nib.Nifti1Image(np.asanyarray(img.dataobj), img.affine, img.header)
        shutil.rmtree(os.path.dirname(tmp), ignore_errors=True)
    else:
        img = nib.load(path)
    return nib.as_closest_canonical(img)


def find_pairs(src, images=None, labels=None):
    """Find (image, label) file pairs with the same name. Label folder is detected by name
    (label / mask / ground / seg) unless given."""
    if not (images and labels):
        dirs = [d for d, _, f in os.walk(src) if f]
        lab_dirs = [d for d in dirs if re.search(r"label|mask|ground|seg", os.path.basename(d), re.I)]
        for ld in lab_dirs:
            ls = {stem(f) for f in os.listdir(ld)}
            for d in dirs:
                if d != ld and d not in lab_dirs and len(ls & {stem(f) for f in os.listdir(d)}) >= 0.8 * len(ls):
                    images, labels = d, ld
                    break
            if images:
                break
    if not (images and labels):
        raise SystemExit("Could not find image and label folders with matching names; pass --images and --labels")
    lab = {stem(p): p for p in glob.glob(os.path.join(labels, "*")) if os.path.isfile(p)}
    pairs = [(p, lab[stem(p)]) for p in sorted(glob.glob(os.path.join(images, "*")))
             if os.path.isfile(p) and stem(p) in lab]
    return pairs, images, labels


def volume_to_slices(hu_vol, label_vol, size, box=None):
    """RAS volumes (X, Y, Z) -> images (Z, 3, size, size) uint8 and masks (Z, size, size) uint8 (0..5).
    box = (r0, r1, c0, c1) crops every slice (in display coordinates) before resizing."""
    imgs, masks = [], []
    r0, r1, c0, c1 = box if box else (0, None, 0, None)
    for k in range(hu_vol.shape[2]):
        img = radiological(hu_vol[:, :, k].astype(np.float32))[r0:r1, c0:c1]
        lab = radiological(label_vol[:, :, k])[r0:r1, c0:c1]
        ch = to_channels(img)
        imgs.append(np.stack([cv2.resize(c, (size, size), interpolation=cv2.INTER_AREA) for c in ch]))
        masks.append(cv2.resize(lab.astype(np.uint8), (size, size), interpolation=cv2.INTER_NEAREST))
    return np.stack(imgs), np.stack(masks)


def head_box(hu_vol, margin=0.05, thresh=-300):
    """Square crop box (r0, r1, c0, c1) in display coordinates around the head, the same for
    every slice of the scan. Air and the scanner table are cut away, so after resizing the brain
    fills more of the image. Falls back to the full slice if nothing is found."""
    proj = radiological((hu_vol > thresh).any(axis=2).astype(np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(proj, connectivity=8)
    H, W = proj.shape
    if n < 2:
        return 0, H, 0, W
    k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))          # largest blob = head (table is thin)
    x, y, w, h = stats[k, :4]
    side = int(max(w, h) * (1 + 2 * margin))
    side = min(side, max(H, W))
    cy, cx = y + h // 2, x + w // 2
    r0 = int(np.clip(cy - side // 2, 0, max(H - side, 0))); c0 = int(np.clip(cx - side // 2, 0, max(W - side, 0)))
    return r0, min(r0 + side, H), c0, min(c0 + side, W)


def case_split(case_ids, seed=42, frac=(0.70, 0.15, 0.15)):
    """Split by SCAN: all slices of one scan stay in one part (no leakage between train and test)."""
    ids = sorted(case_ids)
    rng = np.random.default_rng(seed)
    rng.shuffle(ids)
    n_tr, n_va = int(frac[0] * len(ids)), int(frac[1] * len(ids))
    return ids[:n_tr], ids[n_tr:n_tr + n_va], ids[n_tr + n_va:]
