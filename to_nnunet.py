"""Convert public datasets to nnU-Net v2 format (used by train_3d.py AND by nnU-Net itself).

  Brain tumour (MRI, BraTS 2021 or 2023 glioma):
    python to_nnunet.py brats   --src /content/data/brats   --out /content/nnUNet_raw --id 501
  Ischaemic stroke (MRI, ISLES 2022: DWI + ADC):
    python to_nnunet.py isles22 --src /content/data/ISLES-2022 --out /content/nnUNet_raw --id 502
  Haemorrhagic stroke (CT) or any image/label folder pair (BHSD, INSTANCE 2022, your own data):
    python to_nnunet.py pairs   --src /content/data/bhsd --out /content/nnUNet_raw --id 503 --name BHSD \
                                --channel CT --label-names bleed_1,bleed_2,bleed_3,bleed_4,bleed_5
    (--src is searched for an image folder and a label/mask folder with matching file names;
     or give --images DIR --labels DIR explicitly)
"""

import argparse
import glob
import json
import os
import re
import shutil
import tempfile

import nibabel as nib
import numpy as np


# --------------------------------------------------------------------------- helpers
def ds_dir(out, did, name):
    root = os.path.join(out, f"Dataset{did:03d}_{name}")
    for sub in ("imagesTr", "labelsTr"):
        os.makedirs(os.path.join(root, sub), exist_ok=True)
    return root


def link(src, dst):
    if os.path.exists(dst):
        return
    try:
        os.symlink(os.path.abspath(src), dst)
    except OSError:                       # Windows without symlink rights
        shutil.copy(src, dst)


def load_any(path):
    """nibabel needs a .nii/.nii.gz ending; BHSD ships files without one."""
    if path.endswith((".nii", ".nii.gz")):
        return nib.load(path)
    with open(path, "rb") as f:
        is_gz = f.read(2) == b"\x1f\x8b"
    tmp = os.path.join(tempfile.mkdtemp(), "x.nii.gz" if is_gz else "x.nii")
    shutil.copy(path, tmp)
    img = nib.load(tmp)
    img = nib.Nifti1Image(np.asanyarray(img.dataobj), img.affine, img.header)   # load into memory
    shutil.rmtree(os.path.dirname(tmp), ignore_errors=True)
    return img


def save_label(arr, ref, dst):
    nib.save(nib.Nifti1Image(np.asarray(arr).astype(np.uint8), ref.affine), dst)


def write_json(root, channels, labels, n, extra=None):
    meta = {"channel_names": {str(i): c for i, c in enumerate(channels)}, "labels": labels,
            "numTraining": n, "file_ending": ".nii.gz"}
    meta.update(extra or {})
    json.dump(meta, open(os.path.join(root, "dataset.json"), "w"), indent=2)
    print(f"Done: {n} cases -> {root}")


def stem(p):
    b = os.path.basename(p)
    for ext in (".nii.gz", ".nii"):
        if b.endswith(ext):
            return b[: -len(ext)]
    return b


# --------------------------------------------------------------------------- BraTS
BRATS_MODS = [("flair", "t2f"), ("t1", "t1n"), ("t1ce", "t1c"), ("t2", "t2w")]


def brats(a):
    root = ds_dir(a.out, a.id, "BraTS")
    n = 0
    for seg in sorted(glob.glob(os.path.join(a.src, "**", "*seg.nii.gz"), recursive=True)):
        d, files = os.path.dirname(seg), os.listdir(os.path.dirname(seg))
        imgs = []
        for names in BRATS_MODS:
            hit = [f for f in files for nm in names if f.endswith((f"_{nm}.nii.gz", f"-{nm}.nii.gz"))]
            imgs.append(os.path.join(d, hit[0]) if hit else None)
        if None in imgs:
            continue
        cid = os.path.basename(d)
        for ch, p in enumerate(imgs):
            link(p, os.path.join(root, "imagesTr", f"{cid}_{ch:04d}.nii.gz"))
        s_img = nib.load(seg)
        s = np.asanyarray(s_img.dataobj)
        new = np.zeros(s.shape, np.uint8)
        new[s == 2] = 1                      # edema
        new[s == 1] = 2                      # necrosis / non-enhancing core
        new[(s == 4) | (s == 3)] = 3         # enhancing (4 in 2021, 3 in 2023)
        save_label(new, s_img, os.path.join(root, "labelsTr", f"{cid}.nii.gz"))
        n += 1
    # region-based training, the set-up nnU-Net uses for BraTS
    write_json(root, ["FLAIR", "T1", "T1ce", "T2"],
               {"background": 0, "whole_tumor": [1, 2, 3], "tumor_core": [2, 3], "enhancing_tumor": [3]},
               n, {"regions_class_order": [1, 2, 3]})


# --------------------------------------------------------------------------- ISLES 2022
def isles22(a):
    root = ds_dir(a.out, a.id, "ISLES22")
    n = 0
    for dwi in sorted(glob.glob(os.path.join(a.src, "**", "*_dwi.nii.gz"), recursive=True)):
        if "derivatives" in dwi:
            continue
        cid = re.search(r"(sub-strokecase\d+)", dwi).group(1)
        adc = dwi.replace("_dwi.nii.gz", "_adc.nii.gz")
        msk = glob.glob(os.path.join(a.src, "**", "derivatives", "**", f"{cid}_*msk.nii.gz"), recursive=True)
        if not (os.path.exists(adc) and msk):
            print("skip (missing ADC or mask):", cid)
            continue
        link(dwi, os.path.join(root, "imagesTr", f"{cid}_0000.nii.gz"))
        link(adc, os.path.join(root, "imagesTr", f"{cid}_0001.nii.gz"))
        m = nib.load(msk[0])
        ref = nib.load(dwi)
        save_label(np.asanyarray(m.dataobj) > 0, ref, os.path.join(root, "labelsTr", f"{cid}.nii.gz"))
        n += 1
    # FLAIR is left out: ISLES'22 releases it in its own (unregistered) space
    write_json(root, ["DWI", "ADC"], {"background": 0, "infarct": 1}, n)


# --------------------------------------------------------------------------- generic image/label pairs
def find_pair_dirs(src):
    dirs = [d for d, _, f in os.walk(src) if f]
    lab = [d for d in dirs if re.search(r"label|mask|ground|seg", os.path.basename(d), re.I)]
    for ld in lab:
        lstems = {stem(f) for f in os.listdir(ld)}
        for d in dirs:
            if d != ld and d not in lab and len(lstems & {stem(f) for f in os.listdir(d)}) >= 0.8 * len(lstems):
                return d, ld
    raise SystemExit("Could not find matching image and label folders; pass --images and --labels")


def pairs(a):
    img_dir, lab_dir = (a.images, a.labels) if a.images and a.labels else find_pair_dirs(a.src)
    print("images:", img_dir, "| labels:", lab_dir)
    root = ds_dir(a.out, a.id, a.name)
    labels_by_stem = {stem(p): p for p in glob.glob(os.path.join(lab_dir, "*")) if os.path.isfile(p)}
    names = a.label_names.split(",")
    n = 0
    for p in sorted(glob.glob(os.path.join(img_dir, "*"))):
        s = stem(p)
        if not os.path.isfile(p) or s not in labels_by_stem:
            continue
        cid = re.sub(r"[^A-Za-z0-9-]", "-", s)
        img = load_any(p)
        nib.save(nib.Nifti1Image(np.asanyarray(img.dataobj).astype(np.float32), img.affine),
                 os.path.join(root, "imagesTr", f"{cid}_0000.nii.gz"))
        lab = np.asanyarray(load_any(labels_by_stem[s]).dataobj)
        lab = np.rint(lab).astype(np.int64)
        if lab.max() > len(names):
            raise SystemExit(f"{s}: label value {lab.max()} but only {len(names)} --label-names given")
        save_label(lab, img, os.path.join(root, "labelsTr", f"{cid}.nii.gz"))
        n += 1
    labels = {"background": 0, **{nm: i + 1 for i, nm in enumerate(names)}}
    write_json(root, [a.channel], labels, n)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, fn in (("brats", brats), ("isles22", isles22), ("pairs", pairs)):
        p = sub.add_parser(name)
        p.set_defaults(fn=fn)
        p.add_argument("--src", required=name != "pairs")
        p.add_argument("--out", required=True, help="nnUNet_raw folder")
        p.add_argument("--id", type=int, required=True)
        if name == "pairs":
            p.add_argument("--images"); p.add_argument("--labels")
            p.add_argument("--name", default="Custom")
            p.add_argument("--channel", default="CT", help="CT, or an MRI name like FLAIR")
            p.add_argument("--label-names", default="lesion", help="Comma list for label values 1, 2, ...")
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
