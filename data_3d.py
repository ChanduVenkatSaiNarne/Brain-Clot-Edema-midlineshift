"""Read any nnU-Net-format dataset and build MONAI transforms for it.

A dataset folder (made by to_nnunet.py) looks like:
  Dataset501_BraTS/
    dataset.json          channel names + labels
    imagesTr/<case>_0000.nii.gz, <case>_0001.nii.gz ...   one file per channel
    labelsTr/<case>.nii.gz

Two label modes, decided by dataset.json:
  regions  some label values are lists (BraTS): overlapping regions, one sigmoid channel each
  classes  plain integer labels (stroke lesion, haemorrhage types): softmax over background + classes

The same format is used by nnU-Net v2, so one converted dataset serves both pipelines.
"""

import glob
import json
import os
import random

import numpy as np

CT_WINDOW = (-15.0, 115.0)   # HU range kept for CT (brain + blood), scaled to 0-1


class DatasetInfo:
    def __init__(self, root):
        self.root = root
        self.meta = json.load(open(os.path.join(root, "dataset.json")))
        self.channels = [self.meta["channel_names"][k] for k in sorted(self.meta["channel_names"], key=int)]
        labels = {k: v for k, v in self.meta["labels"].items() if k != "background"}
        self.is_ct = any(c.upper() == "CT" for c in self.channels)
        self.regions = any(isinstance(v, list) for v in labels.values())
        if self.regions:
            self.names = list(labels)                                   # e.g. whole_tumor, tumor_core, enhancing
            self.region_values = [v if isinstance(v, list) else [v] for v in labels.values()]
            self.region_order = self.meta.get("regions_class_order", list(range(1, len(labels) + 1)))
            self.out_channels = len(labels)
        else:
            self.names = [k for k, _ in sorted(labels.items(), key=lambda kv: kv[1])]
            self.out_channels = len(labels) + 1                         # + background

    def cases(self, split="Tr"):
        out = []
        for lab in sorted(glob.glob(os.path.join(self.root, f"labels{split}", "*.nii.gz"))):
            cid = os.path.basename(lab)[:-7]
            imgs = [os.path.join(self.root, f"images{split}", f"{cid}_{i:04d}.nii.gz") for i in range(len(self.channels))]
            if all(os.path.exists(p) for p in imgs):
                out.append({"image": imgs, "label": lab, "id": cid})
        return out

    def median_spacing(self, cases, n=50):
        import nibabel as nib
        sp = np.array([nib.load(c["image"][0]).header.get_zooms()[:3] for c in cases[:n]], float)
        med = np.median(sp, axis=0)
        med[:2] = np.maximum(med[:2], 0.8)          # do not train on sub-0.8 mm in-plane (CT is ~0.4 mm)
        return tuple(float(round(v, 2)) for v in med)

    def default_roi(self, spacing, cube):
        """Cube patch for isotropic data; flat patch (e.g. 160x160x32) for thick-slice data like CT."""
        if spacing[2] > 2.5 * min(spacing[:2]):
            side = max(96, int(cube * 1.25) // 32 * 32)
            return (side, side, 32)
        return (cube, cube, cube)

    # label conversion ------------------------------------------------------
    def to_regions(self, lab):
        lab = np.asarray(lab)[0] if np.asarray(lab).ndim == 4 else np.asarray(lab)
        return np.stack([np.isin(lab, v) for v in self.region_values]).astype(np.float32)

    def probs_to_labels(self, probs, threshold=0.5):
        probs = np.asarray(probs)
        lab = np.zeros(probs.shape[1:], np.uint8)
        if self.regions:
            for i, value in enumerate(self.region_order):   # later regions overwrite earlier ones
                lab[probs[i] > threshold] = value
        else:
            lab = probs.argmax(0).astype(np.uint8)
        return lab


def split_cases(cases, out_json=None, seed=42, frac=(0.70, 0.15, 0.15)):
    """Case-level 70/15/15 split, saved to split.json so every run and model uses the same test set."""
    if out_json and os.path.exists(out_json):
        ids = json.load(open(out_json))
        by_id = {c["id"]: c for c in cases}
        return [[by_id[i] for i in ids[k] if i in by_id] for k in ("train", "val", "test")]
    order = cases[:]
    random.Random(seed).shuffle(order)
    n_tr, n_va = int(frac[0] * len(order)), int(frac[1] * len(order))
    parts = [order[:n_tr], order[n_tr:n_tr + n_va], order[n_tr + n_va:]]
    if out_json:
        os.makedirs(os.path.dirname(out_json) or ".", exist_ok=True)
        json.dump({k: [c["id"] for c in p] for k, p in zip(("train", "val", "test"), parts)},
                  open(out_json, "w"), indent=1)
    return parts


def get_transforms(info, spacing, roi, mode):
    """mode: 'train', 'val' (image + label, full volume) or 'infer' (image only)."""
    from monai import transforms as T

    class ToRegions(T.MapTransform):
        def __call__(self, data):
            d = dict(data)
            for k in self.keys:
                d[k] = info.to_regions(d[k])
            return d

    keys = ["image"] if mode == "infer" else ["image", "label"]
    interp = ("bilinear",) if mode == "infer" else ("bilinear", "nearest")
    t = [
        T.LoadImaged(keys=keys),
        T.EnsureChannelFirstd(keys=keys),
        T.Orientationd(keys=keys, axcodes="RAS"),
        T.Spacingd(keys=keys, pixdim=spacing, mode=interp),
    ]
    if info.is_ct:
        t.append(T.ScaleIntensityRanged(keys="image", a_min=CT_WINDOW[0], a_max=CT_WINDOW[1],
                                        b_min=0.0, b_max=1.0, clip=True))
    else:
        t.append(T.NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True))
    if mode == "train":
        t += [
            T.SpatialPadd(keys=keys, spatial_size=roi),
            # half the patches centred on lesion, half anywhere: essential for small stroke lesions
            T.RandCropByPosNegLabeld(keys=keys, label_key="label", spatial_size=roi, pos=1, neg=1, num_samples=2),
            T.RandFlipd(keys=keys, prob=0.5, spatial_axis=0),
            T.RandFlipd(keys=keys, prob=0.5, spatial_axis=1),
            T.RandRotate90d(keys=keys, prob=0.3, spatial_axes=(0, 1)),
            T.RandScaleIntensityd(keys="image", factors=0.1, prob=0.8),
            T.RandShiftIntensityd(keys="image", offsets=0.1, prob=0.8),
        ]
    if mode != "infer" and info.regions:
        t.append(ToRegions(keys="label"))
    t.append(T.EnsureTyped(keys=keys))
    return T.Compose(t)
