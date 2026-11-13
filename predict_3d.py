"""Segment one scan with a trained model and save a NIfTI label map on the ORIGINAL scan grid.

  python predict_3d.py --ckpt runs/isles_swin/best.pt --images dwi.nii.gz adc.nii.gz --out pred/case1.nii.gz
  python predict_3d.py --ckpt runs/brats_segresnet/best.pt --case BraTS2021_00000 --out pred/brats0.nii.gz
      (--case looks up <case>_0000.nii.gz ... in the dataset's imagesTr folder)

Label values follow the dataset (see dataset.json). Prints the volume of each label in mL.
Then: python export_mesh.py --labels pred/case1.nii.gz --preset stroke ...  (3D model for VR)
"""

import argparse
import os

import nibabel as nib
import numpy as np
import torch

from data_3d import DatasetInfo, get_transforms
from models_3d import build_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--images", nargs="+", help="Channel files in the training order")
    ap.add_argument("--case", help="Case id inside the dataset's imagesTr/imagesTs")
    ap.add_argument("--dataset", help="Dataset folder (default: the one stored in the checkpoint)")
    ap.add_argument("--out", required=True, help="Output .nii.gz")
    ap.add_argument("--overlap", type=float, default=0.5)
    args = ap.parse_args()

    from monai import transforms as T
    from monai.inferers import sliding_window_inference

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(args.ckpt, map_location=device, weights_only=False)
    info = DatasetInfo(args.dataset or ck["dataset"])
    roi, spacing = tuple(ck["roi"]), tuple(ck["spacing"])
    model = build_model(ck["model_name"], len(info.channels), info.out_channels, roi).to(device).eval()
    model.load_state_dict(ck["model"])

    if args.images:
        files = args.images
    elif args.case:
        files = []
        for i in range(len(info.channels)):
            hits = [p for s in ("Ts", "Tr") if os.path.exists(p := os.path.join(info.root, f"images{s}", f"{args.case}_{i:04d}.nii.gz"))]
            files.append(hits[0] if hits else None)
        if None in files:
            raise SystemExit(f"Case {args.case} not found in {info.root}/imagesTs or imagesTr")
    else:
        raise SystemExit("Give --images or --case")
    if len(files) != len(info.channels):
        raise SystemExit(f"Model needs {len(info.channels)} channels {info.channels}, got {len(files)} files")

    pre = get_transforms(info, spacing, roi, "infer")
    d = pre({"image": files})
    x = d["image"].unsqueeze(0).to(device)
    with torch.no_grad(), torch.autocast(device.type, enabled=device.type == "cuda"):
        logits = sliding_window_inference(x, roi, 2, model, overlap=args.overlap).float()
    probs = torch.sigmoid(logits) if info.regions else torch.softmax(logits, 1)
    d["pred"] = probs[0].cpu()
    # undo resampling/reorientation so the result lines up with the original scan
    d = T.Invertd(keys="pred", transform=pre, orig_keys="image", nearest_interp=False)(d)
    lab = info.probs_to_labels(d["pred"].numpy())

    ref = nib.load(files[0])
    if lab.shape != ref.shape[:3]:
        raise SystemExit(f"Shape mismatch after inversion {lab.shape} vs {ref.shape[:3]}")
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    nib.save(nib.Nifti1Image(lab, ref.affine), args.out)
    ml = float(np.prod(ref.header.get_zooms()[:3])) / 1000.0
    if info.regions:     # e.g. whole tumour = all tumour labels together
        masks = [np.isin(lab, v) for v in info.region_values]
    else:
        masks = [lab == v for v in range(1, info.out_channels)]
    vols = {name: round(float(m.sum() * ml), 2) for name, m in zip(info.names, masks)}
    print(f"Saved {args.out} | volume (mL): {vols}  - research use only, not for diagnosis")


if __name__ == "__main__":
    main()
