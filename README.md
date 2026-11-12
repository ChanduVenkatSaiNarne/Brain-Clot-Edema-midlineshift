# U-Net Segmentation of Clot, Edema and Midline Shift on Brain MRI Slices

A research prototype that segments **clot**, **edema** and the **midline structure** on 2D axial brain MRI slices with three U-Net models (TensorFlow/Keras), then reports simple measurements.

> ⚠️ Research and education only. Not validated for clinical use.

## Repository structure

| File | Purpose |
|---|---|
| `train_unet.py` | Train one U-Net (`--target clot`, `edema` or `midline`) |
| `predict.py` | Run all three models on images; save overlays and `measurements.csv` |
| `measurements.py` | Areas, hemisphere side and midline-shift estimate (pure NumPy/OpenCV) |
| `DeepLearningModelsh5files/` | Earlier trained models (`.h5`) and sample images |
| `requirements.txt` | Python dependencies |

## Data layout

Images and masks are 2D slices with **matching filenames**:

```
images/  P01_s12.png  P01_s13.png ...
masks/   P01_s12.png  P01_s13.png ...   (white = target, black = background)
```

## Training

```bash
pip install -r requirements.txt
python train_unet.py --images images/ --masks masks/ --target edema --patient-regex '^(P\d+)_'
```

| Setting | Value |
|---|---|
| Input | 256 × 256 × 1, intensity scaled to [0, 1] |
| Model | 4-level U-Net with batch normalisation (base 32 filters) |
| Loss | Binary cross-entropy + Dice loss |
| Optimiser | Adam, learning rate 1e-4, halved on plateau |
| Epochs | Up to 300, early stopping (patience 30) on validation Dice |
| Augmentation | Left–right flip, small brightness/contrast changes |
| Split | 70 / 15 / 15 train / validation / **held-out test** |

- Exact duplicate images are removed before splitting.
- Use `--patient-regex` so that all slices from one patient stay in the same split. Without it, slices from one patient can leak between train and test and inflate scores.
- The best model (by validation Dice) is saved as `models/unet_<target>_best.keras`, and the **test-set Dice** is written to `models/test_results_<target>.json`.

## Prediction

```bash
python predict.py --images DeepLearningModelsh5files/new_images/new_images \
  --clot-model DeepLearningModelsh5files/unet_clot_segmentation_single_class_1000.h5 \
  --edema-model DeepLearningModelsh5files/unet_clot_segmentation_single_class_edema_1000.h5 \
  --midline-model DeepLearningModelsh5files/unet_clot_segmentation_single_class_midline_1000.h5
```

Outputs per image: an overlay figure and a row in `predictions/measurements.csv`.

## How measurements work (and their limits)

- **Areas** are reported in pixels. They are converted to mm² **only** if you pass the real `--pixel-spacing` (mm) from the original DICOM/NIfTI header; JPG/PNG files do not store it.
- **Hemisphere** uses **radiological convention** (image left = patient's right) and the head's midline, not the image centre. Use `--neurological` if your images follow the other convention.
- **Midline shift** is a **distance**: the largest horizontal distance between the predicted midline structure and the ideal midline (centre of the head outline). This is an approximation of the clinical measurement and has not been validated against expert readings.
- Clot and edema overlap is counted once in the lesion percentage.
- No symptoms, diagnoses or severity grades are produced.

## Known limitations

- 2D slices only; no 3D volumes.
- Head outline uses a simple intensity threshold and may fail on unusual backgrounds.
- The older `.h5` models were trained with the previous script (BCE loss, no early stopping, random image-level split). Retrain with `train_unet.py` before reporting results.

## Future work

- 3D volumes from NIfTI/DICOM with real voxel spacing
- Validation of midline-shift estimates against expert measurements
- PyTorch / MONAI implementation
