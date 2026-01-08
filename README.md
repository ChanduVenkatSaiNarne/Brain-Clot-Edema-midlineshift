# 🧠 U-Net Based Midline Shift & Edema Segmentation from Brain MRI

This repository contains a deep learning pipeline for **automatic midline shift / edema segmentation** from 2D brain MRI slices using a **U-Net architecture** implemented in **TensorFlow/Keras**. The project is designed for medical image analysis workflows, particularly in **stroke and neuroimaging applications**.

---

## 📌 Project Overview

Manual delineation of edema and midline shift regions in brain MRI is time-consuming and subject to inter-observer variability. This project addresses the problem by leveraging a **U-Net-based semantic segmentation model** to automatically generate accurate binary masks from grayscale MRI slices.

### Key Features

* ✅ Single-class binary segmentation (edema / midline shift)
* ✅ U-Net encoder–decoder architecture
* ✅ Dice Coefficient–based evaluation
* ✅ Robust preprocessing and normalization
* ✅ Train/validation split with model checkpointing
* ✅ Designed for medical imaging datasets

---

## 🧠 Model Architecture

The segmentation model is based on **U-Net**, a widely used convolutional neural network for biomedical image segmentation.

**Architecture Highlights:**

* Encoder–decoder structure
* Skip connections for spatial feature preservation
* Convolution + ReLU blocks
* Sigmoid activation for binary mask output

---

## 📂 Dataset Structure

The model expects **2D MRI slices** and corresponding **binary segmentation masks** stored as image files.

```
dataset/
├── edema_original/
│   ├── image_001.png
│   ├── image_002.png
│   └── ...
└── midline_shift_mask/
    ├── image_001.png
    ├── image_002.png
    └── ...
```

* Images: Grayscale MRI slices
* Masks: Binary masks (0 = background, 1 = edema/midline shift)
* Image–mask filenames must match

---

## ⚙️ Preprocessing Pipeline

The following preprocessing steps are applied:

1. **Grayscale loading** of MRI slices
2. **Resizing** to `256 × 256`
3. **Intensity normalization** to `[0, 1]`
4. **Binary thresholding** of masks
5. **Channel expansion** for CNN compatibility
6. **Train–validation split (80/20)**

---

## 🏋️ Training Configuration

| Parameter     | Value                             |
| ------------- | --------------------------------- |
| Input Size    | 256 × 256 × 1                     |
| Batch Size    | 8                                 |
| Optimizer     | Adam                              |
| Learning Rate | 1e-4                              |
| Loss Function | Binary Cross-Entropy / Dice-based |
| Epochs        | Up to 300 (with Early Stopping)   |
| Metrics       | Dice Coefficient                  |

---

## 📊 Evaluation Metric

### Dice Coefficient

Measures overlap between predicted and ground-truth masks:

```
Dice = (2 × |Prediction ∩ GroundTruth|) / (|Prediction| + |GroundTruth|)
```

* Dice = 1.0 → Perfect segmentation
* Dice ≥ 0.9 → Clinically strong performance

---

## 🚀 Training the Model

```bash
python train_unet.py
```

The trained model will be saved as:

```
best_midline_unet.h5
```

---

## 🖼️ Inference & Visualization

The trained model can be used to:

* Generate binary segmentation masks
* Overlay predictions on MRI slices
* Assist clinicians in identifying midline shift and edema regions

(Optional visualization can be implemented using Matplotlib or OpenCV.)

---

## 🧩 Tech Stack

* **Python 3.x**
* **TensorFlow / Keras**
* **OpenCV**
* **NumPy**
* **Scikit-learn**

---

## 📈 Future Enhancements

* 🔹 Dice + BCE combined loss
* 🔹 Data augmentation (Albumentations)
* 🔹 IoU metric integration
* 🔹 PyTorch + MONAI implementation
* 🔹 3D MRI volume support
* 🔹 Midline deviation measurement (mm)
* 🔹 Clinical UI integration (Streamlit / PyQt)

---

---

## ⚠️ Disclaimer

This project is intended for **research and educational purposes only** and should not be used for direct clinical decision-making without regulatory approval.

---

⭐ *If you find this repository useful, please consider giving it a star!*

