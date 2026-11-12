"""Train a 2D U-Net for one target (clot, edema or midline).

Fixes compared with the original notebook script:
  * duplicate images are removed before splitting (e.g. "12.jpg" and "12 - Copy.jpg")
  * optional patient-level split (--patient-regex) so slices from one patient
    never appear in both training and test data
  * separate held-out TEST set (70/15/15); the reported Dice comes from it
  * Dice + BCE loss (plain BCE lets small targets collapse to "all background")
  * light data augmentation
  * early stopping + saving the BEST model, not the last epoch
  * deeper U-Net with batch normalisation

Example (Colab):
  python train_unet.py --images /content/drive/MyDrive/only_edema/edema_original \
      --masks /content/drive/MyDrive/only_edema/midline_shift_mask --target midline
"""

import argparse
import hashlib
import json
import os
import re

import cv2
import numpy as np

IMG_SIZE = 256
EXTS = (".png", ".jpg", ".jpeg")


def load_pairs(image_dir, mask_dir):
    """Load matching image/mask pairs, skipping exact duplicate images."""
    images, masks, names, seen = [], [], [], set()
    for name in sorted(os.listdir(image_dir)):
        if not name.lower().endswith(EXTS):
            continue
        img_path, mask_path = os.path.join(image_dir, name), os.path.join(mask_dir, name)
        if not os.path.exists(mask_path):
            print(f"Skipping {name}: no matching mask")
            continue
        with open(img_path, "rb") as f:
            digest = hashlib.md5(f.read()).hexdigest()
        if digest in seen:
            print(f"Skipping {name}: duplicate of an earlier image")
            continue
        seen.add(digest)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if img is None or mask is None:
            print(f"Skipping {name}: unreadable file")
            continue
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA) / 255.0
        mask = cv2.resize(mask, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST)
        images.append(img[..., None].astype(np.float32))
        masks.append((mask > 127).astype(np.float32)[..., None])
        names.append(name)
    return np.array(images), np.array(masks), names


def split_indices(names, patient_regex, seed=42):
    """70/15/15 split, grouped by patient ID when a regex is given."""
    from sklearn.model_selection import GroupShuffleSplit, train_test_split

    idx = np.arange(len(names))
    if patient_regex:
        groups = []
        for n in names:
            m = re.search(patient_regex, n)
            groups.append(m.group(1) if m else n)
        groups = np.array(groups)
        gss = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=seed)
        train, rest = next(gss.split(idx, groups=groups))
        gss2 = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=seed)
        val_r, test_r = next(gss2.split(rest, groups=groups[rest]))
        return train, rest[val_r], rest[test_r]
    print("WARNING: no --patient-regex given, splitting by image. If several slices come "
          "from the same patient, test scores will be over-optimistic.")
    train, rest = train_test_split(idx, test_size=0.30, random_state=seed)
    val, test = train_test_split(rest, test_size=0.50, random_state=seed)
    return train, val, test


def build_model(base=32):
    import tensorflow as tf
    L = tf.keras.layers

    def block(x, f):
        for _ in range(2):
            x = L.Conv2D(f, 3, padding="same", use_bias=False)(x)
            x = L.BatchNormalization()(x)
            x = L.Activation("relu")(x)
        return x

    inputs = tf.keras.Input((IMG_SIZE, IMG_SIZE, 1))
    skips, x = [], inputs
    for i in range(4):                      # encoder: 256 -> 16
        x = block(x, base * 2 ** i)
        skips.append(x)
        x = L.MaxPooling2D()(x)
    x = L.Dropout(0.3)(block(x, base * 16))  # bottleneck
    for i in reversed(range(4)):            # decoder
        x = L.Conv2DTranspose(base * 2 ** i, 2, strides=2, padding="same")(x)
        x = block(L.Concatenate()([x, skips[i]]), base * 2 ** i)
    outputs = L.Conv2D(1, 1, activation="sigmoid")(x)
    return tf.keras.Model(inputs, outputs)


def dice_coefficient(y_true, y_pred, smooth=1.0):
    import tensorflow as tf
    y_true = tf.reshape(tf.cast(y_true, tf.float32), [-1])
    y_pred = tf.reshape(tf.cast(y_pred, tf.float32), [-1])
    inter = tf.reduce_sum(y_true * y_pred)
    return (2.0 * inter + smooth) / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + smooth)


def dice_bce_loss(y_true, y_pred):
    import tensorflow as tf
    bce = tf.reduce_mean(tf.keras.losses.binary_crossentropy(y_true, y_pred))
    return bce + (1.0 - dice_coefficient(y_true, y_pred))


def make_dataset(x, y, batch, augment):
    import tensorflow as tf

    def aug(img, mask):
        # Left-right flip is applied to image AND mask together.
        if tf.random.uniform(()) > 0.5:
            img, mask = tf.image.flip_left_right(img), tf.image.flip_left_right(mask)
        img = tf.clip_by_value(tf.image.random_brightness(img, 0.1), 0.0, 1.0)
        img = tf.clip_by_value(tf.image.random_contrast(img, 0.9, 1.1), 0.0, 1.0)
        return img, mask

    ds = tf.data.Dataset.from_tensor_slices((x, y))
    if augment:
        ds = ds.shuffle(len(x), seed=42).map(aug, num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(batch).prefetch(tf.data.AUTOTUNE)


def per_image_dice(y_true, y_pred, threshold=0.5):
    scores = []
    for t, p in zip(y_true, y_pred):
        t, p = t > 0.5, p > threshold
        denom = t.sum() + p.sum()
        scores.append(1.0 if denom == 0 else 2.0 * np.logical_and(t, p).sum() / denom)
    return np.array(scores)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True)
    ap.add_argument("--masks", required=True)
    ap.add_argument("--target", required=True, choices=["clot", "edema", "midline"])
    ap.add_argument("--out", default="models")
    ap.add_argument("--patient-regex", default=None,
                    help=r"Regex with one group that captures the patient ID from the filename, e.g. '^(P\d+)_'")
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--patience", type=int, default=30)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    args = ap.parse_args()

    import tensorflow as tf
    tf.keras.utils.set_random_seed(42)

    x, y, names = load_pairs(args.images, args.masks)
    tr, va, te = split_indices(names, args.patient_regex)
    print(f"Loaded {len(names)} unique pairs -> train {len(tr)}, val {len(va)}, test {len(te)}")

    model = build_model()
    model.compile(optimizer=tf.keras.optimizers.Adam(args.lr),
                  loss=dice_bce_loss, metrics=[dice_coefficient])

    os.makedirs(args.out, exist_ok=True)
    best_path = os.path.join(args.out, f"unet_{args.target}_best.keras")
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(best_path, monitor="val_dice_coefficient",
                                           mode="max", save_best_only=True),
        tf.keras.callbacks.EarlyStopping(monitor="val_dice_coefficient", mode="max",
                                         patience=args.patience, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=10),
        tf.keras.callbacks.CSVLogger(os.path.join(args.out, f"history_{args.target}.csv")),
    ]
    model.fit(make_dataset(x[tr], y[tr], args.batch, augment=True),
              validation_data=make_dataset(x[va], y[va], args.batch, augment=False),
              epochs=args.epochs, callbacks=callbacks)

    pred = model.predict(x[te], batch_size=args.batch)
    scores = per_image_dice(y[te], pred)
    result = {"target": args.target, "n_test": int(len(te)),
              "test_dice_mean": float(scores.mean()), "test_dice_std": float(scores.std()),
              "test_images": [names[i] for i in te]}
    with open(os.path.join(args.out, f"test_results_{args.target}.json"), "w") as f:
        json.dump(result, f, indent=2)
    print(f"Held-out TEST Dice ({args.target}): {scores.mean():.3f} ± {scores.std():.3f} "
          f"over {len(te)} images. Best model saved to {best_path}")


if __name__ == "__main__":
    main()
