# Research log

Each entry records the question, what was run, what came out and what it changed. Raw outputs are in [results/](results/). The 30 test scans stay the same in every entry, so results can be compared directly.

## 2026-11-13: clearing out the prototype

**Question.** Can the original prototype models be trusted?

**Finding.** No. The three `.h5` models (clot, edema, midline) had been trained with an image-level split and no held-out test set, so there was no honest way to report how well they worked. They were removed, and the code was rewritten in PyTorch around a public dataset with expert outlines (BHSD) and a split by scan.

**Next.** Train a clot model on BHSD and report results on held-out scans.

## 2026-11-17: quick test on BHSD (2 epochs)

**Question.** Does the whole pipeline (download, slicing, training, test metrics) work on real data?

**Setup.** U-Net++ on an ImageNet-pretrained EfficientNet-B3 (13.62 M parameters), three CT windows as input, 2 epochs on a Colab T4, about 85 s per epoch.

**Result** ([raw output](results/2026-11-17_quick_test.txt)). On the 30 test scans: 3D Dice 0.550 mean and 0.659 median, bleed volume error 14.8 mL with r = 0.83, slice sensitivity 0.874 and specificity 0.779.

**What it showed.** The pipeline works end to end, and after only two epochs the model already finds most bleeds and tracks their volume. Specificity is the weak point: about one in five bleed-free slices gets a false spot. The mean Dice sits below the median because a few hard scans, usually with small bleeds, pull it down.

**Next.** Full training. If specificity stays low, raise the minimum speck size (`--min-px`), which only needs prediction to be rerun.

## 2026-11-26: full training

**Setup.** The same model on an L4 GPU, up to 60 epochs with early stopping. The data has 192 scans (134 training, 28 validation, 30 test); the training scans contain 1,633 slices with bleeding and 2,840 without.

**Result.** Early stopping after 15 epochs with no validation improvement. Test (30 scans): 3D Dice 0.628 (median 0.756), volume error 12.2 mL (r = 0.90), slice sensitivity 0.756, specificity 0.965.

**What it tells me.**
- Compared with the 2-epoch test, overlap and volume both improved (Dice +0.08, volume error 14.8 → 12.2 mL, r 0.83 → 0.90).
- Specificity jumped from 0.78 to 0.97 but sensitivity dropped from 0.87 to 0.76. The model became cautious: it misses some bleeding slices, most likely the small, thin or faint ones at the edges of a bleed.
- Mean Dice is well below the median, so a handful of scans score very low. Those are probably small bleeds, where a few missed pixels cost a lot of Dice.

**Example test scan** (`results/figures/2026-11-26_test_scan_example.png`, red = model, green = expert). A large haemorrhage on four slices (12 to 15). The outlines almost coincide on every slice, including the darker pockets inside the clot. One small false positive on slice 13 at the edge of the skull near the temporal bone, where bright bone next to brain can look like blood. This is the kind of speck that a higher `--min-px` or a brain mask would remove, so the threshold tuning has to balance it against small real bleeds.

**Decision.** Before changing the model, find out which scans fail (error analysis by bleed size and type). Then try the cheap fixes for sensitivity: lower probability threshold and smaller `--min-px`, chosen on the validation scans, not the test scans.

## 2026-11-26: threshold tuning and error analysis

**Setup.** `tune_clot.py` on the full-training model. 40 threshold / min_px pairs scored on the 28 validation scans; the best pair applied once to the 30 test scans. Raw output: `results/2026-11-26_threshold_tuning.txt`.

**Result.** Best on validation: threshold 0.35, min_px 5.

| Test (30 scans) | Default (0.5, 10 px) | Tuned (0.35, 5 px) |
|---|---|---|
| 3D Dice | 0.628 (median 0.756) | 0.632 (median 0.747) |
| Volume error | 12.2 mL, r = 0.90 | 10.9 mL, r = 0.91 |
| Slice sensitivity / specificity | 0.756 / 0.965 | 0.811 / 0.925 |

Dice by bleed size (tuned): under 5 mL 0.324 (7 scans), 5 to 30 mL 0.719 (12 scans), over 30 mL 0.732 (11 scans). Four of the five worst scans have bleeds under 7 mL.

**What it tells me.**
- The threshold barely moves Dice (+0.004) but improves volume error and finds 5% more bleeding slices. Speck size matters little below 20 px.
- The real problem is small bleeds. Medium and large bleeds are already around 0.72 to 0.73; small ones are at 0.32. On a 0.2 mL bleed, a 3.8 mL false positive gives Dice 0.03, so Dice is also a harsh measure there. I should report detection (found / missed) for small bleeds as well as Dice.
- At 256 px the whole field of view, including air and the table, is squeezed in, so a small bleed is only a handful of pixels.

**Decision.** Next run: crop each scan to the head before resizing (`prepare_bhsd.py --crop`). Same model, same settings, same test scans; only the preprocessing changes. On a synthetic check, a small bleed went from 50 to 220 pixels with nothing cut off.

## Plan to raise the score (one change per run, compared on validation)

| Step | Change | Why | Cost |
|---|---|---|---|
| A | Tune threshold and `--min-px` on validation (`tune_clot.py`) | Win back sensitivity without retraining | done: small gain |
| A2 | Crop to the head before resizing (`--crop`) | Small bleeds get 2 to 4 times more pixels | 1 run, next |
| B | Tversky / focal loss that weights missed bleed more | Model is too cautious (sens 0.76) | 1 run |
| C | 2.5D input: neighbouring slices as extra channels | Faint bleeds are clearer with context above and below | 1 run |
| D | 384 or 512 px instead of 256 | Small bleeds lose pixels at 256 | 1 run, slower |
| E | Average 3 to 5 models (seeds or CV folds) | Usually a steady gain | several runs |

## Planned experiments

| Order | Question | Plan |
|---|---|---|
| 1 | Is the score stable across splits? | 5-fold cross-validation by scan |
| 2 | Does 3D beat 2D for measuring volume? | SegResNet and nnU-Net on the `3d` branch, A100 |
| 3 | Can interpretable features predict the bleed type? | Volume, side, density and shape from the segmentation into XGBoost, explained with SHAP |
| 4 | Which bleeds are missed? | Error analysis by bleed type and size |
