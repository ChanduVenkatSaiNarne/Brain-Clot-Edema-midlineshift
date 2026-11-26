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

**Decision.** Before changing the model, find out which scans fail (error analysis by bleed size and type). Then try the cheap fixes for sensitivity: lower probability threshold and smaller `--min-px`, chosen on the validation scans, not the test scans.

## Planned experiments

| Order | Question | Plan |
|---|---|---|
| 1 | Is the score stable across splits? | 5-fold cross-validation by scan |
| 2 | Does 3D beat 2D for measuring volume? | SegResNet and nnU-Net on the `3d` branch, A100 |
| 3 | Can interpretable features predict the bleed type? | Volume, side, density and shape from the segmentation into XGBoost, explained with SHAP |
| 4 | Which bleeds are missed? | Error analysis by bleed type and size |
