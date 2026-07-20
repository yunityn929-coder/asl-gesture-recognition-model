# Evaluation methodology — read before retraining this model

## Never use a random row-level train/val split for this dataset

`X_clean.npy`/`y_clean.npy` contain MediaPipe hand-landmark samples from
~10 volunteers (tracked in `static/person_ids.npy`), each of whom
contributed hundreds of near-duplicate frames per class. A random
row-level `train_test_split` (even `stratify`d on the class label) puts
near-identical frames of the *same person's hand* into both train and
validation. The model then partly memorizes per-person hand geometry
instead of learning the general shape of each sign, and validation
accuracy comes out misleadingly high — the original random-split
evaluation reported **100.00% validation accuracy** on this exact model
architecture, which did not hold up when tested on new people.

**Always evaluate with a person-disjoint split**: hold out entire
volunteers' data for validation, never a random subset of rows. Load
`static/person_ids.npy` (aligned row-for-row with `X_clean.npy`/
`y_clean.npy`) and split on that.

## Honest numbers on record

- Single P5+P6 holdout (first diagnostic): **94.79%** val accuracy —
  still leakage-free, but a single held-out pair is a noisy estimate.
- 5-fold person-disjoint cross-validation (`P1+P2`, `P3+P4`, `P5+P6`,
  `P7+P8`, `P9+P10`), raw+engineered-features+augmentation variant:
  **87.44% ± 4.54pp** across folds (range 80.75%–93.61%). **This is the
  number to cite as "how well this model generalizes to a new person."**
  See `static/model/summary_fold_*.json` for the per-fold detail and
  `static/model/summary_v2_candidate.json` for the final production
  model's training record.
- Classes `O` and `C` are genuinely weak (not just low-sample) —
  they collapse into each other and into `M` under a person-disjoint
  split. `O` in particular stayed weak (~49% recall, high variance)
  even after augmentation and engineered curl/distance features; fixing
  it likely needs more real-world data collection, not just better
  training on the existing dataset.

## P5/P6 duplicate-image contamination (fixed, but know the history)

The raw ASL-HG source images for classes `I` and `V` contained literal
byte-identical file duplicates between volunteers P5 and P6 (confirmed
via MD5) — 68 and 64 files respectively, quarantined to
`static/raw_dataset/duplicates_removed/` (see `dedup_raw_report.json`).
The already-extracted `X_clean.npy` happened to contain zero duplicate
*rows* from this at the time it was checked (random capping had already
dropped one side of most pairs by chance) — verified exhaustively, see
`dedup_report.json` — but the raw fix is what makes this permanent: any
future re-run of `preprocessing.py`/`data_cleaning.ipynb` from the raw
images will no longer be able to reintroduce it.

If you ever re-run the k-fold split and P5/P6 land in different folds,
that's fine post-fix — the contamination was in the raw files, which are
now deduplicated. It's kept as one fold here mostly for continuity with
the original diagnostic, not because it's still required for safety.

## Where things live

- `static/person_ids.npy` — person id per row, aligned to `X_clean.npy`/`y_clean.npy`.
- `static/landmark_features.py` — augmentation (rotation/scale/noise) + engineered features (finger curl angles, fingertip-palm distances).
- `static/train_ablation.py` — the training script all of the above numbers came from (`--held-out`, `--variant`, `--augment`, `--final` flags).
- `static/model/mlp_model_v2_candidate.h5` / `.tflite` — candidate retrain on all 10 persons' deduplicated data. **Not yet promoted to production** — swapping `mlp_model.tflite` is a separate, deliberate step.

## v3 experiment: camera-degradation augmentation + external digit dataset (still not promoted)

Two additive data experiments, both evaluated with `static/train_combined.py`
(group-disjoint CV: each ASL-HG person as its own fold, plus a 6th fold that
holds out an entire external dataset):

- **Camera degradation (Task A)**: `static/degrade_and_extract_taskA.py`
  re-encodes every raw training image through mild Gaussian blur + noise +
  JPEG q85 (matching the app's `YuvImage.compressToJpeg` step), re-extracts
  landmarks, tagged `source=degraded` alongside the original `source=clean`
  rows (see `X_degraded.npy`/`y_degraded.npy`/`person_ids_degraded.npy`).
  **Result: training on clean+degraded together produces validation accuracy
  that's essentially identical whether the validation sample is clean or
  degraded** (e.g. fold P7+P8: 81.0% clean vs 82.6% degraded) — the model
  already handles this specific, mild degradation fine. This means JPEG
  compression + mild blur/noise is **not** the main driver of the live-camera
  accuracy drop; the remaining domain gap is more likely camera distance,
  angle, background, and lighting variety, none of which this experiment adds.
- **External digit dataset (Task B)**: `static/extract_ardamavi_digits.py`
  pulls in the Apache-2.0 ardamavi/Sign-Language-Digits-Dataset (218 people).
  Manual handshape spot-check found only digits **0/1/2/4/5 use the same
  handshape convention as ours**; **3/6/7/8/9 use a different plain
  finger-count convention** (no thumb-touch) and are excluded from training
  (see `handshape_verified_ardamavi_digits.npy`, `extract_ardamavi_report.json`).
  Holding out this entire external source as its own CV fold (never seen in
  training) gave only **53.1% overall accuracy** — but broken down: digits 4
  and 5 transferred almost perfectly (99–100% recall), while 0/1/2 collapsed
  into their well-known ASL homograph letters (**0→O, 1→D, 2→V** — real ASL
  handshape ambiguities, not a labeling bug), showing the model doesn't yet
  disambiguate these under a camera/framing shift it's never seen.
- **Combined effect on the existing 5 person-pair folds**: 88.45% ± 2.71pp
  (vs the original 87.44% ± 4.54pp clean-only baseline) — noticeably lower
  fold-to-fold variance, but **not a uniform per-class win**: O improved
  (49.2% → 57.2% recall, still noisy across folds), but T regressed hard
  (86.7% → 45.8%), and C/M/Q/V/Z also regressed. See
  `static/model/summary_taskA_degraded_only_*.json`,
  `static/model/summary_taskAB_combined_*.json`.
- Candidate files: `static/model/mlp_model_v3_candidate.h5`/`.tflite` +
  `label_encoder_v3_candidate.pkl` — trained on ALL clean+degraded+verified-
  ardamavi rows. **Not promoted to production or over v2_candidate** — this
  is a mixed result, not a clear win, and needs a call on whether the O/0
  improvement is worth the T/C/Q/V/Z regression before going further.
