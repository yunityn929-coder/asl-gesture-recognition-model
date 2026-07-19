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
