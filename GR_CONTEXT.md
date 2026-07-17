# AI_PROJECT_CONTEXT.md
> This file is for AI assistants to understand the project architecture, decisions, and context.
> Do NOT modify this file manually. Update it when architecture or decisions change.
> Last updated: 2026-07-17

---

## Project Overview

- **Project name:** HiASL — ASL Learning App (FYP 2026)
- **Repo name:** `asl-gesture-recognition-model`
- **Parent project:** `HiASL_BSE_FYP_2026` (C:\HiASL_BSE_FYP_2026\)
- **Goal:** Recognize ASL hand gestures in real-time for a learning/quiz app
- **Scope:** A–Z alphabet + digits 0–9, one hand only
- **Future scope:** May expand to simple/common words

---

## Key Architecture Decision: MLP Only

**Decision: Single MLP model for all 36 classes (A–Z + 0–9)**

| Type | Signs | Count | Model |
|------|-------|-------|-------|
| All signs | A–Z + 0–9 | 36 classes | MLP only |

**Why MLP only (not MLP + LSTM):**
- J and Z are technically dynamic signs (motion-defined in real ASL)
- However, the dataset contains static images of J and Z
- Decision made to keep FYP scope simple — one model, no routing logic needed
- J and Z will likely have lower MLP accuracy due to inconsistent static images in dataset
- This is a documented known limitation in the FYP report
- FYP report justification: "J and Z were included in the MLP model for scope simplicity.
  As static images of J and Z capture inconsistent motion frames, accuracy for these two
  classes may be lower. Dynamic sequence modeling (LSTM) is identified as future work."

**Why NOT LSTM:**
- Adds routing complexity (how to decide static vs dynamic at inference time)
- Requires separate dynamic data collection (recording sequences)
- For a learning app quiz flow, MLP-only is simpler and sufficient for FYP

---

## Reference Architecture

Based on a validated Malaysian Sign Language (MySL) recognition system built by a peer.
- MySL used: 2 hands (126 features), MLP for 50 static classes, LSTM for 40 dynamic classes
- This project adapts the same MLP pipeline for 1 hand and 36 classes
- Callbacks, layer sizes, and training config directly follow the reference

---

## Feature Extraction

- **Tool:** MediaPipe Hands (`hand_landmarker.task` model, MediaPipe 0.10+)
- **API:** MediaPipe 0.10+ Tasks API (`HandLandmarker`, not old `mp.solutions.hands`)
- **Hands:** 1 hand only (`num_hands=1`)
- **Raw features:** 21 landmarks × 3 coords (x, y, z) = **63 features per frame**
- **Detection confidence:** 0.5 (preprocessing), 0.7 (live inference)
- **Tracking confidence:** 0.5 (preprocessing), 0.6 (live inference)

### Normalization (CRITICAL)
```python
# Step 1: Subtract wrist (landmark 0) — removes position dependency
landmarks -= landmarks[0].copy()

# Step 2: Divide by wrist-to-middle-MCP (landmark 9) distance — removes scale dependency
scale = np.linalg.norm(landmarks[9])
if scale < 1e-6:
    return None  # skip degenerate detections
landmarks /= scale

# Output: (63,) array — position and scale invariant
```
**Why normalization matters:** Without it, the same gesture signed at different
positions or hand sizes produces completely different numbers. This was the likely
cause of a previous failed Google Colab attempt.

---

## Dataset

### Source
- **Dataset:** ASL-HG (American Sign Language Hand Gesture Image Dataset)
- **URL:** https://data.mendeley.com/datasets/j4y5w2c8w9/1
- **Version used:** ASL_Processed_Images.zip
  - MediaPipe-segmented hand regions, clean backgrounds
  - Pre-organized into train/test splits (80/20)
- **Original size:** 36,000 images, 36 classes (A–Z + 0–9), 1,000 images per class
- **Image format:** .jpg, named as P{person}_{class}_{number}.jpg
- **Collected from:** 10 volunteers, indoor + outdoor environments, diverse skin tones

**Why ASL-HG over Kaggle datasets:**
- One download covers exact project scope (A–Z + 0–9)
- Balanced (1,000 per class), consistent image style
- Pre-processed version available (cleaner MediaPipe extraction)
- No need to merge two separate datasets (avoids cross-dataset bias)

### Raw Dataset Structure
```
static/raw_dataset/asl_processed/
├── train/
│   ├── A/   ← P{person}_A_{number}.jpg
│   ├── B/
│   └── ... (36 class folders: A-Z + 0-9)
└── test/
```

### Preprocessing (preprocessing.py — run on Windows)
- **Script:** `static/preprocessing.py`
- **MediaPipe model:** `hand_landmarker.task` (auto-downloaded on first run)
- **API used:** MediaPipe 0.10+ `HandLandmarker` Tasks API
- Processes all images in `train/` folder
- Skips images where no hand detected
- Normalizes landmarks (wrist origin + scale)
- Saves valid samples as `(63,)` .npy files

**Extraction results per class:**
| Class | Extracted | Class | Extracted | Class | Extracted |
|-------|-----------|-------|-----------|-------|-----------|
| A | 353 | M | 717 | 0 | 464 |
| B | 800 | N | 454 | 1 | 783 |
| C | 308 | O | 239 | 2 | 800 |
| D | 800 | P | 753 | 3 | 800 |
| E | 579 | Q | 491 | 4 | 800 |
| F | 800 | R | 800 | 5 | 800 |
| G | 712 | S | 545 | 6 | 800 |
| H | 791 | T | 338 | 7 | 800 |
| I | 790 | U | 799 | 8 | 797 |
| J | 768 | V | 800 | 9 | 800 |
| K | 800 | W | 800 | | |
| L | 798 | X | 613 | | |
| | | Y | 704 | | |
| | | Z | 794 | | |

**Total extracted:** 24,790 .npy files
**Low rate classes (<80%):** A, C, E, N, O, Q, S, T, X, 0
- Caused by MediaPipe failing to detect hand in some images

### Data Cleaning (data_cleaning.ipynb — run in WSL)
**Strategy: Option D — Cap at 600, keep all if below 600**

| Step | Action |
|------|--------|
| 1. Load | Load all .npy files, verify shape=(63,), skip NaN/Inf |
| 2. Outliers | Remove samples with z-score > 3.0 per class |
| 3. Cap | Randomly keep 600 per class; keep all if < 600 |
| 4. Verify | Final shape, NaN, Inf checks |
| 5. Save | X_clean.npy and y_clean.npy to static/ |

**Why cap at 600:**
- Healthy MLP training size; avoids extreme class imbalance
- Low-sample classes (O=239, C=308, A=353, T=338) kept as-is
- After first training run, check which low-sample classes have poor accuracy
- Supplement those specific classes with self-collected samples if needed

**Output files:**
- `static/X_clean.npy` — shape: (N, 63)
- `static/y_clean.npy` — shape: (N,) string labels

---

## MLP Model Architecture

### Model
```python
model = Sequential([
    BatchNormalization(input_shape=(63,)),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(36, activation='softmax')    # 36 classes: A-Z + 0-9
])
```

### Layer Decisions
| Layer | Value | Reason |
|-------|-------|--------|
| Input | 63 | 1 hand × 21 landmarks × 3 coords |
| BatchNorm first | Yes | Stabilizes training; mirrors reference architecture |
| Dense 1 | 128 units, relu | Same as reference; sufficient capacity for 63-dim input |
| Dropout 1 | 0.3 | Prevent overfitting; same as reference |
| Dense 2 | 64 units, relu | Progressive dimensionality reduction; same as reference |
| Dropout 2 | 0.3 | Prevent overfitting; same as reference |
| Output | 36, softmax | 26 letters (A–Z) + 10 digits (0–9) |

### Compilation
```python
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
```

### Training
```python
model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=200,
    batch_size=32,
    callbacks=[early_stopping, reduce_lr]
)
```

### Callbacks (identical to reference architecture)
```python
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=20,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=7,
    min_lr=1e-6,
    verbose=1
)
```

| Callback | Purpose |
|----------|---------|
| EarlyStopping patience=20 | Stops if val_loss doesn't improve for 20 epochs |
| restore_best_weights=True | Rolls back to best epoch weights, not last epoch |
| ReduceLROnPlateau factor=0.5 | Halves LR when val_loss plateaus for 7 epochs |
| min_lr=1e-6 | Floor so LR never becomes uselessly tiny |

### Saved model
- `static/model/mlp_model.h5`

---

## Inference / App Integration

### Interaction Flow
```
App loads quiz question (e.g. "Sign the letter B")
        ↓
User presses Start
        ↓
Camera captures frames via MediaPipe (live)
        ↓
Wait for landmarks to stabilize (~5 frames below movement threshold)
        ↓
Extract + normalize single frame → (63,) array
        ↓
MLP predicts → (36,) softmax → argmax → class label + confidence
        ↓
App shows result to user
```

- **One button only** (Start) — no Stop button needed
- Auto-classifies once hand is stable
- Single MLP handles all 36 classes

### Model Output
- Shape: `(36,)` softmax probabilities
- `argmax` → predicted class index
- Map index to label using `label_encoder.classes_`
- Confidence score = `max(softmax)` — consider threshold (e.g. > 0.7 to show result)

---

## Environment

| Item | Detail |
|------|--------|
| Windows Python | 3.12.10 — data collection + preprocessing only |
| WSL | Ubuntu, Python 3.11 venv (`asl_env`) — training only |
| TensorFlow | CPU only (no CUDA drivers in WSL) |
| MediaPipe | 0.10.35 — uses Tasks API, NOT old mp.solutions.hands |
| Jupyter | Installed in asl_env, accessed via browser at localhost:8888 |
| Save format | .npy for datasets, .h5 for models |

### Why split Windows/WSL
- Camera (`cv2.VideoCapture`) doesn't work in WSL2 without USB passthrough
- Windows handles: data collection scripts, preprocessing (image reading)
- WSL handles: Jupyter notebooks, model training (TensorFlow)
- Shared filesystem: C:\HiASL_BSE_FYP_2026\ ↔ /mnt/c/HiASL_BSE_FYP_2026/

### MediaPipe 0.10+ Important Note
Old API (`mp.solutions.hands`) no longer works in 0.10+.
Must use Tasks API:
```python
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions
from mediapipe import Image, ImageFormat
```
Requires `hand_landmarker.task` model file downloaded from:
https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task

---

## File Structure

```
asl-gesture-recognition-model/
├── AI_PROJECT_CONTEXT.md              ← this file (AI memory)
├── README.md                          ← human-readable overview
├── requirements.txt                   ← WSL pip dependencies
│
├── static/
│   ├── preprocessing.py               ← run on Windows
│   ├── data_cleaning.ipynb            ← run in WSL
│   ├── mlp_training.ipynb             ← run in WSL (TODO)
│   ├── hand_landmarker.task           ← MediaPipe model (auto-downloaded)
│   ├── X_clean.npy                    ← cleaned features (post data_cleaning)
│   ├── y_clean.npy                    ← cleaned labels (post data_cleaning)
│   ├── class_distribution.png         ← generated by data_cleaning.ipynb
│   ├── raw_dataset/
│   │   └── asl_processed/
│   │       ├── train/  (A-Z + 0-9 folders, .jpg images)
│   │       └── test/
│   ├── dataset/
│   │   ├── A/ ... Z/  (36 class folders, .npy files)
│   │   └── 0/ ... 9/
│   └── model/
│       └── mlp_model.h5               ← trained model (TODO)
│
├── dynamic/                           ← reserved for future LSTM expansion
│   ├── dataset/
│   │   ├── J/
│   │   └── Z/
│   └── model/
│
└── asl_env/                           ← WSL venv (gitignored)
```

---

## Scripts Summary

| Script | Where to run | Purpose |
|--------|-------------|---------|
| `static/preprocessing.py` | Windows PowerShell | Extract MediaPipe landmarks from images → .npy |
| `static/data_cleaning.ipynb` | WSL Jupyter | Clean, balance, save X_clean/y_clean |
| `static/mlp_training.ipynb` | WSL Jupyter | Train MLP, save mlp_model.h5 |

---

## Training Results

### MLP Model — Final Results (2026-07-17)

| Metric | Value |
|--------|-------|
| Stopped at epoch | 170 |
| Best epoch | 150 (restored by EarlyStopping) |
| Best val_loss | 0.0005 |
| Best val_accuracy | **100.00%** |
| Validation samples | 3,963 |
| Classes below 80% accuracy | **None** |

**Per-class results:** All 36 classes achieved 1.00 precision, recall, and f1-score.
- Low-sample classes (O=45, C=59, A=66, T=64 val samples) still achieved 100% accuracy
- J and Z (static images in MLP) achieved 100% — no issues on validation set

**Training curves:** Healthy convergence, no overfitting. Val loss consistently
below or matching train loss. Both converge cleanly around epoch 150.

**Saved model files:**
- `static/model/mlp_model.h5` — trained MLP model
- `static/model/label_encoder.pkl` — sklearn LabelEncoder for class mapping
- `static/model/training_curves.png` — loss + accuracy plots
- `static/model/confusion_matrix.png` — 36x36 confusion matrix (perfect diagonal)

**Note on 100% val accuracy:** Achievable because ASL-HG uses clean MediaPipe-segmented
images with normalized landmarks. Real-world webcam accuracy will be lower due to
lighting/angle variation — this is expected and acceptable.

---

## Progress Tracker

- [x] WSL environment set up (Python 3.11, asl_env)
- [x] Dependencies installed (mediapipe, tensorflow, seaborn, etc.)
- [x] GitHub repo created and cloned
- [x] Folder structure created
- [x] Dataset downloaded (ASL-HG Mendeley)
- [x] preprocessing.py written and run → 24,790 .npy files created
- [x] data_cleaning.ipynb completed → X_clean.npy (19,812 × 63) + y_clean.npy saved
- [x] mlp_training.ipynb written and run
- [x] MLP model trained — 100% val accuracy, stopped epoch 170, best epoch 150
- [x] mlp_model.h5 + label_encoder.pkl saved to static/model/
- [ ] Model integrated into asl-sign-recognition-app

---

## Known Limitations

- J and Z use static images in MLP — validated to work on dataset but may confuse
  with similar handshapes (I vs J) in real-world webcam use
- CPU-only training (no GPU in WSL) — training took ~15-30 min but completed fine
- Low-sample classes (O=225, C=296, T=319, A=331) — achieved 100% val accuracy
  but may be less robust in real-world use due to fewer training examples
- Real-world accuracy expected to be lower than 100% due to webcam variation

---

## Open Questions / Future Decisions

- Confidence threshold for inference (what softmax score = show result to user?)
- App framework for model integration (TensorFlow.js / TFLite / Python backend?)
- Future: LSTM for J and Z if real-world MLP accuracy is unacceptable
- Future: expand to simple words (dynamic signs)
- Future: supplement low-sample classes (O, C, A, T) if real-world accuracy is poor