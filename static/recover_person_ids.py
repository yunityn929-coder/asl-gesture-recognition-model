"""
DIAGNOSTIC SCRIPT — recovers person_id per row of X_clean.npy/y_clean.npy.

Person ID is not tracked anywhere past the raw JPG filenames
(static/raw_dataset/asl_processed/train/<CLASS>/P{n}_{CLASS}_{num}.jpg).
preprocessing.py saves landmarks as static/dataset/<CLASS>/<sequential_index>.npy,
discarding the source filename.

Approach: re-run the exact same MediaPipe HandLandmarker extraction used by
preprocessing.py over the raw images, keyed by the resulting (63,) landmark
array's raw bytes. Since data_cleaning.ipynb only filters/subsamples rows
(never modifies values), every row in X_clean.npy is a byte-identical copy of
one image's extracted landmark vector. So we can recover the source filename
(and person id) for each row via exact value match, independent of any
directory-listing order.

Does NOT modify X_clean.npy, y_clean.npy, or any production model file.
Writes: static/person_ids.npy, static/person_id_match_report.json
"""
import cv2
import mediapipe as mp
import numpy as np
import os
import re
import json
import sys
import time

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
RAW_DATASET = os.path.join(BASE_DIR, "raw_dataset", "asl_processed", "train")
X_PATH      = os.path.join(BASE_DIR, "X_clean.npy")
Y_PATH      = os.path.join(BASE_DIR, "y_clean.npy")
OUT_PERSON  = os.path.join(BASE_DIR, "person_ids.npy")
OUT_REPORT  = os.path.join(BASE_DIR, "person_id_match_report.json")

ALL_CLASSES = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") + [str(i) for i in range(10)]
FNAME_RE = re.compile(r"^P(\d+)_")

from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions
from mediapipe import Image, ImageFormat


def normalize_landmarks(landmarks_list):
    landmarks = np.array([[lm.x, lm.y, lm.z] for lm in landmarks_list])
    landmarks -= landmarks[0].copy()
    scale = np.linalg.norm(landmarks[9])
    if scale < 1e-6:
        return None
    landmarks /= scale
    return landmarks.flatten()


def build_class_lookup(class_name, detector):
    """Return dict: landmark_bytes -> filename, for all extractable images in this class."""
    input_dir = os.path.join(RAW_DATASET, class_name)
    lookup = {}
    collisions = 0
    if not os.path.exists(input_dir):
        return lookup, collisions

    images = [f for f in os.listdir(input_dir)
              if f.lower().endswith((".jpg", ".jpeg", ".png"))]

    for img_file in images:
        img_path = os.path.join(input_dir, img_file)
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            continue
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        mp_image = Image(image_format=ImageFormat.SRGB, data=img_rgb)
        result = detector.detect(mp_image)
        if not result.hand_landmarks or len(result.hand_landmarks) == 0:
            continue
        landmarks = normalize_landmarks(result.hand_landmarks[0])
        if landmarks is None or landmarks.shape != (63,):
            continue
        key = landmarks.tobytes()
        if key in lookup:
            collisions += 1
        lookup[key] = img_file

    return lookup, collisions


def main():
    model_path = os.path.join(BASE_DIR, "hand_landmarker.task")
    if not os.path.exists(model_path):
        print(f"[ERROR] Missing model: {model_path}")
        sys.exit(1)

    X = np.load(X_PATH)
    y = np.load(Y_PATH)
    print(f"Loaded X_clean: {X.shape}, y_clean: {y.shape}")

    options = HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=model_path),
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        running_mode=vision.RunningMode.IMAGE
    )

    person_ids = np.full(len(X), "UNKNOWN", dtype=object)
    report = {}

    with HandLandmarker.create_from_options(options) as detector:
        for ci, class_name in enumerate(ALL_CLASSES):
            t0 = time.time()
            class_row_idx = np.where(y == class_name)[0]
            if len(class_row_idx) == 0:
                continue

            lookup, collisions = build_class_lookup(class_name, detector)

            matched = 0
            unmatched = 0
            for ridx in class_row_idx:
                key = X[ridx].tobytes()
                fname = lookup.get(key)
                if fname is None:
                    unmatched += 1
                    continue
                m = FNAME_RE.match(fname)
                if not m:
                    unmatched += 1
                    continue
                person_ids[ridx] = f"P{m.group(1)}"
                matched += 1

            dt = time.time() - t0
            report[class_name] = {
                "rows": int(len(class_row_idx)),
                "matched": matched,
                "unmatched": unmatched,
                "lookup_size": len(lookup),
                "lookup_collisions": collisions,
                "seconds": round(dt, 1),
            }
            print(f"[{ci+1:02d}/{len(ALL_CLASSES)}] {class_name}: "
                  f"rows={len(class_row_idx)} matched={matched} unmatched={unmatched} "
                  f"lookup_size={len(lookup)} collisions={collisions} ({dt:.1f}s)")

    total_rows = len(X)
    total_matched = int((person_ids != "UNKNOWN").sum())
    print(f"\nTOTAL: matched {total_matched}/{total_rows} "
          f"({total_matched/total_rows*100:.2f}%)")

    np.save(OUT_PERSON, person_ids)
    with open(OUT_REPORT, "w") as f:
        json.dump({
            "total_rows": total_rows,
            "total_matched": total_matched,
            "match_rate": total_matched / total_rows,
            "per_class": report,
        }, f, indent=2)

    print(f"Saved: {OUT_PERSON}")
    print(f"Saved: {OUT_REPORT}")


if __name__ == "__main__":
    main()
