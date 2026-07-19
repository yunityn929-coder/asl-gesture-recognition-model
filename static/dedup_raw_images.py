"""
Permanently fixes the P5/P6 duplicate-image contamination at its ROOT
CAUSE: the raw source JPGs in static/raw_dataset/asl_processed/train/I/
and .../train/V/ contain literal byte-identical file copies between P5
and P6 (confirmed via MD5). Even though the current X_clean.npy happens
to contain zero duplicate rows (random capping already dropped one side
of most pairs by chance), leaving the duplicate raw files in place means
any future re-run of preprocessing.py/data_cleaning.ipynb (different cap,
different seed, more samples) could reintroduce the contamination.

For each cross-person duplicate group (identical landmark content, hence
identical image content, across different person folders): keep the
lowest-numbered person's file, MOVE (not delete) the other copy/copies to
static/raw_dataset/duplicates_removed/<CLASS>/<filename>, preserving the
data instead of destroying it.

Read-mostly with respect to the dataset: does not touch X_clean.npy,
y_clean.npy, person_ids.npy, or any model file.
"""
import cv2
import numpy as np
import os
import re
import json
import shutil

from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions
from mediapipe import Image, ImageFormat

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
RAW_DATASET = os.path.join(BASE_DIR, "raw_dataset", "asl_processed", "train")
QUARANTINE  = os.path.join(BASE_DIR, "raw_dataset", "duplicates_removed")
OUT_REPORT  = os.path.join(BASE_DIR, "dedup_raw_report.json")

TARGET_CLASSES = ["I", "V"]  # only classes with known contamination (see duplicate_scan_report.json)
FNAME_RE = re.compile(r"^P(\d+)_")


def normalize_landmarks(landmarks_list):
    landmarks = np.array([[lm.x, lm.y, lm.z] for lm in landmarks_list])
    landmarks -= landmarks[0].copy()
    scale = np.linalg.norm(landmarks[9])
    if scale < 1e-6:
        return None
    landmarks /= scale
    return landmarks.flatten()


def build_full_lookup(class_name, detector):
    input_dir = os.path.join(RAW_DATASET, class_name)
    lookup = {}
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
        if not result.hand_landmarks:
            continue
        lm = normalize_landmarks(result.hand_landmarks[0])
        if lm is None or lm.shape != (63,):
            continue
        lookup.setdefault(lm.tobytes(), []).append(img_file)
    return lookup


def main():
    model_path = os.path.join(BASE_DIR, "hand_landmarker.task")
    options = HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=model_path),
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        running_mode=vision.RunningMode.IMAGE
    )

    report = {}

    with HandLandmarker.create_from_options(options) as detector:
        for class_name in TARGET_CLASSES:
            lookup = build_full_lookup(class_name, detector)
            moved = []
            for key, fnames in lookup.items():
                if len(fnames) < 2:
                    continue
                persons = sorted(set(FNAME_RE.match(f).group(1) for f in fnames), key=int)
                if len(persons) <= 1:
                    continue  # within-person duplicate, not this script's concern
                keep_person = persons[0]
                kept_one = False
                for f in fnames:
                    p = FNAME_RE.match(f).group(1)
                    if p == keep_person and not kept_one:
                        kept_one = True
                        continue
                    src = os.path.join(RAW_DATASET, class_name, f)
                    dst_dir = os.path.join(QUARANTINE, class_name)
                    os.makedirs(dst_dir, exist_ok=True)
                    dst = os.path.join(dst_dir, f)
                    shutil.move(src, dst)
                    moved.append(f)

            report[class_name] = {
                "files_moved": moved,
                "count": len(moved),
                "remaining_images": len([
                    f for f in os.listdir(os.path.join(RAW_DATASET, class_name))
                    if f.lower().endswith((".jpg", ".jpeg", ".png"))
                ]),
            }
            print(f"{class_name}: moved {len(moved)} duplicate files to quarantine, "
                  f"{report[class_name]['remaining_images']} images remain")

    with open(OUT_REPORT, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved: {OUT_REPORT}")
    print(f"Quarantined files (recoverable, not deleted): {QUARANTINE}")


if __name__ == "__main__":
    main()
