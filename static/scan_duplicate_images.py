"""
DIAGNOSTIC SCRIPT — scans all 36 classes for images whose extracted landmark
vectors are bit-identical across different source images (same class).
Flags cases where the duplicate pair/group spans more than one person id,
since that means the "different person" label is not actually independent
data (e.g. P5 vs P6 image duplication found in classes I and V).

Read-only with respect to production data. Writes:
  static/duplicate_scan_report.json
"""
import cv2
import numpy as np
import os
import re
import json
import time

from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions
from mediapipe import Image, ImageFormat

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
RAW_DATASET = os.path.join(BASE_DIR, "raw_dataset", "asl_processed", "train")
OUT_REPORT  = os.path.join(BASE_DIR, "duplicate_scan_report.json")

ALL_CLASSES = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") + [str(i) for i in range(10)]
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
    if not os.path.exists(input_dir):
        return lookup
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
        key = lm.tobytes()
        lookup.setdefault(key, []).append(img_file)
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
    pair_summary = {}  # (personA, personB) -> total cross-person dup count across all classes

    with HandLandmarker.create_from_options(options) as detector:
        for ci, class_name in enumerate(ALL_CLASSES):
            t0 = time.time()
            lookup = build_full_lookup(class_name, detector)
            cross_groups = []
            for key, fnames in lookup.items():
                if len(fnames) < 2:
                    continue
                persons = sorted(set(FNAME_RE.match(f).group(1) for f in fnames))
                if len(persons) > 1:
                    cross_groups.append({"files": fnames, "persons": persons})
                    for i in range(len(persons)):
                        for j in range(i + 1, len(persons)):
                            pair = tuple(sorted([persons[i], persons[j]], key=int))
                            pair_summary[pair] = pair_summary.get(pair, 0) + 1

            report[class_name] = {
                "cross_person_duplicate_groups": len(cross_groups),
                "examples": cross_groups[:5],
            }
            dt = time.time() - t0
            print(f"[{ci+1:02d}/{len(ALL_CLASSES)}] {class_name}: "
                  f"cross_person_dup_groups={len(cross_groups)} ({dt:.1f}s)")

    pair_summary_str = {f"P{a}-P{b}": v for (a, b), v in
                         sorted(pair_summary.items(), key=lambda x: -x[1])}

    print("\n=== CROSS-PERSON DUPLICATE PAIR SUMMARY (all classes) ===")
    for pair, count in pair_summary_str.items():
        print(f"  {pair}: {count} duplicate groups")

    with open(OUT_REPORT, "w") as f:
        json.dump({"per_class": report, "pair_summary": pair_summary_str}, f, indent=2)
    print(f"\nSaved: {OUT_REPORT}")


if __name__ == "__main__":
    main()
