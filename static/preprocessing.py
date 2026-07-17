"""
ASL Gesture Recognition - Static Preprocessing Script
Extracts and normalizes MediaPipe hand landmarks from ASL-HG dataset images.
Saves each valid sample as a (63,) .npy file.

Run this on WINDOWS (not WSL) — uses OpenCV image reading, no camera needed.
Compatible with MediaPipe 0.10+

Dataset structure expected:
  static/raw_dataset/asl_processed/train/<CLASS>/<image>.jpg

Output structure:
  static/dataset/<CLASS>/<index>.npy

Classes: A-Z (including J, Z) + 0-9 = 36 classes total
"""

import cv2
import mediapipe as mp
import numpy as np
import os
import sys

# ── Configuration ─────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
RAW_DATASET = os.path.join(BASE_DIR, "raw_dataset", "asl_processed", "train")
OUTPUT_DIR  = os.path.join(BASE_DIR, "dataset")

# All 36 classes — A-Z + 0-9
ALL_CLASSES = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") + [str(i) for i in range(10)]

# ── MediaPipe 0.10+ setup ─────────────────────────────────────────────────────
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions
from mediapipe import Image, ImageFormat


# ── Helpers ───────────────────────────────────────────────────────────────────
def normalize_landmarks(landmarks_list):
    """
    Normalize 21 hand landmarks (x, y, z).
    Returns (63,) numpy array or None if invalid.
    """
    landmarks = np.array([[lm.x, lm.y, lm.z] for lm in landmarks_list])  # (21, 3)

    # Step 1: Subtract wrist (landmark 0) — position invariant
    landmarks -= landmarks[0].copy()

    # Step 2: Scale by wrist-to-middle-MCP (landmark 9) distance — scale invariant
    scale = np.linalg.norm(landmarks[9])
    if scale < 1e-6:
        return None
    landmarks /= scale

    return landmarks.flatten()  # (63,)


def process_class(class_name, detector):
    """
    Process all images for one class.
    Returns (saved_count, rejected_count, rejection_reasons)
    """
    input_dir  = os.path.join(RAW_DATASET, class_name)
    output_dir = os.path.join(OUTPUT_DIR, class_name)
    os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(input_dir):
        return 0, 0, {"folder_missing": 1}

    images = [f for f in os.listdir(input_dir)
              if f.lower().endswith((".jpg", ".jpeg", ".png"))]

    if not images:
        return 0, 0, {"no_images": 1}

    saved    = 0
    rejected = 0
    reasons  = {"no_hand": 0, "bad_scale": 0, "unreadable": 0}

    for img_file in images:
        img_path = os.path.join(input_dir, img_file)

        # Read image
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            rejected += 1
            reasons["unreadable"] += 1
            continue

        # Convert to RGB numpy array then to MediaPipe Image
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        mp_image = Image(image_format=ImageFormat.SRGB, data=img_rgb)

        # Run detection
        result = detector.detect(mp_image)

        if not result.hand_landmarks or len(result.hand_landmarks) == 0:
            rejected += 1
            reasons["no_hand"] += 1
            continue

        # Use first hand only
        landmarks = normalize_landmarks(result.hand_landmarks[0])

        if landmarks is None:
            rejected += 1
            reasons["bad_scale"] += 1
            continue

        if landmarks.shape != (63,):
            rejected += 1
            reasons["bad_scale"] += 1
            continue

        # Save as .npy
        np.save(os.path.join(output_dir, f"{saved}.npy"), landmarks)
        saved += 1

    return saved, rejected, reasons


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("ASL-HG Dataset Preprocessing (MediaPipe 0.10+)")
    print("Extracting MediaPipe landmarks -> .npy files")
    print("=" * 60)
    print(f"Input:   {RAW_DATASET}")
    print(f"Output:  {OUTPUT_DIR}")
    print(f"Classes: {len(ALL_CLASSES)} ({', '.join(ALL_CLASSES)})")
    print("=" * 60)

    if not os.path.exists(RAW_DATASET):
        print(f"\n[ERROR] Raw dataset folder not found:\n  {RAW_DATASET}")
        sys.exit(1)

    # Download hand landmarker model if not present
    model_path = os.path.join(BASE_DIR, "hand_landmarker.task")
    if not os.path.exists(model_path):
        print("\nDownloading MediaPipe hand landmarker model...")
        import urllib.request
        url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
        urllib.request.urlretrieve(url, model_path)
        print(f"  Saved to: {model_path}")

    # Set up MediaPipe 0.10+ HandLandmarker
    options = HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=model_path),
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        running_mode=vision.RunningMode.IMAGE
    )

    summary        = {}
    total_saved    = 0
    total_rejected = 0

    with HandLandmarker.create_from_options(options) as detector:
        for idx, class_name in enumerate(ALL_CLASSES):
            print(f"\n[{idx+1:02d}/{len(ALL_CLASSES)}] Processing: {class_name}")

            saved, rejected, reasons = process_class(class_name, detector)
            total = saved + rejected
            rate  = (saved / total * 100) if total > 0 else 0

            summary[class_name] = {"saved": saved, "rejected": rejected, "rate": rate}
            total_saved    += saved
            total_rejected += rejected

            status = "OK  " if rate >= 80 else "WARN" if rate >= 50 else "LOW "
            print(f"  [{status}] Saved: {saved} | Rejected: {rejected} | Rate: {rate:.1f}%")
            if rejected > 0:
                print(f"    Reasons: {reasons}")

    # ── Report ────────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("PREPROCESSING COMPLETE — SUMMARY REPORT")
    print("=" * 60)
    print(f"{'Class':<8} {'Saved':>6} {'Rejected':>9} {'Rate':>7}")
    print("-" * 36)

    low_classes = []
    for cls, stats in summary.items():
        flag = " <- LOW" if stats["rate"] < 80 else ""
        if stats["rate"] < 80:
            low_classes.append(cls)
        print(f"{cls:<8} {stats['saved']:>6} {stats['rejected']:>9} {stats['rate']:>6.1f}%{flag}")

    grand_total = total_saved + total_rejected
    overall     = (total_saved / grand_total * 100) if grand_total > 0 else 0
    print("-" * 36)
    print(f"{'TOTAL':<8} {total_saved:>6} {total_rejected:>9} {overall:>6.1f}%")
    print(f"\nTotal .npy files created: {total_saved}")
    print(f"Saved to: {os.path.abspath(OUTPUT_DIR)}")

    if low_classes:
        print(f"\n[WARN] Low extraction rate classes (<80%): {low_classes}")
    else:
        print("\n[OK] All classes >= 80% extraction rate")

    print("\nNext step: Run data_cleaning.ipynb in WSL.")


if __name__ == "__main__":
    main()
