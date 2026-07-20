"""
TASK B — extract MediaPipe landmarks from the ardamavi Sign Language Digits
Dataset (Apache-2.0, Turkey Ankara Ayranci Anadolu High School, 218 students,
10 samples/student, https://github.com/ardamavi/Sign-Language-Digits-Dataset).

HANDSHAPE VERIFICATION (done manually before writing this script, see report):
Visual spot-check of this dataset's Examples/example_<digit>.JPG against our
own ASL-HG raw images for the same digit found that classes 0, 1, 2, 4, 5 use
the same handshape convention as our existing data, but classes 3, 6, 7, 8, 9
do NOT: ardamavi uses a plain sequential finger-count (3 = index+middle+ring,
no thumb; 6/7/8 all show near-identical 3-4-finger counts with the thumb
tucked away) instead of the real ASL numeral handshapes our data already uses
(3 = thumb+index+middle; 6/7/8/9 = thumb touching a specific fingertip while
the other fingers extend). Folding those five classes in as-is would teach
the model to associate two different real handshapes with the same label.

So: this script extracts landmarks for ALL 10 classes (for a complete, honest
audit trail), but tags every row with both source="ARDAMAVI_DIGITS" AND a
handshape_verified flag (True for 0/1/2/4/5, False for 3/6/7/8/9). The combined
training script only uses handshape_verified=True rows.
"""
import cv2
import numpy as np
import os
import json

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "raw_dataset", "ardamavi_digits", "repo", "Dataset")

VERIFIED_CLASSES = {"0", "1", "2", "4", "5"}
MISMATCHED_CLASSES = {"3", "6", "7", "8", "9"}


def normalize_landmarks(landmarks_list):
    landmarks = np.array([[lm.x, lm.y, lm.z] for lm in landmarks_list])
    landmarks -= landmarks[0].copy()
    scale = np.linalg.norm(landmarks[9])
    if scale < 1e-6:
        return None
    landmarks /= scale
    return landmarks.flatten()


def build_detector():
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision
    from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions

    model_path = os.path.join(BASE_DIR, "hand_landmarker.task")
    options = HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=model_path),
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        running_mode=vision.RunningMode.IMAGE,
    )
    return HandLandmarker.create_from_options(options)


def main():
    from mediapipe import Image, ImageFormat

    X_rows, y_rows, verified_rows = [], [], []
    summary = {}
    total_saved, total_rejected = 0, 0

    with build_detector() as detector:
        for cls in sorted(os.listdir(DATASET_DIR)):
            in_dir = os.path.join(DATASET_DIR, cls)
            if not os.path.isdir(in_dir):
                continue
            images = sorted(f for f in os.listdir(in_dir)
                             if f.lower().endswith((".jpg", ".jpeg", ".png")))
            saved, rejected = 0, 0
            reasons = {"no_hand": 0, "bad_scale": 0, "unreadable": 0}

            for fname in images:
                img_bgr = cv2.imread(os.path.join(in_dir, fname))
                if img_bgr is None:
                    rejected += 1
                    reasons["unreadable"] += 1
                    continue
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                result = detector.detect(Image(image_format=ImageFormat.SRGB, data=img_rgb))
                if not result.hand_landmarks:
                    rejected += 1
                    reasons["no_hand"] += 1
                    continue
                landmarks = normalize_landmarks(result.hand_landmarks[0])
                if landmarks is None or landmarks.shape != (63,):
                    rejected += 1
                    reasons["bad_scale"] += 1
                    continue
                X_rows.append(landmarks)
                y_rows.append(cls)
                verified_rows.append(cls in VERIFIED_CLASSES)
                saved += 1

            total_saved += saved
            total_rejected += rejected
            rate = (saved / (saved + rejected) * 100) if (saved + rejected) > 0 else 0
            summary[cls] = {"saved": saved, "rejected": rejected, "rate": rate,
                             "reasons": reasons, "handshape_verified": cls in VERIFIED_CLASSES}
            print(f"{cls}: saved={saved} rejected={rejected} rate={rate:.1f}% "
                  f"verified={cls in VERIFIED_CLASSES}")

    X = np.array(X_rows, dtype=np.float64)
    y = np.array(y_rows, dtype="<U1")
    verified = np.array(verified_rows, dtype=bool)
    source = np.array(["ARDAMAVI_DIGITS"] * len(X_rows), dtype=object)

    np.save(os.path.join(BASE_DIR, "X_ardamavi_digits.npy"), X)
    np.save(os.path.join(BASE_DIR, "y_ardamavi_digits.npy"), y)
    np.save(os.path.join(BASE_DIR, "source_ardamavi_digits.npy"), source)
    np.save(os.path.join(BASE_DIR, "handshape_verified_ardamavi_digits.npy"), verified)

    n_verified = int(verified.sum())
    report = {
        "total_saved": total_saved,
        "total_rejected": total_rejected,
        "verified_classes": sorted(VERIFIED_CLASSES),
        "mismatched_classes": sorted(MISMATCHED_CLASSES),
        "mismatch_reason": (
            "ardamavi's 3/6/7/8/9 use a plain sequential finger-count "
            "handshape (e.g. 3 = index+middle+ring, no thumb) that visually "
            "conflicts with our ASL-HG dataset's real ASL numeral handshapes "
            "for the same digits (3 = thumb+index+middle; 6-9 = thumb "
            "touching a specific fingertip). Only 0/1/2/4/5 matched on "
            "manual spot-check and are used in training."
        ),
        "rows_usable_for_training": n_verified,
        "rows_excluded_mismatched": total_saved - n_verified,
        "per_class": summary,
    }
    with open(os.path.join(BASE_DIR, "extract_ardamavi_report.json"), "w") as f:
        json.dump(report, f, indent=2)

    print(f"\nTOTAL saved={total_saved} rejected={total_rejected}")
    print(f"Usable for training (handshape-verified): {n_verified}")
    print(f"Excluded (mismatched convention): {total_saved - n_verified}")


if __name__ == "__main__":
    main()
