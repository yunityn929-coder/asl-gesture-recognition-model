"""
TASK A — camera-degradation augmentation (additive, new files only).

Simulates the production app's known live-camera degradation on the existing
deduplicated ASL-HG training images:
  1. Slight Gaussian blur (motion/focus softness).
  2. Mild Gaussian sensor noise.
  3. JPEG re-encode/decode at quality 85 (matches Android's
     YuvImage.compressToJpeg(..., 85, ...) used by the live camera pipeline
     before MediaPipe ever sees a frame).

Degraded copies are saved to raw_dataset/degraded_train/<CLASS>/<file> (does
NOT touch raw_dataset/asl_processed/train), then run through the SAME
HandLandmarker + normalize_landmarks pipeline as preprocessing.py to produce
X_degraded.npy / y_degraded.npy / person_ids_degraded.npy — new files,
row-aligned, tagged source="degraded" (vs "clean" for the original arrays).

Person id is parsed directly from the filename convention P<id>_<CLASS>_<n>.jpg
(confirmed present on every raw training file), so the existing person-disjoint
CV grouping carries over unchanged to the degraded rows.
"""
import cv2
import numpy as np
import os
import sys
import json

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
RAW_DATASET = os.path.join(BASE_DIR, "raw_dataset", "asl_processed", "train")
DEGRADED_DIR = os.path.join(BASE_DIR, "raw_dataset", "degraded_train")
OUT_DIR     = BASE_DIR

ALL_CLASSES = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") + [str(i) for i in range(10)]

RNG_SEED = 42
BLUR_SIGMA_RANGE = (0.4, 1.0)
NOISE_SIGMA_RANGE = (3.0, 8.0)
JPEG_QUALITY = 85


def degrade_image(img_bgr, rng):
    """Blur -> sensor noise -> JPEG q85 encode/decode roundtrip.
    Returns (degraded_bgr_array, jpeg_bytes)."""
    sigma_blur = rng.uniform(*BLUR_SIGMA_RANGE)
    blurred = cv2.GaussianBlur(img_bgr, (0, 0), sigmaX=sigma_blur)

    sigma_noise = rng.uniform(*NOISE_SIGMA_RANGE)
    noise = rng.normal(0.0, sigma_noise, size=blurred.shape)
    noisy = np.clip(blurred.astype(np.float64) + noise, 0, 255).astype(np.uint8)

    ok, enc = cv2.imencode(".jpg", noisy, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    if not ok:
        return None, None
    jpeg_bytes = enc.tobytes()
    degraded_bgr = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    return degraded_bgr, jpeg_bytes


def parse_person_id(filename):
    # convention: P<id>_<CLASS>_<n>.jpg
    return filename.split("_")[0]


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


def normalize_landmarks(landmarks_list):
    landmarks = np.array([[lm.x, lm.y, lm.z] for lm in landmarks_list])
    landmarks -= landmarks[0].copy()
    scale = np.linalg.norm(landmarks[9])
    if scale < 1e-6:
        return None
    landmarks /= scale
    return landmarks.flatten()


def main():
    from mediapipe import Image, ImageFormat

    rng = np.random.default_rng(RNG_SEED)

    if not os.path.exists(RAW_DATASET):
        print(f"[ERROR] raw dataset not found: {RAW_DATASET}")
        sys.exit(1)

    X_rows, y_rows, person_rows = [], [], []
    summary = {}
    total_saved, total_rejected = 0, 0

    with build_detector() as detector:
        for idx, class_name in enumerate(ALL_CLASSES):
            in_dir = os.path.join(RAW_DATASET, class_name)
            out_dir = os.path.join(DEGRADED_DIR, class_name)
            os.makedirs(out_dir, exist_ok=True)

            if not os.path.isdir(in_dir):
                summary[class_name] = {"saved": 0, "rejected": 0, "reason": "folder_missing"}
                continue

            images = sorted(f for f in os.listdir(in_dir)
                             if f.lower().endswith((".jpg", ".jpeg", ".png")))

            saved, rejected = 0, 0
            reasons = {"no_hand": 0, "bad_scale": 0, "unreadable": 0, "encode_fail": 0}

            print(f"[{idx+1:02d}/{len(ALL_CLASSES)}] {class_name}: {len(images)} source images")

            for fname in images:
                src_path = os.path.join(in_dir, fname)
                img_bgr = cv2.imread(src_path)
                if img_bgr is None:
                    rejected += 1
                    reasons["unreadable"] += 1
                    continue

                degraded_bgr, jpeg_bytes = degrade_image(img_bgr, rng)
                if degraded_bgr is None:
                    rejected += 1
                    reasons["encode_fail"] += 1
                    continue

                with open(os.path.join(out_dir, fname), "wb") as f:
                    f.write(jpeg_bytes)

                img_rgb = cv2.cvtColor(degraded_bgr, cv2.COLOR_BGR2RGB)
                mp_image = Image(image_format=ImageFormat.SRGB, data=img_rgb)
                result = detector.detect(mp_image)

                if not result.hand_landmarks or len(result.hand_landmarks) == 0:
                    rejected += 1
                    reasons["no_hand"] += 1
                    continue

                landmarks = normalize_landmarks(result.hand_landmarks[0])
                if landmarks is None or landmarks.shape != (63,):
                    rejected += 1
                    reasons["bad_scale"] += 1
                    continue

                X_rows.append(landmarks)
                y_rows.append(class_name)
                person_rows.append(parse_person_id(fname))
                saved += 1

            total_saved += saved
            total_rejected += rejected
            rate = (saved / (saved + rejected) * 100) if (saved + rejected) > 0 else 0
            summary[class_name] = {"saved": saved, "rejected": rejected, "rate": rate, "reasons": reasons}
            print(f"    saved={saved} rejected={rejected} rate={rate:.1f}% reasons={reasons}")

    X = np.array(X_rows, dtype=np.float64)
    y = np.array(y_rows, dtype="<U1")
    person_ids = np.array(person_rows, dtype=object)

    np.save(os.path.join(OUT_DIR, "X_degraded.npy"), X)
    np.save(os.path.join(OUT_DIR, "y_degraded.npy"), y)
    np.save(os.path.join(OUT_DIR, "person_ids_degraded.npy"), person_ids)

    report = {
        "total_source_images": total_saved + total_rejected,
        "total_saved": total_saved,
        "total_rejected": total_rejected,
        "overall_rate": (total_saved / (total_saved + total_rejected) * 100) if (total_saved + total_rejected) > 0 else 0,
        "blur_sigma_range": BLUR_SIGMA_RANGE,
        "noise_sigma_range": NOISE_SIGMA_RANGE,
        "jpeg_quality": JPEG_QUALITY,
        "per_class": summary,
        "output_shapes": {"X_degraded": list(X.shape), "y_degraded": list(y.shape), "person_ids_degraded": list(person_ids.shape)},
    }
    with open(os.path.join(OUT_DIR, "degrade_taskA_report.json"), "w") as f:
        json.dump(report, f, indent=2)

    print("\n" + "=" * 60)
    print(f"TOTAL saved={total_saved} rejected={total_rejected} rate={report['overall_rate']:.1f}%")
    print(f"X_degraded.npy shape={X.shape}")
    print("Report: degrade_taskA_report.json")


if __name__ == "__main__":
    main()
