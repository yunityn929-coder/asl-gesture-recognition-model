"""
ASL Gesture Recognition - Live Webcam Model Evaluation
Runs the trained MLP (mlp_model.tflite) against your laptop webcam in
real time, using the same MediaPipe Tasks API config as the Android app
(HandLandmarkDetector.kt: num_hands=1, min_hand/presence/tracking
confidence=0.5, RunningMode.IMAGE) and the same normalization as
preprocessing.py, so results are as close as possible to what the phone
app will see — without needing a phone at all.

Run this on WINDOWS (not WSL) — needs camera access, same as
static_data_collection.py.

You declare the sign you're about to hold (ground truth) by pressing its
key; the script logs the model's live prediction against it and exports a
CSV in the schema docs/analysis/analyze_recognition_log.py expects
(same schema the in-app Recognition Test screen produces — files from both
sources can be mixed in one analysis run).

Controls:
  0-9, A-Z  - set the target sign (ground truth) to that key
  SPACE     - start/pause logging (buffer persists across pauses)
  Shift+E   - export CSV now (doesn't stop the session)
  ESC       - quit (auto-exports if there's unsaved logged data)

Usage:
    python static/live_recognition_test.py --session-tag laptop_daylight
"""

import argparse
import csv
import os
import time

import cv2
import numpy as np

from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions
from mediapipe import Image, ImageFormat

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL = os.path.join(BASE_DIR, "model", "mlp_model.tflite")
DEFAULT_LABEL_ENCODER = os.path.join(BASE_DIR, "model", "label_encoder.pkl")
DEFAULT_HAND_MODEL = os.path.join(BASE_DIR, "hand_landmarker.task")

# Fallback if label_encoder.pkl can't be unpickled (e.g. sklearn version
# mismatch) — matches kSignLabels in lib/data/sign_label_map.dart.
FALLBACK_LABELS = [str(d) for d in range(10)] + [chr(c) for c in range(ord("A"), ord("Z") + 1)]

CONFIDENCE_THRESHOLD = 0.85  # matches kRecognitionConfidenceThreshold


def load_labels(path: str) -> list[str]:
    try:
        import pickle
        with open(path, "rb") as f:
            le = pickle.load(f)
        labels = [str(c) for c in le.classes_]
        if labels != FALLBACK_LABELS:
            print("[WARN] label_encoder.pkl class order differs from "
                  "kSignLabels in the Flutter app — check for a mismatch!")
            print(f"  pkl:      {labels}")
            print(f"  app-side: {FALLBACK_LABELS}")
        return labels
    except Exception as e:
        print(f"[WARN] Could not load label_encoder.pkl ({e}); "
              f"falling back to hardcoded kSignLabels order.")
        return FALLBACK_LABELS


def normalize_landmarks(landmarks_list) -> np.ndarray | None:
    """Same normalization as preprocessing.py / the app's _normalise()."""
    landmarks = np.array([[lm.x, lm.y, lm.z] for lm in landmarks_list], dtype=np.float32)
    landmarks -= landmarks[0].copy()
    scale = np.linalg.norm(landmarks[9])
    if scale < 1e-6:
        return None
    landmarks /= scale
    return landmarks.flatten()  # (63,)


class TfliteModel:
    def __init__(self, model_path: str):
        import tensorflow as tf
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        print(f"[DIAG] Input shape:  {self.input_details[0]['shape']}")
        print(f"[DIAG] Output shape: {self.output_details[0]['shape']}")

    def predict(self, features: np.ndarray) -> np.ndarray:
        x = features.reshape(1, -1).astype(np.float32)
        self.interpreter.set_tensor(self.input_details[0]["index"], x)
        self.interpreter.invoke()
        return self.interpreter.get_tensor(self.output_details[0]["index"])[0]


class SessionLogger:
    def __init__(self, session_tag: str, outdir: str):
        self.session_tag = session_tag
        self.outdir = outdir
        self.start_ts = int(time.time() * 1000)
        self.rows: list[list] = []

    def log(self, target: str, top_label: str, top_conf: float, second_label: str,
             second_conf: float, hand_detected: bool, is_confident: bool, latency_ms: int):
        correct = hand_detected and top_label == target
        self.rows.append([
            int(time.time() * 1000), target, top_label or "-", f"{top_conf:.4f}",
            second_label or "-", f"{second_conf:.4f}", hand_detected, is_confident,
            correct, latency_ms,
        ])

    def export(self) -> str:
        safe = "".join(c if c.isalnum() or c in "_-" else "_" for c in self.session_tag)
        path = os.path.join(self.outdir, f"hiasl_recotest_{safe}_{self.start_ts}.csv")
        os.makedirs(self.outdir, exist_ok=True)
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["timestamp_ms", "target_letter", "top_label", "top_confidence",
                        "second_label", "second_confidence", "hand_detected",
                        "is_confident", "correct", "latency_ms"])
            w.writerows(self.rows)
        print(f"[SessionLogger] wrote {len(self.rows)} rows to {path}")
        return path


def draw_hud(frame, target, top_label, top_conf, second_label, second_conf,
             hand_detected, latency_ms, logging_on, entry_count, per_target_stats):
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 120), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    cv2.putText(frame, f"Target: {target}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
    cv2.putText(frame, f"top: {top_label or '-'} ({top_conf*100:.1f}%)  "
                        f"2nd: {second_label or '-'} ({second_conf*100:.1f}%)",
                (10, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1)
    cv2.putText(frame, f"hand: {hand_detected}   latency: {latency_ms}ms",
                (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

    correct, total = per_target_stats
    acc_str = f"{100*correct/total:.0f}% ({correct}/{total})" if total else "-"
    status = "LOGGING" if logging_on else "PAUSED"
    status_color = (0, 0, 255) if logging_on else (0, 200, 255)
    cv2.putText(frame, f"[{status}] frames logged: {entry_count}   target acc: {acc_str}",
                (10, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.55, status_color, 1)

    cv2.putText(frame, "0-9/A-Z: set target | SPACE: log on/off | Shift+E: export | ESC: quit",
                (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)
    return frame


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--session-tag", default="laptop_session",
                         help="e.g. yourname_daylight, yourname_dimlight")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--label-encoder", default=DEFAULT_LABEL_ENCODER)
    parser.add_argument("--hand-model", default=DEFAULT_HAND_MODEL)
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--outdir", default=os.path.join(BASE_DIR, "..", "..", "test_data"))
    args = parser.parse_args()

    if not os.path.exists(args.hand_model):
        print("Downloading MediaPipe hand landmarker model...")
        import urllib.request
        url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
        urllib.request.urlretrieve(url, args.hand_model)

    labels = load_labels(args.label_encoder)
    model = TfliteModel(args.model)

    options = HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=args.hand_model),
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        running_mode=vision.RunningMode.IMAGE,
    )

    cap = cv2.VideoCapture(args.camera_index)
    if not cap.isOpened():
        print(f"[ERROR] Could not open camera at index {args.camera_index}.")
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    logger = SessionLogger(args.session_tag, os.path.abspath(args.outdir))
    logging_on = False
    exported_at_count = 0
    target = labels[0]
    last_result = ("", 0.0, "", 0.0, False, 0)  # top,topc,second,secondc,hand,latency

    key_to_label = {c.lower(): c for c in labels if len(c) == 1}

    with HandLandmarker.create_from_options(options) as detector:
        print(f"\nSession tag: {args.session_tag}")
        print("Press a key (0-9 / A-Z) to set target, SPACE to start logging.\n")
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[ERROR] Failed to read camera frame.")
                break
            frame = cv2.flip(frame, 1)  # mirror, matches front-camera app usage
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = Image(image_format=ImageFormat.SRGB, data=rgb)

            t0 = time.perf_counter()
            result = detector.detect(mp_image)
            hand_detected = bool(result.hand_landmarks)

            top_label, top_conf, second_label, second_conf = "", 0.0, "", 0.0
            if hand_detected:
                features = normalize_landmarks(result.hand_landmarks[0])
                if features is not None:
                    probs = model.predict(features)
                    order = np.argsort(probs)[::-1]
                    top_idx, second_idx = order[0], order[1]
                    top_label, top_conf = labels[top_idx], float(probs[top_idx])
                    second_label, second_conf = labels[second_idx], float(probs[second_idx])
                else:
                    hand_detected = False  # degenerate landmarks — treat as no-detect

            latency_ms = int((time.perf_counter() - t0) * 1000)
            is_confident = top_conf >= CONFIDENCE_THRESHOLD
            last_result = (top_label, top_conf, second_label, second_conf, hand_detected, latency_ms)

            if logging_on:
                logger.log(target, top_label, top_conf, second_label, second_conf,
                           hand_detected, is_confident, latency_ms)

            per_target = [r for r in logger.rows if r[1] == target]
            correct_count = sum(1 for r in per_target if r[8] is True)
            stats = (correct_count, len(per_target))

            frame = draw_hud(frame, target, *last_result, logging_on, len(logger.rows), stats)
            cv2.imshow("HiASL Live Recognition Test", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                break
            elif key == ord(" "):
                logging_on = not logging_on
                print(f"Logging {'STARTED' if logging_on else 'PAUSED'} "
                      f"({len(logger.rows)} rows buffered)")
            elif key == ord("E"):  # Shift+E
                logger.export()
                exported_at_count = len(logger.rows)
            elif chr(key) in key_to_label:
                target = key_to_label[chr(key)]

    cap.release()
    cv2.destroyAllWindows()

    if len(logger.rows) > exported_at_count:
        print("Unsaved data — exporting before exit.")
        logger.export()


if __name__ == "__main__":
    main()
