"""
ASL Gesture Recognition - Static Data Collection
Collects single-frame landmark data for static signs (A-I, K-Y, 0-9)
Run this script on WINDOWS (not WSL) for camera access.

Controls:
  SPACE - capture a sample
  Q     - quit current class, move to next
  ESC   - exit program
"""

import cv2
import mediapipe as mp
import numpy as np
import os
import time

# ── Configuration ────────────────────────────────────────────────────────────
DATASET_PATH = os.path.join(os.path.dirname(__file__), "static", "dataset")
SAMPLES_PER_CLASS = 100   # number of samples to collect per sign
CAMERA_INDEX = 0          # change to 1 if your webcam is not the default

# Static classes: A-Z excluding J and Z, plus digits 0-9
STATIC_CLASSES = [c for c in "ABCDEFGHIKLMNOPQRSTUVWXY"] + \
                 [str(i) for i in range(10)]
# Note: J and Z are dynamic signs handled in dynamic_data_collection.py

# ── MediaPipe setup ───────────────────────────────────────────────────────────
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# ── Helpers ───────────────────────────────────────────────────────────────────
def extract_landmarks(hand_landmarks):
    """
    Extract and normalize 21 hand landmarks (x, y, z) relative to wrist.
    Returns a (63,) numpy array, scale-invariant and position-invariant.
    """
    landmarks = []
    for lm in hand_landmarks.landmark:
        landmarks.append([lm.x, lm.y, lm.z])
    landmarks = np.array(landmarks)  # shape: (21, 3)

    # Normalize: subtract wrist position (landmark 0) so wrist is at origin
    wrist = landmarks[0]
    landmarks -= wrist

    # Scale normalize: divide by the distance between wrist and middle finger MCP (landmark 9)
    # This makes the features scale-invariant (hand size doesn't matter)
    scale = np.linalg.norm(landmarks[9])
    if scale > 0:
        landmarks /= scale

    return landmarks.flatten()  # shape: (63,)


def draw_ui(frame, class_name, sample_count, total_samples, status, countdown=None):
    """Draw overlay UI on the camera frame."""
    h, w = frame.shape[:2]

    # Semi-transparent background bar at top
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 90), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

    # Class name
    cv2.putText(frame, f"Sign: {class_name}", (10, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

    # Sample count
    cv2.putText(frame, f"Samples: {sample_count}/{total_samples}", (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 100), 2)

    # Status message
    color = (0, 255, 0) if status == "CAPTURED" else \
            (0, 200, 255) if status == "READY" else (255, 255, 0)
    cv2.putText(frame, status, (w - 220, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    # Controls hint
    cv2.putText(frame, "SPACE: capture | Q: next class | ESC: exit",
                (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

    # Countdown overlay
    if countdown is not None:
        cv2.putText(frame, str(countdown), (w // 2 - 30, h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 4.0, (0, 0, 255), 6)

    return frame


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    # Create all class directories upfront
    for cls in STATIC_CLASSES:
        os.makedirs(os.path.join(DATASET_PATH, cls), exist_ok=True)

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print(f"[ERROR] Could not open camera at index {CAMERA_INDEX}.")
        print("Try changing CAMERA_INDEX to 1 or 2 at the top of this script.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.6
    ) as hands:

        for class_idx, class_name in enumerate(STATIC_CLASSES):
            class_dir = os.path.join(DATASET_PATH, class_name)

            # Count existing samples (resume if interrupted)
            existing = len([f for f in os.listdir(class_dir) if f.endswith(".npy")])
            sample_count = existing
            print(f"\n[{class_idx+1}/{len(STATIC_CLASSES)}] Collecting: {class_name} "
                  f"(already have {existing}/{SAMPLES_PER_CLASS})")

            if existing >= SAMPLES_PER_CLASS:
                print(f"  Skipping {class_name} — already complete.")
                continue

            status = "READY"
            last_capture_time = 0

            # Countdown before starting each class
            countdown_start = time.time()
            countdown_done = False

            while True:
                ret, frame = cap.read()
                if not ret:
                    print("[ERROR] Failed to read camera frame.")
                    break

                frame = cv2.flip(frame, 1)  # mirror for natural interaction
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(rgb)

                # Draw hand landmarks
                if results.multi_hand_landmarks:
                    for hand_lms in results.multi_hand_landmarks:
                        mp_drawing.draw_landmarks(
                            frame, hand_lms,
                            mp_hands.HAND_CONNECTIONS,
                            mp_drawing_styles.get_default_hand_landmarks_style(),
                            mp_drawing_styles.get_default_hand_connections_style()
                        )

                # 3-second countdown at start of each class
                if not countdown_done:
                    elapsed = time.time() - countdown_start
                    remaining = int(3 - elapsed)
                    if elapsed >= 3:
                        countdown_done = True
                    else:
                        draw_ui(frame, class_name, sample_count,
                                SAMPLES_PER_CLASS, "GET READY", countdown=remaining + 1)
                        cv2.imshow("ASL Static Data Collection", frame)
                        cv2.waitKey(1)
                        continue

                # Flash "CAPTURED" for 0.4s after each capture
                if time.time() - last_capture_time < 0.4:
                    status = "CAPTURED"
                else:
                    status = "READY" if results.multi_hand_landmarks else "NO HAND"

                draw_ui(frame, class_name, sample_count, SAMPLES_PER_CLASS, status)
                cv2.imshow("ASL Static Data Collection", frame)

                key = cv2.waitKey(1) & 0xFF

                if key == 27:  # ESC — exit everything
                    print("\n[EXIT] Exiting early.")
                    cap.release()
                    cv2.destroyAllWindows()
                    return

                elif key == ord('q'):  # Q — skip to next class
                    print(f"  Skipping to next class (collected {sample_count}).")
                    break

                elif key == ord(' '):  # SPACE — capture sample
                    if results.multi_hand_landmarks:
                        landmarks = extract_landmarks(results.multi_hand_landmarks[0])
                        save_path = os.path.join(class_dir, f"{sample_count}.npy")
                        np.save(save_path, landmarks)
                        sample_count += 1
                        last_capture_time = time.time()
                        print(f"  Saved sample {sample_count}/{SAMPLES_PER_CLASS} "
                              f"for '{class_name}'")
                    else:
                        print("  No hand detected — move your hand into frame.")

                # Auto-advance when class is complete
                if sample_count >= SAMPLES_PER_CLASS:
                    print(f"  ✓ {class_name} complete!")
                    time.sleep(0.5)
                    break

    cap.release()
    cv2.destroyAllWindows()
    print("\n[DONE] Static data collection complete!")
    print(f"Dataset saved to: {os.path.abspath(DATASET_PATH)}")


if __name__ == "__main__":
    main()
