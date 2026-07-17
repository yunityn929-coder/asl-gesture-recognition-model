"""
ASL Gesture Recognition - Dynamic Data Collection
Collects sequences of frames for dynamic signs (J and Z).
Run this script on WINDOWS (not WSL) for camera access.

Each sample = 30 consecutive frames of hand landmarks = one J or Z motion.

Controls:
  SPACE - start recording a sequence
  Q     - quit current class, move to next
  ESC   - exit program
"""

import cv2
import mediapipe as mp
import numpy as np
import os
import time

# ── Configuration ─────────────────────────────────────────────────────────────
DATASET_PATH = os.path.join(os.path.dirname(__file__), "dynamic", "dataset")
SAMPLES_PER_CLASS = 100   # sequences to collect per sign
SEQUENCE_LENGTH = 30      # frames per sequence (~1 second at 30fps)
CAMERA_INDEX = 0          # change to 1 if webcam is not default

DYNAMIC_CLASSES = ["J", "Z"]

# ── MediaPipe setup ───────────────────────────────────────────────────────────
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles


# ── Helpers ───────────────────────────────────────────────────────────────────
def extract_landmarks(hand_landmarks):
    """
    Extract and normalize 21 hand landmarks (x, y, z) relative to wrist.
    Returns a (63,) numpy array, scale-invariant and position-invariant.
    Same normalization as static — consistency is important.
    """
    landmarks = []
    for lm in hand_landmarks.landmark:
        landmarks.append([lm.x, lm.y, lm.z])
    landmarks = np.array(landmarks)  # shape: (21, 3)

    # Normalize: subtract wrist (landmark 0)
    wrist = landmarks[0]
    landmarks -= wrist

    # Scale normalize: divide by wrist-to-middle-MCP distance
    scale = np.linalg.norm(landmarks[9])
    if scale > 0:
        landmarks /= scale

    return landmarks.flatten()  # shape: (63,)


def draw_ui(frame, class_name, sample_count, total_samples,
            status, frames_recorded=0, sequence_length=SEQUENCE_LENGTH):
    """Draw overlay UI on the camera frame."""
    h, w = frame.shape[:2]

    # Top bar
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 100), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

    # Class name
    cv2.putText(frame, f"Sign: {class_name}", (10, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

    # Sample count
    cv2.putText(frame, f"Samples: {sample_count}/{total_samples}", (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 100), 2)

    # Status
    color_map = {
        "READY": (0, 200, 255),
        "RECORDING": (0, 0, 255),
        "SAVED": (0, 255, 0),
        "NO HAND": (100, 100, 255),
        "GET READY": (255, 200, 0),
    }
    color = color_map.get(status, (255, 255, 255))
    cv2.putText(frame, status, (w - 220, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    # Recording progress bar
    if status == "RECORDING":
        bar_w = int((frames_recorded / sequence_length) * (w - 20))
        cv2.rectangle(frame, (10, h - 40), (w - 10, h - 20), (50, 50, 50), -1)
        cv2.rectangle(frame, (10, h - 40), (10 + bar_w, h - 20), (0, 0, 255), -1)
        cv2.putText(frame, f"Recording: {frames_recorded}/{sequence_length} frames",
                    (10, h - 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    else:
        cv2.putText(frame, "SPACE: start recording | Q: next class | ESC: exit",
                    (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

    return frame


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    # Create all class directories
    for cls in DYNAMIC_CLASSES:
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

        for class_idx, class_name in enumerate(DYNAMIC_CLASSES):
            class_dir = os.path.join(DATASET_PATH, class_name)

            # Resume if interrupted
            existing = len([f for f in os.listdir(class_dir) if f.endswith(".npy")])
            sample_count = existing
            print(f"\n[{class_idx+1}/{len(DYNAMIC_CLASSES)}] Collecting: {class_name} "
                  f"(already have {existing}/{SAMPLES_PER_CLASS})")

            if existing >= SAMPLES_PER_CLASS:
                print(f"  Skipping {class_name} — already complete.")
                continue

            status = "READY"
            recording = False
            sequence_buffer = []  # holds frames during recording
            last_save_time = 0

            # Countdown before each class
            countdown_start = time.time()
            countdown_done = False

            while True:
                ret, frame = cap.read()
                if not ret:
                    print("[ERROR] Failed to read camera frame.")
                    break

                frame = cv2.flip(frame, 1)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(rgb)

                # Draw landmarks
                if results.multi_hand_landmarks:
                    for hand_lms in results.multi_hand_landmarks:
                        mp_drawing.draw_landmarks(
                            frame, hand_lms,
                            mp_hands.HAND_CONNECTIONS,
                            mp_drawing_styles.get_default_hand_landmarks_style(),
                            mp_drawing_styles.get_default_hand_connections_style()
                        )

                # Countdown
                if not countdown_done:
                    elapsed = time.time() - countdown_start
                    remaining = int(3 - elapsed)
                    if elapsed >= 3:
                        countdown_done = True
                    else:
                        draw_ui(frame, class_name, sample_count,
                                SAMPLES_PER_CLASS, "GET READY")
                        cv2.putText(frame, str(remaining + 1),
                                    (frame.shape[1] // 2 - 30, frame.shape[0] // 2),
                                    cv2.FONT_HERSHEY_SIMPLEX, 4.0, (0, 0, 255), 6)
                        cv2.imshow("ASL Dynamic Data Collection", frame)
                        cv2.waitKey(1)
                        continue

                # ── Recording logic ───────────────────────────────────────────
                if recording:
                    if results.multi_hand_landmarks:
                        landmarks = extract_landmarks(results.multi_hand_landmarks[0])
                        sequence_buffer.append(landmarks)
                    else:
                        # Pad with zeros if hand briefly disappears
                        sequence_buffer.append(np.zeros(63))

                    status = "RECORDING"

                    # Sequence complete
                    if len(sequence_buffer) >= SEQUENCE_LENGTH:
                        sequence = np.array(sequence_buffer)  # shape: (30, 63)
                        save_path = os.path.join(class_dir, f"{sample_count}.npy")
                        np.save(save_path, sequence)
                        sample_count += 1
                        last_save_time = time.time()
                        print(f"  Saved sequence {sample_count}/{SAMPLES_PER_CLASS} "
                              f"for '{class_name}' — shape: {sequence.shape}")

                        # Reset for next recording
                        recording = False
                        sequence_buffer = []
                        status = "SAVED"
                else:
                    # Flash SAVED for 0.5s
                    if time.time() - last_save_time < 0.5:
                        status = "SAVED"
                    else:
                        status = "READY" if results.multi_hand_landmarks else "NO HAND"

                draw_ui(frame, class_name, sample_count, SAMPLES_PER_CLASS,
                        status, len(sequence_buffer))
                cv2.imshow("ASL Dynamic Data Collection", frame)

                key = cv2.waitKey(1) & 0xFF

                if key == 27:  # ESC
                    print("\n[EXIT] Exiting early.")
                    cap.release()
                    cv2.destroyAllWindows()
                    return

                elif key == ord('q'):  # Q — next class
                    print(f"  Skipping to next class (collected {sample_count}).")
                    recording = False
                    sequence_buffer = []
                    break

                elif key == ord(' ') and not recording:  # SPACE — start recording
                    if results.multi_hand_landmarks:
                        print(f"  Recording sequence {sample_count + 1}... "
                              f"perform the '{class_name}' sign now!")
                        recording = True
                        sequence_buffer = []
                    else:
                        print("  No hand detected — show your hand first.")

                # Auto-advance when class complete
                if sample_count >= SAMPLES_PER_CLASS:
                    print(f"  ✓ {class_name} complete!")
                    time.sleep(0.5)
                    break

    cap.release()
    cv2.destroyAllWindows()
    print("\n[DONE] Dynamic data collection complete!")
    print(f"Dataset saved to: {os.path.abspath(DATASET_PATH)}")
    print(f"\nEach .npy file shape: ({SEQUENCE_LENGTH}, 63)")
    print("= 30 frames × 21 landmarks × 3 coords (normalized)")


if __name__ == "__main__":
    main()
