"""
register_face.py
=================
Standalone registration script — run this ONCE before using run_inference.py.

Flow:
  1. Opens your webcam.
  2. Shows a live preview. Press SPACE to capture a snapshot.
  3. The largest detected face in each snapshot is embedded.
  4. After 5 captures, embeddings are averaged, L2-normalised,
     and saved to face_auth/registered_face.npy.

Usage:
    python register_face.py

Controls:
    SPACE  — capture current frame
    Q      — abort registration
"""

import sys
import cv2
import numpy as np

# Make sure we can import from the face_auth package
import os
sys.path.insert(0, os.path.dirname(__file__))

from face_auth.face_engine import FaceEngine
from face_auth import config


# ── Helpers ────────────────────────────────────────────────────────────────────

def _largest_face(faces):
    """Return (bbox, embedding) for the face with the largest bounding-box area."""
    best, best_area = None, -1
    for bbox, emb in faces:
        area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        if area > best_area:
            best_area = area
            best = (bbox, emb)
    return best


def _draw_hud(display, count, total):
    """Overlay capture count and instructions on display frame."""
    h, w = display.shape[:2]

    # Semi-transparent status bar at top
    overlay = display.copy()
    cv2.rectangle(overlay, (0, 0), (w, 75), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.6, display, 0.4, 0, display)

    # Progress text
    cv2.putText(display, f"Captured: {count}/{total}", (10, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 100), 2, cv2.LINE_AA)
    cv2.putText(display, "Press SPACE to capture  |  Q to abort", (10, 62),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)

    # Progress bar
    bar_x, bar_y, bar_w, bar_h = 10, 75, w - 20, 6
    cv2.rectangle(display, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h),
                  (60, 60, 60), -1)
    filled = int(bar_w * count / max(total, 1))
    if filled > 0:
        cv2.rectangle(display, (bar_x, bar_y),
                      (bar_x + filled, bar_y + bar_h), (0, 255, 100), -1)

    return display


# ── Main registration loop ─────────────────────────────────────────────────────

def register():
    print("\n" + "=" * 60)
    print("  FACE REGISTRATION")
    print(f"  Capture {config.REGISTRATION_IMAGES} images of your face.")
    print("=" * 60)
    print("  • Look straight at the camera.")
    print("  • Good lighting, neutral background preferred.")
    print("  • Press SPACE to capture, Q to abort.\n")

    engine = FaceEngine()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Cannot open webcam. Check your camera connection.")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    cv2.namedWindow("Face Registration", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Face Registration", 960, 540)

    embeddings = []
    total      = config.REGISTRATION_IMAGES
    flash_msg  = ""
    flash_until = 0.0

    while len(embeddings) < total:
        ret, frame = cap.read()
        if not ret:
            continue

        # Mirror for natural selfie feel
        display = cv2.flip(frame, 1)
        _draw_hud(display, len(embeddings), total)

        # Timed flash message (success / error feedback)
        import time
        if time.time() < flash_until:
            color = (0, 255, 100) if flash_msg.startswith("✓") else (0, 80, 255)
            cv2.putText(display, flash_msg, (10, 130),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)

        cv2.imshow("Face Registration", display)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            print("\n[INFO] Registration aborted.")
            break

        if key == ord(' '):
            # Use the un-flipped frame for InsightFace (natural orientation)
            faces = engine.detect_and_embed(frame)
            if not faces:
                flash_msg   = "✗ No face detected — please try again"
                flash_until = time.time() + 2.0
                print("  ✗ No face detected — try again.")
                continue

            _, emb = _largest_face(faces)
            embeddings.append(emb)
            flash_msg   = f"✓ Captured {len(embeddings)}/{total}"
            flash_until = time.time() + 1.5
            print(f"  ✓ Captured {len(embeddings)}/{total}")

    cap.release()
    cv2.destroyAllWindows()

    if not embeddings:
        print("\n[ERROR] No embeddings captured. Registration failed.")
        sys.exit(1)

    # Average all captures and re-normalise to unit length
    avg_emb = np.mean(embeddings, axis=0)
    avg_emb = avg_emb / np.linalg.norm(avg_emb)

    np.save(config.EMBEDDING_FILE, avg_emb)
    print(f"\n[SUCCESS] Registered face saved → {config.EMBEDDING_FILE}")
    print(f"          Embedding shape : {avg_emb.shape}")
    print(f"          Captures used   : {len(embeddings)}/{total}")
    print("\nYou can now run:  python run_inference.py")


if __name__ == "__main__":
    register()
