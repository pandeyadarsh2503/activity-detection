"""
FIR Action Detection – Live Camera + Dataset Mode
==================================================
USAGE:
  python run_inference.py               --> Opens LIVE CAMERA (default)
  python run_inference.py --dataset     --> Runs on the cloned dataset
  python run_inference.py --video video105  --> Runs on one video from dataset

HOW LIVE MODE WORKS:
  Uses MediaPipe Pose to detect your body skeleton in real time.
  Classifies your action based on the ANGLE of your body:
    - Nearly horizontal  --> sleeping / lying
    - Diagonal           --> falling
    - Upright + still    --> standing
    - Upright + moving   --> walking
    - Hips low           --> sitting

CONTROLS (live mode):
  Q        --> Quit
  S        --> Save screenshot
  R        --> Reset / clear history
"""

import cv2
import numpy as np
import argparse
import time
import json
import glob
import os
from pathlib import Path
from collections import deque

# ─────────────────────────────────────────────────────────────────────────────
#  DATASET MODE helpers (unchanged from before)
# ─────────────────────────────────────────────────────────────────────────────

DATASET_DIR = Path(__file__).parent / "dataset"
OUTPUT_DIR  = Path(__file__).parent / "output_results"

ACTION_LABELS = {
    0:"sit", 1:"stand", 2:"walk", 3:"fall", 4:"lie",
    5:"standup", 6:"sitdown", 7:"lying", 8:"sleeping",
    9:"falling", 10:"falled", 11:"active",
}
ACTION_COLORS = {
    0:(255,200,50), 1:(50,220,50), 2:(50,200,255), 3:(50,50,255),
    4:(255,100,230), 5:(100,255,180), 6:(255,160,60), 7:(200,80,255),
    8:(30,100,255), 9:(0,180,255), 10:(50,50,180), 11:(200,230,0),
}
IMG_W, IMG_H = 320, 240

def yolo_to_abs(cx, cy, w, h):
    x1 = int((cx - w/2)*IMG_W); y1 = int((cy - h/2)*IMG_H)
    x2 = int((cx + w/2)*IMG_W); y2 = int((cy + h/2)*IMG_H)
    return max(0,x1), max(0,y1), min(IMG_W,x2), min(IMG_H,y2)

def parse_bbox(path):
    anns = []
    if not Path(path).exists() or Path(path).stat().st_size == 0:
        return anns
    with open(path) as f:
        for line in f:
            p = line.strip().split()
            if len(p) >= 5:
                anns.append((int(p[0]), *map(float, p[1:5])))
    return anns

def get_fnum(path):
    stem = Path(path).stem
    idx  = stem.rfind("_frame_")
    try:   return int(stem[idx+7:]) if idx != -1 else -1
    except: return -1

def draw_box(frame, anns):
    out = frame.copy()
    for cls, cx, cy, w, h in anns:
        x1,y1,x2,y2 = yolo_to_abs(cx, cy, w, h)
        color = ACTION_COLORS.get(cls, (180,180,180))
        label = ACTION_LABELS.get(cls, str(cls))
        cv2.rectangle(out, (x1,y1), (x2,y2), color, 2)
        (tw,th),_ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(out, (x1, y1-th-8), (x1+tw+6, y1), color, -1)
        cv2.putText(out, label, (x1+3,y1-4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (20,20,20), 1, cv2.LINE_AA)
    return out

def run_dataset(video_name=None, max_frames=20):
    """Show annotated frames from dataset in a window (slideshow)."""
    dirs = [DATASET_DIR / video_name] if video_name else sorted(DATASET_DIR.iterdir())
    dirs = [d for d in dirs if d.is_dir()]
    print(f"\n[Dataset mode] {len(dirs)} video(s). Press SPACE=next  Q=quit\n")
    for vdir in dirs:
        bbox_dir = vdir / "BBOX"
        rgb_dir  = vdir / "RGB"
        if not bbox_dir.is_dir(): continue
        rgb_map  = {get_fnum(p): p for p in rgb_dir.glob("*.jpg")}
        bbox_files = sorted(bbox_dir.glob("*.txt"), key=lambda p: get_fnum(p.name))
        shown = 0
        for bf in bbox_files:
            fnum = get_fnum(bf.name)
            if fnum not in rgb_map: continue
            frame = cv2.imread(str(rgb_map[fnum]))
            if frame is None: continue
            anns = parse_bbox(bf)
            if not anns: continue
            annotated = draw_box(frame, anns)
            # Scale up for visibility
            display = cv2.resize(annotated, (640, 480))
            labels = [ACTION_LABELS.get(a[0], str(a[0])) for a in anns]
            info = f"[{vdir.name}]  frame {fnum}  |  Action: {', '.join(labels)}"
            cv2.setWindowTitle("FIR Dataset Viewer", info)
            cv2.imshow("FIR Dataset Viewer", display)
            key = cv2.waitKey(100) & 0xFF   # auto-advance every 100ms
            if key == ord('q'): cv2.destroyAllWindows(); return
            shown += 1
            if max_frames and shown >= max_frames: break
    cv2.destroyAllWindows()

# ─────────────────────────────────────────────────────────────────────────────
#  LIVE CAMERA helpers
# ─────────────────────────────────────────────────────────────────────────────

# Action info: (label, description, BGR color, emoji-like prefix)
LIVE_ACTIONS = {
    "sleeping":  ("SLEEPING",  "Person is sleeping on the floor",  (30, 80, 220),  "[ZZZ]"),
    "lying":     ("LYING DOWN","Person is lying on the floor",      (200,60, 255),  "[LIE]"),
    "falling":   ("FALLING!",  "Person appears to be falling",      (0,  60, 255),  "[!!!]"),
    "sitting":   ("SITTING",   "Person is sitting",                  (255,140, 0),   "[ S ]"),
    "standing":  ("STANDING",  "Person is standing still",           (50, 200, 50),  "[ | ]"),
    "walking":   ("WALKING",   "Person is walking / moving",         (255,180, 20),  "[->]"),
    "unknown":   ("DETECTING...","Waiting for clear pose detection", (120,120,120),  "[ ? ]"),
}

def angle_with_vertical(p1, p2):
    """Angle between vector p1->p2 and the vertical axis (degrees). 0=up, 90=horizontal."""
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]  # positive = downward in image
    angle = abs(np.degrees(np.arctan2(abs(dx), abs(dy))))
    return angle                # 0° = perfectly vertical, 90° = horizontal

def midpoint(a, b):
    return ((a[0]+b[0])/2, (a[1]+b[1])/2)

def classify_action(landmarks, h, w, motion_score, horizontal_duration):
    """
    Returns action key from LIVE_ACTIONS.
    Uses MediaPipe landmark indices:
      0=nose  11=L-shoulder 12=R-shoulder
      23=L-hip  24=R-hip  25=L-knee  26=R-knee
    """
    try:
        lm = landmarks.landmark
        # Get key coordinates (normalized 0-1)
        nose        = (lm[0].x,  lm[0].y)
        l_shoulder  = (lm[11].x, lm[11].y)
        r_shoulder  = (lm[12].x, lm[12].y)
        l_hip       = (lm[23].x, lm[23].y)
        r_hip       = (lm[24].x, lm[24].y)
        l_knee      = (lm[25].x, lm[25].y)
        r_knee      = (lm[26].x, lm[26].y)

        shoulder_mid = midpoint(l_shoulder, r_shoulder)
        hip_mid      = midpoint(l_hip, r_hip)
        knee_mid     = midpoint(l_knee, r_knee)

        # Visibility check
        vis = [lm[i].visibility for i in [11,12,23,24]]
        if min(vis) < 0.3:
            return "unknown"

        # Body tilt: angle from vertical (mid-shoulder → mid-hip vector)
        body_angle = angle_with_vertical(shoulder_mid, hip_mid)

        # How high are the hips in the frame (0=top, 1=bottom)
        hip_y = hip_mid[1]

        # ── Classification rules ──────────────────────────────────────────

        # 1. Sleeping / lying: body nearly horizontal (>55°)
        if body_angle > 55:
            if horizontal_duration > 3.0:
                return "sleeping"
            return "lying"

        # 2. Falling: body tilted 35–55° AND not sustained
        if body_angle > 35:
            return "falling"

        # 3. Sitting: body upright but hips are below middle of frame
        #    AND knees are roughly at hip level
        knee_hip_diff = abs(knee_mid[1] - hip_mid[1])
        if hip_y > 0.55 and knee_hip_diff < 0.15:
            return "sitting"

        # 4. Walking: upright + significant motion
        if motion_score > 1800:
            return "walking"

        # 5. Standing still
        return "standing"

    except Exception:
        return "unknown"


def draw_skeleton(frame, results, mp_drawing, mp_pose):
    """Draw pose landmarks on frame."""
    mp_drawing.draw_landmarks(
        frame,
        results.pose_landmarks,
        mp_pose.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0,220,255), thickness=2, circle_radius=3),
        connection_drawing_spec=mp_drawing.DrawingSpec(color=(255,255,255), thickness=2),
    )


def draw_hud(frame, action_key, body_angle, fps, motion_score, h_duration, screenshot_flash):
    """Draw the heads-up display overlay on the frame."""
    fh, fw = frame.shape[:2]
    info    = LIVE_ACTIONS[action_key]
    label, description, color, prefix = info

    # ── Top banner ──────────────────────────────────────────────────────────
    banner_h = 80
    overlay  = frame.copy()
    cv2.rectangle(overlay, (0,0), (fw, banner_h), (15,15,15), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    # Action label (big, colored)
    text = f"{prefix}  {label}"
    cv2.putText(frame, text, (20, 52),
                cv2.FONT_HERSHEY_DUPLEX, 1.5, color, 2, cv2.LINE_AA)

    # ── Bottom info bar ──────────────────────────────────────────────────────
    bar_y = fh - 85
    overlay2 = frame.copy()
    cv2.rectangle(overlay2, (0, bar_y), (fw, fh), (15,15,15), -1)
    cv2.addWeighted(overlay2, 0.75, frame, 0.25, 0, frame)

    # Description
    cv2.putText(frame, description, (15, bar_y + 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (220,220,220), 1, cv2.LINE_AA)

    # Stats line
    stats = f"Body angle: {body_angle:.1f}deg   Motion: {int(motion_score)}   FPS: {fps:.1f}"
    if action_key in ("sleeping","lying"):
        stats += f"   On floor: {h_duration:.1f}s"
    cv2.putText(frame, stats, (15, bar_y + 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (160,160,160), 1, cv2.LINE_AA)

    # ── Color accent line at left edge ──────────────────────────────────────
    cv2.rectangle(frame, (0, banner_h), (6, bar_y), color, -1)

    # ── Controls hint ──────────────────────────────────────────────────────
    ctrl = "Q=Quit  S=Screenshot  R=Reset"
    cv2.putText(frame, ctrl, (fw-250, fh-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100,100,100), 1)

    # ── Screenshot flash ────────────────────────────────────────────────────
    if screenshot_flash > 0:
        flash = frame.copy()
        cv2.rectangle(flash, (0,0), (fw,fh), (255,255,255), -1)
        cv2.addWeighted(flash, 0.3 * min(1.0, screenshot_flash), frame, 1 - 0.3*min(1.0,screenshot_flash), 0, frame)

    return frame


def run_live_camera(cam_index=0):
    """Main live camera loop."""
    try:
        import mediapipe as mp
    except ImportError:
        print("[ERROR] mediapipe not installed. Run: pip install mediapipe")
        return

    mp_pose    = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        print(f"[ERROR] Could not open camera index {cam_index}.")
        print("  Try: python run_inference.py --cam 1")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print("\n" + "="*60)
    print("  FIR Live Action Detection – Camera Active")
    print("="*60)
    print("  Controls: Q=Quit  S=Screenshot  R=Reset")
    print("  The system will detect: sleeping, lying, falling,")
    print("  sitting, standing, walking")
    print("="*60 + "\n")

    cv2.namedWindow("FIR Live Action Detection", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("FIR Live Action Detection", 1280, 720)

    # History / smoothing
    action_history     = deque(maxlen=15)   # last 15 frames for smoothing
    prev_gray          = None
    fps_counter        = deque(maxlen=30)
    horizontal_start   = None               # when body went horizontal
    horizontal_duration = 0.0
    screenshot_flash   = 0.0
    screenshot_dir     = OUTPUT_DIR / "screenshots"
    screenshot_dir.mkdir(parents=True, exist_ok=True)
    shot_count = 0

    with mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        model_complexity=1,
    ) as pose:

        while True:
            t0 = time.time()
            ret, frame = cap.read()
            if not ret:
                print("[WARN] Frame capture failed. Retrying...")
                time.sleep(0.1)
                continue

            # Flip so it's mirror-like (more natural)
            frame = cv2.flip(frame, 1)
            fh, fw = frame.shape[:2]

            # ── Motion detection (frame difference) ──────────────────────
            gray       = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray_blur  = cv2.GaussianBlur(gray, (21,21), 0)
            if prev_gray is None:
                prev_gray = gray_blur
            diff         = cv2.absdiff(prev_gray, gray_blur)
            motion_score = float(np.sum(diff > 25))
            prev_gray    = gray_blur

            # ── Pose estimation ──────────────────────────────────────────
            rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            results  = pose.process(rgb)
            rgb.flags.writeable = True

            body_angle = 0.0
            raw_action = "unknown"

            if results.pose_landmarks:
                draw_skeleton(frame, results, mp_drawing, mp_pose)

                # Body angle for HUD
                lm = results.pose_landmarks.landmark
                sh_mid  = midpoint((lm[11].x, lm[11].y), (lm[12].x, lm[12].y))
                hip_mid = midpoint((lm[23].x, lm[23].y), (lm[24].x, lm[24].y))
                body_angle = angle_with_vertical(sh_mid, hip_mid)

                # Track horizontal duration
                if body_angle > 55:
                    if horizontal_start is None:
                        horizontal_start = time.time()
                    horizontal_duration = time.time() - horizontal_start
                else:
                    horizontal_start    = None
                    horizontal_duration = 0.0

                raw_action = classify_action(
                    results.pose_landmarks, fh, fw,
                    motion_score, horizontal_duration
                )

            # ── Smooth action over last N frames ─────────────────────────
            action_history.append(raw_action)
            counts     = {a: action_history.count(a) for a in set(action_history)}
            best_action = max(counts, key=counts.get)

            # ── FPS ───────────────────────────────────────────────────────
            fps_counter.append(time.time() - t0)
            fps = 1.0 / (sum(fps_counter) / len(fps_counter)) if fps_counter else 0

            # ── Draw HUD ──────────────────────────────────────────────────
            if screenshot_flash > 0:
                screenshot_flash -= 0.15
            frame = draw_hud(
                frame, best_action, body_angle,
                fps, motion_score, horizontal_duration, screenshot_flash
            )

            cv2.imshow("FIR Live Action Detection", frame)

            # ── Key handling ──────────────────────────────────────────────
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                shot_path = screenshot_dir / f"shot_{int(time.time())}.jpg"
                cv2.imwrite(str(shot_path), frame)
                shot_count += 1
                screenshot_flash = 1.5
                print(f"  [Screenshot] Saved: {shot_path.name}")
            elif key == ord('r'):
                action_history.clear()
                horizontal_start    = None
                horizontal_duration = 0.0
                print("  [Reset] History cleared.")

    cap.release()
    cv2.destroyAllWindows()
    print(f"\n[Done] {shot_count} screenshot(s) saved to: {screenshot_dir}")


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="FIR Action Detection – Live Camera or Dataset Mode"
    )
    parser.add_argument("--dataset",    action="store_true", help="Run on the cloned dataset instead of live camera")
    parser.add_argument("--video",      default=None,        help="Dataset video folder name, e.g. 'video105'")
    parser.add_argument("--max-frames", default=20, type=int,help="Max frames per video in dataset mode (0=all)")
    parser.add_argument("--cam",        default=0,  type=int,help="Camera index (default 0). Try 1 if 0 doesn't work")
    args = parser.parse_args()

    if args.video or args.dataset:
        # ── Dataset slideshow mode ────────────────────────────────────────
        mf = args.max_frames if args.max_frames > 0 else None
        run_dataset(video_name=args.video, max_frames=mf)
    else:
        # ── Live camera mode (default) ────────────────────────────────────
        run_live_camera(cam_index=args.cam)


if __name__ == "__main__":
    main()
