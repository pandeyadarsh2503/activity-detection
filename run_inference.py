"""
run_inference.py  —  Face-Gated Unified Activity Detection
===========================================================
Pipeline:
  1. Face Authentication   (InsightFace buffalo_s)
       → ONLY the pre-registered user passes through.
  2. Posture Detection     (MediaPipe Pose)
       sleeping / lying / falling / sitting / standing / walking
  3. Intake Detection      (MediaPipe Pose landmarks)
       eating / drinking + Bites Per Minute

PRE-REQUISITE:
  python register_face.py    ← run once to save face_auth/registered_face.npy

USAGE:
  python run_inference.py               -->  Live camera (default)
  python run_inference.py --cam 1       -->  Use camera index 1
  python run_inference.py --dataset     -->  Dataset slideshow mode
  python run_inference.py --video video105   -->  One dataset video

LIVE CONTROLS:
  Q  –  Quit
  S  –  Save screenshot to output_results/screenshots/
  R  –  Reset session counters (bites, posture history)
"""

import cv2
import sys
import numpy as np
import argparse
import time
import threading
from pathlib import Path
from collections import deque

from detectors.posture_detector import PostureDetector, POSTURE_ACTIONS
from detectors.intake_detector  import IntakeDetector, IntakeResult
from detectors.posture_detector import PostureResult
from display.hud import draw_skeleton, draw_mouth_box, draw_hud, draw_auth_overlay

# ── Dataset helpers ────────────────────────────────────────────────────────────
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


def _yolo_to_abs(cx, cy, w, h):
    x1 = int((cx - w/2)*IMG_W); y1 = int((cy - h/2)*IMG_H)
    x2 = int((cx + w/2)*IMG_W); y2 = int((cy + h/2)*IMG_H)
    return max(0,x1), max(0,y1), min(IMG_W,x2), min(IMG_H,y2)

def _parse_bbox(path):
    anns = []
    p = Path(path)
    if not p.exists() or p.stat().st_size == 0:
        return anns
    with open(p) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                anns.append((int(parts[0]), *map(float, parts[1:5])))
    return anns

def _get_fnum(path):
    stem = Path(path).stem
    idx  = stem.rfind("_frame_")
    try:   return int(stem[idx+7:]) if idx != -1 else -1
    except: return -1

def _draw_dataset_boxes(frame, anns):
    out = frame.copy()
    for cls, cx, cy, w, h in anns:
        x1, y1, x2, y2 = _yolo_to_abs(cx, cy, w, h)
        color = ACTION_COLORS.get(cls, (180,180,180))
        label = ACTION_LABELS.get(cls, str(cls))
        cv2.rectangle(out, (x1,y1), (x2,y2), color, 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(out, (x1, y1-th-8), (x1+tw+6, y1), color, -1)
        cv2.putText(out, label, (x1+3, y1-4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (20,20,20), 1, cv2.LINE_AA)
    return out

def run_dataset(video_name=None, max_frames=20):
    dirs = [DATASET_DIR / video_name] if video_name else sorted(DATASET_DIR.iterdir())
    dirs = [d for d in dirs if d.is_dir()]
    print(f"\n[Dataset mode] {len(dirs)} video(s). SPACE=next  Q=quit\n")
    for vdir in dirs:
        bbox_dir = vdir / "BBOX"
        rgb_dir  = vdir / "RGB"
        if not bbox_dir.is_dir():
            continue
        rgb_map    = {_get_fnum(p): p for p in rgb_dir.glob("*.jpg")}
        bbox_files = sorted(bbox_dir.glob("*.txt"), key=lambda p: _get_fnum(p.name))
        shown = 0
        for bf in bbox_files:
            fnum = _get_fnum(bf.name)
            if fnum not in rgb_map:
                continue
            frame = cv2.imread(str(rgb_map[fnum]))
            if frame is None:
                continue
            anns = _parse_bbox(bf)
            if not anns:
                continue
            display = cv2.resize(_draw_dataset_boxes(frame, anns), (640, 480))
            labels  = [ACTION_LABELS.get(a[0], str(a[0])) for a in anns]
            cv2.setWindowTitle("FIR Dataset Viewer",
                               f"[{vdir.name}]  frame {fnum}  |  {', '.join(labels)}")
            cv2.imshow("FIR Dataset Viewer", display)
            key = cv2.waitKey(120) & 0xFF
            if key == ord('q'):
                cv2.destroyAllWindows()
                return
            shown += 1
            if max_frames and shown >= max_frames:
                break
    cv2.destroyAllWindows()


# ── Face Authentication Worker ─────────────────────────────────────────────────

class FaceAuthWorker:
    """
    Runs InsightFace face authentication in a dedicated background thread.

    The worker re-embeds every FACE_SKIP_FRAMES frames to keep CPU usage
    manageable.  Authentication state transitions follow this FSM:

        WAITING  --[10 consecutive matches]--> AUTHENTICATED
        AUTHENTICATED  --[no match > 2 s]--> WAITING (grace + reset)

    Thread-safety: all shared state is protected by a single lock.
    The main thread ONLY calls submit_frame() and get_results().
    """

    def __init__(self, registered_emb: np.ndarray):
        from face_auth.face_engine import FaceEngine
        from face_auth import config as fa_cfg

        self._registered_emb = registered_emb
        self._fa_cfg = fa_cfg

        # Shared state (protected by lock)
        self._lock            = threading.Lock()
        self._latest_frame    = None          # BGR frame to process
        self._is_authenticated = False
        self._auth_bbox        = None         # smoothed bbox [x1,y1,x2,y2]
        self._auth_sim         = 0.0
        self._smooth_bbox      = None         # EMA smoother state

        self._running   = True
        self._fa_engine = None                # created inside thread

        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    # ── Public API ─────────────────────────────────────────────────────────────

    def submit_frame(self, bgr_frame: np.ndarray):
        """Hand a new BGR frame to the auth worker (non-blocking)."""
        with self._lock:
            self._latest_frame = bgr_frame

    def get_results(self):
        """
        Return (is_authenticated, auth_bbox, auth_sim).
        Always safe to call from the main thread.
        """
        with self._lock:
            return self._is_authenticated, self._auth_bbox, self._auth_sim

    def stop(self):
        self._running = False

    # ── Internal ───────────────────────────────────────────────────────────────

    def _run(self):
        from face_auth.face_engine import FaceEngine
        fa_cfg = self._fa_cfg

        engine = FaceEngine()

        consecutive_matches  = 0
        last_match_time      = None
        frame_counter        = 0

        while self._running:
            # Grab latest frame
            with self._lock:
                frame = self._latest_frame
                self._latest_frame = None

            if frame is None:
                time.sleep(0.005)
                continue

            frame_counter += 1

            # ── Re-embed only every FACE_SKIP_FRAMES ──────────────────────────
            if frame_counter % fa_cfg.FACE_SKIP_FRAMES != 0:
                continue

            faces = engine.detect_and_embed(frame)
            match = engine.best_match(faces, self._registered_emb)

            now = time.time()

            if match is not None:
                raw_bbox, sim = match
                last_match_time = now
                consecutive_matches += 1

                # EMA smooth the bounding box
                smooth = self._ema_bbox(raw_bbox)

                with self._lock:
                    self._auth_sim  = sim
                    self._auth_bbox = smooth
                    if consecutive_matches >= fa_cfg.AUTH_CONSECUTIVE_FRAMES:
                        self._is_authenticated = True
            else:
                consecutive_matches = 0
                # Grace period: keep authenticated for IDENTITY_GRACE_SECONDS
                # after the face disappears
                if last_match_time is not None:
                    elapsed = now - last_match_time
                    if elapsed > fa_cfg.IDENTITY_GRACE_SECONDS:
                        with self._lock:
                            self._is_authenticated = False
                            self._auth_bbox        = None
                            self._auth_sim         = 0.0
                        last_match_time = None
                        self._smooth_bbox = None
                else:
                    with self._lock:
                        self._is_authenticated = False

    def _ema_bbox(self, new_bbox: np.ndarray) -> np.ndarray:
        """Apply Exponential Moving Average to bbox to reduce jitter."""
        alpha = self._fa_cfg.EMA_ALPHA
        if self._smooth_bbox is None:
            self._smooth_bbox = new_bbox.astype(float)
        else:
            self._smooth_bbox = alpha * new_bbox.astype(float) + \
                                (1 - alpha) * self._smooth_bbox
        return self._smooth_bbox.astype(int)


# ── Detection Worker (runs in background thread) ───────────────────────────────

class DetectionWorker:
    """
    Runs MediaPipe + both detectors in a background thread.
    Main thread feeds frames in; reads results out — no blocking.
    """

    def __init__(self, mp_pose, mp_face):
        self._pose_model  = mp_pose
        self._face_model  = mp_face
        self._posture_det = PostureDetector(smoothing_frames=8)
        self._intake_det  = IntakeDetector()

        # Shared state (protected by lock)
        self._lock         = threading.Lock()
        self._latest_frame = None            # frame to process next
        self._posture      = None            # latest posture result
        self._intake       = None            # latest intake result
        self._pose_result  = None            # latest raw pose result (for skeleton)
        self._motion_score = 0.0

        self._running = True
        self._thread  = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def submit_frame(self, rgb_frame, motion_score: float):
        """Hand a new frame to the worker (non-blocking)."""
        with self._lock:
            self._latest_frame = rgb_frame
            self._motion_score = motion_score

    def get_results(self):
        """Return latest (posture, intake, pose_result). May be None at startup."""
        with self._lock:
            return self._posture, self._intake, self._pose_result

    def reset(self):
        with self._lock:
            self._posture_det.reset()
            self._intake_det.reset()

    def stop(self):
        self._running = False

    def _run(self):
        """Worker loop — runs in background thread."""
        with self._pose_model as pose_model, self._face_model as face_model:
            while self._running:
                # Grab latest frame (skip if none ready)
                with self._lock:
                    frame    = self._latest_frame
                    motion   = self._motion_score
                    self._latest_frame = None   # consume it

                if frame is None:
                    time.sleep(0.001)
                    continue

                # Run MediaPipe
                frame.flags.writeable = False
                pose_res = pose_model.process(frame)
                face_res = face_model.process(frame)
                frame.flags.writeable = True

                # Run detectors
                h, w = frame.shape[:2]
                fake_shape = (h, w, 3)
                posture = self._posture_det.update(pose_res, motion)
                intake  = self._intake_det.update(pose_res, face_res, fake_shape)

                with self._lock:
                    self._posture    = posture
                    self._intake     = intake
                    self._pose_result = pose_res


# ── Fallback "idle" results for startup ───────────────────────────────────────

def _idle_posture() -> PostureResult:
    info = POSTURE_ACTIONS["unknown"]
    return PostureResult(
        action="unknown", label=info[0], description=info[1],
        color=info[2], icon=info[3], body_angle=0.0, horizontal_duration=0.0,
    )

def _idle_intake() -> IntakeResult:
    return IntakeResult(
        is_intake=False, label="NOT EATING", description="",
        color=(80, 80, 80), bpm=0.0, bite_count=0,
        mouth_center=None, mouth_box=None, wrist_pt=None, confidence=0.0,
    )


# ── Live camera ────────────────────────────────────────────────────────────────

def run_live_camera(cam_index: int = 0):
    try:
        import mediapipe as mp
    except ImportError:
        print("[ERROR] mediapipe not installed. Run: pip install mediapipe")
        return

    # ── Load registered face embedding ────────────────────────────────────────
    from face_auth import config as fa_cfg
    emb_path = Path(fa_cfg.EMBEDDING_FILE)
    if not emb_path.exists():
        print("\n" + "="*64)
        print("  [ERROR] No registered face found.")
        print(f"  Expected: {emb_path}")
        print("\n  Please run:  python register_face.py")
        print("  This captures 5 photos of your face and saves the")
        print("  embedding so the system can recognise you.")
        print("="*64 + "\n")
        sys.exit(1)

    registered_emb = np.load(str(emb_path))
    print(f"[FaceAuth] Loaded registered embedding from {emb_path.name}")

    mp_pose    = mp.solutions.pose
    mp_face    = mp.solutions.face_mesh
    mp_drawing = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open camera {cam_index}. Try --cam 1")
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print("\n" + "="*62)
    print("  Face-Gated Unified Activity Detection  —  Camera Active")
    print("  Step 1: Look at the camera to authenticate.")
    print("  Step 2: Activity detection starts automatically.")
    print("="*62)
    print("  Q=Quit   S=Screenshot   R=Reset session\n")

    cv2.namedWindow("Unified Activity Detection", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Unified Activity Detection", 1280, 720)

    OUTPUT_DIR.mkdir(exist_ok=True)
    shots_dir = OUTPUT_DIR / "screenshots"
    shots_dir.mkdir(exist_ok=True)

    # ── Start workers ──────────────────────────────────────────────────────────
    face_worker = FaceAuthWorker(registered_emb)

    activity_worker = DetectionWorker(
        mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            model_complexity=1,
        ),
        mp_face.FaceMesh(
            max_num_faces=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            refine_landmarks=True,
        ),
    )

    # ── Application State Machine ──────────────────────────────────────────────
    class AppPhase:
        WAITING = 0
        SUCCESS = 1
        COUNTDOWN = 2
        ACTIVE = 3

    current_phase = AppPhase.WAITING
    phase_start_time = 0.0

    prev_gray        = None
    fps_buf          = deque(maxlen=30)
    screenshot_flash = 0.0
    shot_count       = 0
    frame_count      = 0

    try:
        while True:
            t0 = time.time()
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.01)
                continue

            frame = cv2.flip(frame, 1)
            frame_count += 1

            # ── Submit to face auth worker (always) ───────────────────────────
            face_worker.submit_frame(frame.copy())

            # ── Motion score ──────────────────────────────────────────────────
            gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gblur = cv2.GaussianBlur(gray, (21, 21), 0)
            if prev_gray is None:
                prev_gray = gblur
            diff         = cv2.absdiff(prev_gray, gblur)
            motion_score = float(np.sum(diff > 25))
            prev_gray    = gblur

            # ── Check auth, manage phase transitions ──────────────────────────
            is_auth, auth_bbox, auth_sim = face_worker.get_results()
            now = time.time()

            if current_phase == AppPhase.WAITING:
                if is_auth:
                    current_phase = AppPhase.SUCCESS
                    phase_start_time = now
            elif current_phase == AppPhase.SUCCESS:
                if now - phase_start_time > 1.5:  # hold success for 1.5s
                    current_phase = AppPhase.COUNTDOWN
                    phase_start_time = now
            elif current_phase == AppPhase.COUNTDOWN:
                if now - phase_start_time > 3.0:  # 3s countdown
                    current_phase = AppPhase.ACTIVE
            elif current_phase == AppPhase.ACTIVE:
                if not is_auth:
                    # User left frame for > graceful timeout
                    current_phase = AppPhase.WAITING
                    activity_worker.reset()

            # ── Gate activity detection ───────────────────────────────────────
            if current_phase == AppPhase.ACTIVE:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                activity_worker.submit_frame(rgb, motion_score)

            # ── Read latest activity results ──────────────────────────────────
            posture, intake, pose_result = activity_worker.get_results()
            if posture is None: posture = _idle_posture()
            if intake  is None: intake  = _idle_intake()

            # ── Draw: skeleton + mouth box (only when ACTIVE) ─────────────────
            if current_phase == AppPhase.ACTIVE:
                draw_skeleton(frame, pose_result, mp_drawing, mp_pose)
                draw_mouth_box(frame, intake)

            # ── Frame timing & FPS ────────────────────────────────────────────
            fps_buf.append(time.time() - t0)
            fps = 1.0 / (sum(fps_buf) / len(fps_buf)) if fps_buf else 0

            if screenshot_flash > 0:
                screenshot_flash -= 0.15

            # ── Draw activity HUD (always — frozen at idle when not active) ───
            draw_hud(frame, posture, intake, fps, motion_score, screenshot_flash)

            # ── Draw auth overlay on top of everything ────────────────────────
            remaining = 3 - int(now - phase_start_time)
            countdown_val = max(1, remaining) if current_phase == AppPhase.COUNTDOWN else 0
            
            draw_auth_overlay(frame, current_phase, auth_bbox, auth_sim, frame_count, countdown_val)

            cv2.imshow("Unified Activity Detection", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                sp = shots_dir / f"shot_{int(time.time())}.jpg"
                cv2.imwrite(str(sp), frame)
                shot_count += 1
                screenshot_flash = 1.5
                print(f"  [Screenshot] {sp.name}")
            elif key == ord('r'):
                activity_worker.reset()
                print("  [Reset] Session counters cleared.")

    finally:
        face_worker.stop()
        activity_worker.stop()
        cap.release()
        cv2.destroyAllWindows()
        if shot_count:
            print(f"\n[Done] {shot_count} screenshot(s) -> {shots_dir}")


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Face-Gated Unified Activity Detection — Live Camera + FIR Dataset"
    )
    parser.add_argument("--dataset",    action="store_true",
                        help="Run in dataset slideshow mode (no face auth)")
    parser.add_argument("--video",      default=None,
                        help="Name of a single dataset video folder")
    parser.add_argument("--max-frames", default=20, type=int,
                        help="Max frames per video in dataset mode (0=all)")
    parser.add_argument("--cam",        default=0,  type=int,
                        help="Camera index (default 0)")
    args = parser.parse_args()

    if args.video or args.dataset:
        run_dataset(video_name=args.video,
                    max_frames=args.max_frames if args.max_frames > 0 else None)
    else:
        run_live_camera(cam_index=args.cam)


if __name__ == "__main__":
    main()
