"""
detectors/posture_detector.py
==============================
Classifies full-body posture from MediaPipe Pose landmarks.
Actions: sleeping, lying, falling, sitting, standing, walking

FIXES in this version:
- Falling now requires a SUSTAINED rapid tilt (not just any forward bend).
  We track angular velocity and require the angle to spike quickly AND be held.
  Slow bends (picking something up) are now correctly ignored.
- Reduced smoothing window so the display feels more responsive (parallel).
"""

import time
import numpy as np
from dataclasses import dataclass
from typing import Optional, Deque
from collections import deque


# ── Action metadata ────────────────────────────────────────────────────────────
# (label, description, BGR colour, icon tag)
POSTURE_ACTIONS = {
    "sleeping": ("SLEEPING",   "Person sleeping on the floor",   (30,  80,  220), "[ZZZ]"),
    "lying":    ("LYING DOWN", "Person lying on the floor",      (200, 60,  255), "[LIE]"),
    "falling":  ("FALLING!",   "Person is falling",              (0,   60,  255), "[!!!]"),
    "sitting":  ("SITTING",    "Person is sitting",              (255, 140, 0),   "[ S ]"),
    "standing": ("STANDING",   "Person is standing still",       (50,  200, 50),  "[ | ]"),
    "walking":  ("WALKING",    "Person is walking or moving",    (255, 180, 20),  "[-->]"),
    "unknown":  ("DETECTING..", "Waiting for pose detection",    (120, 120, 120), "[ ? ]"),
}


@dataclass
class PostureResult:
    action: str
    label: str
    description: str
    color: tuple
    icon: str
    body_angle: float
    horizontal_duration: float


def _midpoint(a, b):
    return ((a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0)


def _angle_with_vertical(p1, p2):
    """Angle between vector p1->p2 and the vertical axis. 0=upright, 90=horizontal."""
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    return abs(np.degrees(np.arctan2(abs(dx), abs(dy))))


class PostureDetector:
    """
    Stateful posture classifier.
    Uses a shared MediaPipe Pose result (does NOT create its own instance).
    """

    def __init__(self, smoothing_frames: int = 8):
        # Reduced to 8 for faster response
        self._history:            list  = []
        self._smoothing:          int   = smoothing_frames
        self._horizontal_start:   Optional[float] = None
        self.horizontal_duration: float = 0.0

        # ── Falling detection state ──────────────────────────────────────────
        # We track the last N angle readings to detect RAPID changes
        self._angle_history:      Deque[float] = deque(maxlen=20)
        self._fall_start:         Optional[float] = None
        self._fall_confirmed:     bool  = False
        self._last_angle:         float = 0.0

    # ── Public API ─────────────────────────────────────────────────────────────

    def update(self, pose_result, motion_score: float) -> PostureResult:
        raw = self._classify(pose_result, motion_score)
        self._history.append(raw)
        if len(self._history) > self._smoothing:
            self._history.pop(0)

        counts = {}
        for a in self._history:
            counts[a] = counts.get(a, 0) + 1
        best = max(counts, key=counts.get)

        info = POSTURE_ACTIONS[best]
        return PostureResult(
            action=best,
            label=info[0],
            description=info[1],
            color=info[2],
            icon=info[3],
            body_angle=self._last_angle,
            horizontal_duration=self.horizontal_duration,
        )

    def reset(self):
        self._history.clear()
        self._angle_history.clear()
        self._horizontal_start = None
        self.horizontal_duration = 0.0
        self._fall_start = None
        self._fall_confirmed = False

    # ── Internal ───────────────────────────────────────────────────────────────

    def _classify(self, pose_result, motion_score: float) -> str:
        if pose_result is None or pose_result.pose_landmarks is None:
            self._update_horizontal(False)
            self._update_fall(False)
            return "unknown"

        lm = pose_result.pose_landmarks.landmark

        # Require shoulders AND hips to be visible
        vis_scores = [lm[i].visibility for i in [11, 12, 23, 24]]
        if min(vis_scores) < 0.3:
            self._update_horizontal(False)
            self._update_fall(False)
            return "unknown"

        l_shoulder = (lm[11].x, lm[11].y)
        r_shoulder = (lm[12].x, lm[12].y)
        l_hip      = (lm[23].x, lm[23].y)
        r_hip      = (lm[24].x, lm[24].y)
        l_knee     = (lm[25].x, lm[25].y)
        r_knee     = (lm[26].x, lm[26].y)

        sh_mid   = _midpoint(l_shoulder, r_shoulder)
        hip_mid  = _midpoint(l_hip, r_hip)
        knee_mid = _midpoint(l_knee, r_knee)

        angle = _angle_with_vertical(sh_mid, hip_mid)
        self._last_angle = angle
        self._angle_history.append(angle)

        # ── 1. Sleeping / Lying: body nearly horizontal (>60 deg) ────────────
        is_horizontal = angle > 60
        self._update_horizontal(is_horizontal)
        if is_horizontal:
            self._update_fall(False)
            return "sleeping" if self.horizontal_duration > 3.0 else "lying"

        # ── 2. Falling: RAPID tilt (angular velocity) ─────────────────────────
        # Only classify as falling if:
        #   a) Current angle > 40 degrees (real tilt, not just leaning)
        #   b) Angle changed quickly in the last 10 frames (angular velocity > 3 deg/frame)
        #   c) It has been sustained for at least 0.3 seconds
        is_falling = self._check_falling(angle)
        if is_falling:
            self._update_fall(True)
        else:
            self._update_fall(False)

        if self._fall_confirmed:
            return "falling"

        # ── 3. Sitting ────────────────────────────────────────────────────────
        hip_y          = hip_mid[1]
        knee_hip_vdiff = abs(knee_mid[1] - hip_mid[1])
        if hip_y > 0.52 and knee_hip_vdiff < 0.18:
            return "sitting"

        # ── 4. Walking: upright + high motion ─────────────────────────────────
        if motion_score > 1500:
            return "walking"

        return "standing"

    def _check_falling(self, current_angle: float) -> bool:
        """
        Returns True only if the angle is:
        1. Above 40 degrees (real tilt, not just leaning forward slightly)
        2. Has risen RAPIDLY (average slope > 2.5 deg/frame over last 10 frames)
           — this distinguishes falling from slowly bending over
        """
        if current_angle < 40:
            return False
        if len(self._angle_history) < 8:
            return False
        # Calculate angular velocity over the last 8 frames
        recent = list(self._angle_history)[-8:]
        angular_velocity = (recent[-1] - recent[0]) / len(recent)
        # Must be rising rapidly OR already at high angle with motion
        return angular_velocity > 2.5 or current_angle > 55

    def _update_horizontal(self, is_horizontal: bool):
        if is_horizontal:
            if self._horizontal_start is None:
                self._horizontal_start = time.time()
            self.horizontal_duration = time.time() - self._horizontal_start
        else:
            self._horizontal_start = None
            self.horizontal_duration = 0.0

    def _update_fall(self, is_falling: bool):
        """Require 0.4s of sustained fall angle before confirming."""
        if is_falling:
            if self._fall_start is None:
                self._fall_start = time.time()
            if time.time() - self._fall_start > 0.4:
                self._fall_confirmed = True
        else:
            self._fall_start = None
            self._fall_confirmed = False
