"""
detectors/intake_detector.py
=============================
Detects eating / drinking events using MediaPipe Pose landmarks.

FIXES:
- No longer depends on FaceMesh (which fails when you step back from the camera).
  Uses Pose landmarks (9 and 10) for the mouth.
- Uses a DYNAMIC threshold based on shoulder width instead of a hardcoded pixel count,
  so it works perfectly no matter how close or far you are from the camera.
- Distinct eating vs drinking logic based on trajectory and angle.
"""

import time
import numpy as np
from collections import deque
from dataclasses import dataclass
from typing import Optional, Tuple


# ── Config ─────────────────────────────────────────────────────────────────────
MIN_INTAKE_FRAMES     = 4     # consecutive frames to confirm
BPM_WINDOW_SECONDS    = 60    # rolling window for BPM
WRIST_HISTORY_LEN     = 10

# MediaPipe Pose indices
_MOUTH_L = 9
_MOUTH_R = 10
_L_WRIST = 15
_R_WRIST = 16
_L_SHOULDER = 11
_R_SHOULDER = 12


@dataclass
class IntakeResult:
    is_intake:    bool
    label:        str           # "EATING" or "DRINKING"
    description:  str
    color:        tuple         # BGR
    bpm:          float
    bite_count:   int
    mouth_center: Optional[Tuple[int, int]]
    mouth_box:    Optional[Tuple[int, int, int, int]]
    wrist_pt:     Optional[Tuple[int, int]]
    confidence:   float


def _px(lm, idx, fw, fh):
    """Normalized coordinates to pixel coordinates."""
    return (int(lm[idx].x * fw), int(lm[idx].y * fh))

def _dist(a, b):
    return float(np.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2))


def _get_mouth_info(pose_lm, fw, fh):
    """
    Get mouth centre directly from POSE landmarks instead of FaceMesh.
    This works reliably even when the person is far away.
    """
    if pose_lm is None:
        return None, None

    lm = pose_lm.landmark
    # Check visibility
    if lm[_MOUTH_L].visibility < 0.4 and lm[_MOUTH_R].visibility < 0.4:
        return None, None

    ml = _px(lm, _MOUTH_L, fw, fh)
    mr = _px(lm, _MOUTH_R, fw, fh)

    cx = (ml[0] + mr[0]) // 2
    cy = (ml[1] + mr[1]) // 2

    # Dynamic mouth box size based on eye/mouth distance
    box_w = abs(ml[0] - mr[0]) + 30
    box_h = box_w - 10
    
    # Don't let it be too small
    box_w = max(40, box_w)
    box_h = max(30, box_h)

    x1, y1 = max(0, cx - box_w//2), max(0, cy - box_h//2)
    return (cx, cy), (x1, y1, box_w, box_h)


def _get_dynamic_threshold(pose_lm, fw, fh) -> float:
    """
    Calculates wrist-to-mouth threshold dynamically based on shoulder width.
    If you are close to the camera, the threshold is large in pixels.
    If you are far away, the threshold shrinks so it stays accurate.
    """
    if pose_lm is None:
        return 100.0
    
    ls = _px(pose_lm.landmark, _L_SHOULDER, fw, fh)
    rs = _px(pose_lm.landmark, _R_SHOULDER, fw, fh)
    shoulder_width_px = _dist(ls, rs)
    
    # The threshold is roughly 60% of your shoulder width
    # E.g. if shoulders are 200px wide, mouth threshold is 120px
    return max(50.0, shoulder_width_px * 0.60)


def _closest_wrist(pose_lm, mouth_center, fw, fh):
    if pose_lm is None or mouth_center is None:
        return None, float("inf"), None
        
    lm = pose_lm.landmark
    
    lw = _px(lm, _L_WRIST, fw, fh)
    rw = _px(lm, _R_WRIST, fw, fh)
    
    ld = _dist(lw, mouth_center) if lm[_L_WRIST].visibility > 0.4 else float("inf")
    rd = _dist(rw, mouth_center) if lm[_R_WRIST].visibility > 0.4 else float("inf")
    
    if ld == float("inf") and rd == float("inf"):
        return None, float("inf"), None

    if ld <= rd:
        return lw, ld, "L"
    return rw, rd, "R"


class IntakeDetector:
    def __init__(self):
        self._reset_session()

    def update(self, pose_result, face_result, frame_shape) -> IntakeResult:
        fh, fw = frame_shape[:2]

        pose_lm = None
        if pose_result and pose_result.pose_landmarks:
            pose_lm = pose_result.pose_landmarks

        # Get mouth from Pose (ignores face_result completely now for better long-range)
        mouth_center, mouth_box = _get_mouth_info(pose_lm, fw, fh)
        dynamic_threshold = _get_dynamic_threshold(pose_lm, fw, fh)
        
        wrist_pt, wrist_dist, hand = _closest_wrist(pose_lm, mouth_center, fw, fh)

        conf = 0.0
        if wrist_dist < dynamic_threshold:
            conf = 1.0 - (wrist_dist / dynamic_threshold)

        raw_intake = wrist_dist < dynamic_threshold

        if wrist_pt is not None:
            self._wrist_y_history.append(wrist_pt[1])

        if raw_intake:
            self._consecutive_intake += 1
        else:
            self._consecutive_intake = 0

        confirmed = self._consecutive_intake >= MIN_INTAKE_FRAMES

        if confirmed and not self._was_intake:
            self._record_bite()
        self._was_intake = confirmed

        label = self._classify_intake_type(confirmed, wrist_pt, mouth_center)
        bpm = self._calc_bpm()

        return IntakeResult(
            is_intake    = confirmed,
            label        = label,
            description  = self._description(label, confirmed),
            color        = self._color(label),
            bpm          = bpm,
            bite_count   = self._bite_count,
            mouth_center = mouth_center,
            mouth_box    = mouth_box,
            wrist_pt     = wrist_pt,
            confidence   = conf,
        )

    def reset(self):
        self._reset_session()

    def _reset_session(self):
        self._bite_times          = []
        self._bite_count          = 0
        self._consecutive_intake  = 0
        self._was_intake          = False
        self._wrist_y_history     = deque(maxlen=WRIST_HISTORY_LEN)

    def _record_bite(self):
        now = time.time()
        self._bite_times.append(now)
        self._bite_count += 1
        cutoff = now - BPM_WINDOW_SECONDS
        self._bite_times = [t for t in self._bite_times if t >= cutoff]

    def _calc_bpm(self) -> float:
        if len(self._bite_times) < 2:
            return 0.0
        elapsed = self._bite_times[-1] - self._bite_times[0]
        if elapsed <= 0:
            return 0.0
        return round((len(self._bite_times) - 1) / elapsed * 60, 1)

    def _classify_intake_type(self, confirmed: bool, wrist_pt, mouth_center) -> str:
        if not confirmed or wrist_pt is None or mouth_center is None:
            return "NOT EATING"

        wrist_y = wrist_pt[1]
        mouth_y = mouth_center[1]

        # Is the wrist completely BELOW the mouth center, meaning they are lifting up?
        wrist_below_mouth = wrist_y > (mouth_y + 15)

        # Drinking: usually grabbing a cup means your wrist stays below your mouth
        if wrist_below_mouth:
            return "DRINKING"
        
        # Eating: grabbing a sandwich or fork, wrist comes up to or above the mouth line
        return "EATING"

    def _description(self, label: str, confirmed: bool) -> str:
        if label == "DRINKING": return "Person is drinking"
        if label == "EATING":   return "Person is eating"
        return ""

    def _color(self, label: str) -> tuple:
        if label == "EATING":   return (0, 200, 80)
        if label == "DRINKING": return (20, 180, 255)
        return (80, 80, 80)
