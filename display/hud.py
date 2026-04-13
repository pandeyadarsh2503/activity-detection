"""
display/hud.py
==============
Renders the unified HUD overlay.

FIXES in this version:
- Intake panel is HIDDEN when not eating/drinking.
  It only lights up and shows when EATING or DRINKING is detected.
- When eating/drinking IS happening, a bright colour badge pops in.
- Cleaner bottom bar — only shows BPM and bites when relevant.
- Added draw_auth_overlay() for face-authentication gating.
"""

import cv2
import numpy as np

_FONT      = cv2.FONT_HERSHEY_DUPLEX
_FONT_MONO = cv2.FONT_HERSHEY_SIMPLEX


def _alpha_rect(frame, x1, y1, x2, y2, color, alpha=0.72):
    """Semi-transparent filled rectangle (in-place)."""
    y1 = max(0, y1); x1 = max(0, x1)
    y2 = min(frame.shape[0], y2); x2 = min(frame.shape[1], x2)
    if y2 <= y1 or x2 <= x1:
        return
    roi     = frame[y1:y2, x1:x2]
    overlay = roi.copy()
    cv2.rectangle(overlay, (0, 0), (x2 - x1, y2 - y1), color, -1)
    cv2.addWeighted(overlay, alpha, roi, 1 - alpha, 0, roi)
    frame[y1:y2, x1:x2] = roi


def draw_skeleton(frame, pose_result, mp_drawing, mp_pose):
    """Draw pose skeleton (in-place)."""
    if pose_result and pose_result.pose_landmarks:
        mp_drawing.draw_landmarks(
            frame,
            pose_result.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing.DrawingSpec(
                color=(0, 220, 255), thickness=2, circle_radius=3
            ),
            connection_drawing_spec=mp_drawing.DrawingSpec(
                color=(200, 200, 200), thickness=2
            ),
        )


def draw_mouth_box(frame, intake_result):
    """
    Draw mouth bounding box and wrist highlight.
    Always draw the mouth box (grey when idle, coloured when active).
    """
    # Mouth box: always visible (grey normally, bright when eating)
    if intake_result.mouth_box is not None:
        x, y, w, h = intake_result.mouth_box
        if intake_result.is_intake:
            color = intake_result.color
            # Glowing double border
            cv2.rectangle(frame, (x - 3, y - 3), (x + w + 3, y + h + 3), color, 1)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        else:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (50, 50, 50), 1)

    # Wrist indicator: only when actively eating/drinking
    if intake_result.wrist_pt is not None and intake_result.is_intake:
        wx, wy = intake_result.wrist_pt
        cv2.circle(frame, (wx, wy), 16, intake_result.color, -1)
        cv2.circle(frame, (wx, wy), 18, (255, 255, 255), 2)
        # Label
        cv2.putText(frame, intake_result.label, (wx + 20, wy),
                    _FONT_MONO, 0.6, intake_result.color, 2, cv2.LINE_AA)


def draw_hud(
    frame,
    posture_result,
    intake_result,
    fps: float,
    motion_score: float,
    screenshot_flash: float = 0.0,
):
    """
    Draw the unified HUD on top of the frame (in-place).

    Layout:
      TOP BANNER  — posture action (always shown)
      INTAKE BADGE — pops in on the RIGHT only when eating/drinking
      BOTTOM BAR  — stats
      SIDE STRIPS — colour accent bars
    """
    fh, fw = frame.shape[:2]

    banner_h = 76
    bar_h    = 62
    strip_w  = 6

    # ── TOP BANNER (posture — always visible) ──────────────────────────────────
    _alpha_rect(frame, 0, 0, fw, banner_h, (12, 12, 12), alpha=0.82)

    p_color = posture_result.color
    p_icon  = posture_result.icon
    p_label = posture_result.label

    cv2.putText(frame, "POSTURE", (strip_w + 10, 20),
                _FONT_MONO, 0.42, (160, 160, 160), 1, cv2.LINE_AA)
    cv2.putText(frame, f"{p_icon} {p_label}", (strip_w + 10, 58),
                _FONT, 1.05, p_color, 2, cv2.LINE_AA)

    # FPS on the right of the banner (always)
    fps_text = f"{fps:.0f} FPS"
    (tw, _), _ = cv2.getTextSize(fps_text, _FONT_MONO, 0.45, 1)
    cv2.putText(frame, fps_text, (fw - tw - 14, 20),
                _FONT_MONO, 0.45, (90, 90, 90), 1, cv2.LINE_AA)

    # ── INTAKE ALERT BADGE (only when eating or drinking) ─────────────────────
    if intake_result.is_intake:
        i_label  = intake_result.label   # "EATING" or "DRINKING"
        i_color  = intake_result.color
        _icons   = {"EATING": "[EAT]", "DRINKING": "[SIP]"}
        i_icon   = _icons.get(i_label, "[EAT]")

        # Right-side badge, same height as banner
        mid = fw // 2
        cv2.line(frame, (mid, 0), (mid, banner_h), (60, 60, 60), 1)
        # Subtle coloured background on right half
        _alpha_rect(frame, mid, 0, fw, banner_h, i_color, alpha=0.14)
        cv2.putText(frame, "INTAKE", (mid + 10, 20),
                    _FONT_MONO, 0.42, (200, 200, 200), 1, cv2.LINE_AA)
        cv2.putText(frame, f"{i_icon} {i_label}", (mid + 10, 58),
                    _FONT, 1.05, i_color, 2, cv2.LINE_AA)

        # Bites & BPM inline
        stats_text = f"Bites: {intake_result.bite_count}   BPM: {intake_result.bpm:.1f}"
        cv2.putText(frame, stats_text, (fw - 280, 20),
                    _FONT_MONO, 0.40, i_color, 1, cv2.LINE_AA)

        # Right accent strip: intake colour
        cv2.rectangle(frame, (fw - strip_w, 0), (fw, banner_h), i_color, -1)

    # ── BOTTOM BAR ─────────────────────────────────────────────────────────────
    bar_y = fh - bar_h
    _alpha_rect(frame, 0, bar_y, fw, fh, (12, 12, 12), alpha=0.82)
    cv2.line(frame, (0, bar_y), (fw, bar_y), (50, 50, 50), 1)

    # Description
    desc = posture_result.description
    if intake_result.is_intake:
        desc += f"  |  {intake_result.description}"
    cv2.putText(frame, desc, (strip_w + 10, bar_y + 22),
                _FONT_MONO, 0.50, (200, 200, 200), 1, cv2.LINE_AA)

    # Stats
    angle_str = f"Body angle: {posture_result.body_angle:.1f}deg"
    horiz_str = ""
    if posture_result.action in ("sleeping", "lying"):
        horiz_str = f"  |  On floor: {posture_result.horizontal_duration:.0f}s"
    motion_str = f"  |  Motion: {int(motion_score)}"
    stats_line = f"{angle_str}{horiz_str}{motion_str}"
    cv2.putText(frame, stats_line, (strip_w + 10, bar_y + 46),
                _FONT_MONO, 0.42, (100, 100, 100), 1, cv2.LINE_AA)

    # Controls hint (right-aligned)
    ctrl = "Q=Quit  S=Screenshot  R=Reset"
    (tw, _), _ = cv2.getTextSize(ctrl, _FONT_MONO, 0.38, 1)
    cv2.putText(frame, ctrl, (fw - tw - 10, fh - 6),
                _FONT_MONO, 0.38, (70, 70, 70), 1, cv2.LINE_AA)

    # ── LEFT COLOUR STRIP (posture) ────────────────────────────────────────────
    cv2.rectangle(frame, (0, banner_h), (strip_w, bar_y), p_color, -1)

    # ── SCREENSHOT FLASH ───────────────────────────────────────────────────────
    if screenshot_flash > 0:
        flash_alpha = min(0.5, screenshot_flash * 0.35)
        _alpha_rect(frame, 0, 0, fw, fh, (255, 255, 255), alpha=flash_alpha)


def draw_auth_overlay(
    frame,
    app_phase: int,
    auth_bbox,
    auth_sim: float,
    frame_count: int = 0,
    countdown_val: int = 0,
):
    """
    Draw face-authentication state on the frame (in-place).

    Parameters
    ----------
    app_phase : int
        0 = WAITING, 1 = SUCCESS, 2 = COUNTDOWN, 3 = ACTIVE
    auth_bbox : np.ndarray | None
        [x1, y1, x2, y2] of the matched face, or None.
    auth_sim : float
        Latest cosine similarity score (0-1).
    frame_count : int
        Running frame counter — used to animate the waiting pulse.
    countdown_val : int
        Remaining seconds for COUNTDOWN phase.
    """
    fh, fw = frame.shape[:2]
    cy = fh // 2
    cx = fw // 2

    # ── Phase 0: WAITING ───────────────────────────────────────────────────────
    if app_phase == 0:
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (fw, fh), (10, 10, 10), -1)
        cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

        pulse = 0.55 + 0.45 * abs(np.sin(frame_count * 0.06))  # 0.1–1.0
        text_color = tuple(int(c * pulse) for c in (80, 220, 80))   # dim green

        line1 = "FACE AUTHENTICATION REQUIRED"
        line2 = "Please look at the camera"
        hint  = "Run  python register_face.py  first if not registered"

        (tw1, th1), _ = cv2.getTextSize(line1, _FONT, 0.9, 2)
        (tw2, th2), _ = cv2.getTextSize(line2, _FONT_MONO, 0.6, 1)
        (twh, thh), _ = cv2.getTextSize(hint,  _FONT_MONO, 0.38, 1)

        cv2.putText(frame, line1, (cx - tw1 // 2, cy - 20), _FONT, 0.9, text_color, 2, cv2.LINE_AA)
        cv2.putText(frame, line2, (cx - tw2 // 2, cy + 20), _FONT_MONO, 0.6, (160, 160, 160), 1, cv2.LINE_AA)
        cv2.putText(frame, hint, (cx - twh // 2, fh - 18), _FONT_MONO, 0.38, (90, 90, 90), 1, cv2.LINE_AA)
        return

    # Draw Green face bounding box in all phases after WAITING
    if auth_bbox is not None:
        x1, y1, x2, y2 = int(auth_bbox[0]), int(auth_bbox[1]), \
                          int(auth_bbox[2]), int(auth_bbox[3])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 230, 80), 2)
        corner_len = 14
        corner_clr = (120, 255, 140)
        for dx, dy in [(0, 0), (x2 - x1, 0), (0, y2 - y1), (x2 - x1, y2 - y1)]:
            bx, by = x1 + dx, y1 + dy
            sign_x = 1 if dx == 0 else -1
            sign_y = 1 if dy == 0 else -1
            cv2.line(frame, (bx, by), (bx + sign_x * corner_len, by), corner_clr, 2)
            cv2.line(frame, (bx, by), (bx, by + sign_y * corner_len), corner_clr, 2)

    # ── Phase 1: SUCCESS ───────────────────────────────────────────────────────
    if app_phase == 1:
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, cy - 80), (fw, cy + 80), (0, 40, 10), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        msg = "AUTHORIZATION SUCCESSFUL"
        (tw, th), _ = cv2.getTextSize(msg, _FONT, 1.2, 2)
        cv2.putText(frame, msg, (cx - tw // 2, cy + 10), _FONT, 1.2, (60, 255, 100), 2, cv2.LINE_AA)

    # ── Phase 2: COUNTDOWN ─────────────────────────────────────────────────────
    elif app_phase == 2:
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (fw, fh), (10, 10, 10), -1)
        cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)

        # Draw huge countdown number
        msg = str(countdown_val)
        font_scale = 5.0 + abs(np.sin(frame_count * 0.15)) * 0.5  # Pop effect
        (tw, th), _ = cv2.getTextSize(msg, _FONT, font_scale, 6)
        cv2.putText(frame, msg, (cx - tw // 2, cy + th // 2), _FONT, font_scale, (255, 200, 50), 6, cv2.LINE_AA)
        
        hint = "Getting ready..."
        (tw_h, th_h), _ = cv2.getTextSize(hint, _FONT, 0.8, 1)
        cv2.putText(frame, hint, (cx - tw_h // 2, cy + 80), _FONT, 0.8, (200, 200, 200), 1, cv2.LINE_AA)

    # ── Phase 3: ACTIVE ────────────────────────────────────────────────────────
    elif app_phase == 3:
        badge = f"\u25cf AUTHORIZED  {auth_sim:.2f}"
        (bw, bh), _ = cv2.getTextSize(badge, _FONT_MONO, 0.44, 1)
        bx = fw - bw - 20
        cv2.putText(frame, badge, (bx, 52), _FONT_MONO, 0.44, (60, 230, 90), 1, cv2.LINE_AA)
