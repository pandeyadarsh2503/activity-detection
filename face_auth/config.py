"""
face_auth/config.py
====================
Configuration constants for the InsightFace authentication system.
All tunable parameters are centralised here.
"""

import os

# ── Model Settings ──────────────────────────────────────────────────────────
MODEL_NAME = "buffalo_s"          # Faster model pack (MobileFaceNet backbone)
DET_SIZE   = (320, 320)           # Detection resolution (lower = faster)
CTX_ID     = -1                   # -1 = CPU; set to 0 for CUDA GPU

# ── Registration ────────────────────────────────────────────────────────────
REGISTRATION_IMAGES = 5           # Face captures required during registration
EMBEDDING_FILE = os.path.join(
    os.path.dirname(__file__), "registered_face.npy"
)

# ── Matching / Auth ──────────────────────────────────────────────────────────
SIMILARITY_THRESHOLD   = 0.45    # Cosine similarity cutoff for positive match
AUTH_CONSECUTIVE_FRAMES = 10     # Consecutive matching frames needed to auth
IDENTITY_GRACE_SECONDS  = 2.0   # Seconds to keep detection alive after face lost
FACE_SKIP_FRAMES        = 3      # Re-embed every N frames (reduces CPU load)

# ── Tracking / Smoothing ────────────────────────────────────────────────────
EMA_ALPHA      = 0.3             # Bbox EMA smoothing (0=frozen, 1=no smoothing)
IOU_THRESHOLD  = 0.5             # IoU cutoff to associate tracked faces

# ── Display ─────────────────────────────────────────────────────────────────
BOX_COLOR         = (0, 255, 0)   # Green (BGR)
BOX_THICKNESS     = 2
LABEL_FONT_SCALE  = 0.7
LABEL_COLOR       = (255, 255, 255)
AUTHORIZED_LABEL  = "AUTHORIZED"
