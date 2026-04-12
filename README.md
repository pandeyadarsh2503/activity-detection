# AI Activity Detection — Face-Gated Live Monitoring

Real-time activity detection that **only monitors the registered user**.  
Built on InsightFace (authentication) + MediaPipe Pose (activity classification).

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.10+-brightgreen.svg)

---

## 🚀 Key Features

- **Face-Gated Pipeline**: Activity detection only activates for the pre-registered user — all other people are ignored.
- **Live Camera Mode**: Real-time pose analysis using MediaPipe.
- **Posture Classification**: Detects Standing, Walking, Sitting, Falling, Lying, and Sleeping.
- **Intake Detection**: Detects Eating and Drinking events with Bites Per Minute (BPM) tracking.
- **Dataset Mode**: Visualizes and analyzes ground-truth labels from the FIR Thermal dataset.
- **Threaded Architecture**: Face auth (InsightFace) and activity detection (MediaPipe) run in separate background threads for smooth 30+ FPS display.

---

## 🛠️ Installation

### 1. Clone the repository
```bash
git clone https://github.com/pandeyadarsh2503/activity-detection.git
cd activity-detection
```

### 2. Install dependencies
```bash
pip install opencv-python numpy mediapipe insightface onnxruntime
```

> **Note**: `insightface` will automatically download the `buffalo_s` model pack (~80 MB) on first run.

---

## 💻 Usage

### Step 1 — Register your face (run once)
```bash
python register_face.py
```
- Opens your webcam.  
- Press **SPACE** 5 times to capture photos of your face.  
- Saves your face embedding to `face_auth/registered_face.npy`.

### Step 2 — Run live detection
```bash
python run_inference.py
```
- The system shows **"FACE AUTHENTICATION REQUIRED"** until it recognises you.
- Once your face matches, activity detection (posture + intake) starts automatically.
- If you leave the frame for more than 2 seconds, it reverts to the waiting state.

**Controls:**
| Key | Action |
|-----|--------|
| `Q` | Quit |
| `S` | Save screenshot |
| `R` | Reset session counters |

### Dataset Mode (no face auth)
```bash
# All videos (first 20 frames each)
python run_inference.py --dataset

# Single video
python run_inference.py --video video105 --max-frames 50

# Change camera index
python run_inference.py --cam 1
```

---

## 📂 Project Structure

```
localisation_activity/
├── run_inference.py          # Main entry point
├── register_face.py          # One-time face registration script
│
├── face_auth/                # InsightFace authentication module
│   ├── face_engine.py        # FaceEngine wrapper (detect + embed + match)
│   ├── config.py             # Tunable auth parameters
│   └── registered_face.npy  # Saved face embedding (git-ignored)
│
├── detectors/                # Activity detection logic
│   ├── posture_detector.py   # MediaPipe Pose → posture classification
│   └── intake_detector.py    # Wrist-to-mouth → eating / drinking
│
├── display/                  # HUD rendering
│   └── hud.py                # draw_hud(), draw_auth_overlay(), etc.
│
├── model_code/               # FIR dataset model architecture
└── dataset/                  # FIR thermal dataset (not included — see below)
```

---

## ⚙️ Configuration

Tune authentication behaviour in `face_auth/config.py`:

| Parameter | Default | Description |
|---|---|---|
| `SIMILARITY_THRESHOLD` | `0.45` | Cosine similarity cutoff for a positive match |
| `AUTH_CONSECUTIVE_FRAMES` | `10` | Consecutive matching frames needed to authenticate |
| `IDENTITY_GRACE_SECONDS` | `2.0` | Seconds to stay authenticated after face disappears |
| `FACE_SKIP_FRAMES` | `3` | Re-embed every N frames (reduce CPU) |

---

## 📊 Dataset Attribution

1. **[ThomasDubail/FIR-Image-Action-Localisation-Dataset](https://github.com/ThomasDubail/FIR-Image-Action-Localisation-Dataset)**: Annotated bounding boxes.
2. **[noahzhy/FIR-Image-Action-Dataset](https://github.com/noahzhy/FIR-Image-Action-Dataset)**: Original FIR sensor data.

> The full `dataset/` folder (~3.5 GB) is excluded. Download and place it in the project root to use Dataset Mode.

---

## ⚖️ License
For educational and research purposes. Refer to the origin datasets for their respective licenses.
