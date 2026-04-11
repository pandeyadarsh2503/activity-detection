# AI Activity Detection (FIR Thermal & Live Pose)

This project performs real-time activity detection using body pose estimation. It supports both analysis of the **FIR-Image-Action-Localisation-Dataset** and **Live Camera Detection**.

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.10+-brightgreen.svg)

---

## 🚀 Key Features

- **Live Camera Mode**: Real-time pose analysis using MediaPipe.
- **Action Classification**: Detects Standing, Walking, Sitting, Falling, Lying, and Sleeping.
- **Dataset Mode**: Visualizes and analyzes ground-truth labels from the FIR Thermal dataset.
- **Sleeping Detection**: Specific logic to identify prolonged lying as "Sleeping" (ideal for health monitoring).

---

## 🛠️ Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/pandeyadarsh2503/activity-detection.git
   cd activity-detection
   ```

2. **Install dependencies**:
   ```bash
   pip install opencv-python numpy matplotlib pillow mediapipe
   ```

---

## 💻 Usage

### 1. Live Camera Mode (Recommended)
Automatically detects your movements via webcam:
```bash
python run_inference.py
```
- **Controls**: 
  - `Q`: Quit
  - `S`: Save screenshot
  - `R`: Reset history

### 2. Dataset Evaluation Mode
Runs the inference pipeline on the FIR localization dataset:
```bash
# Process all videos (first 20 frames each)
python run_inference.py --dataset --max-frames 20

# Evaluate a specific video (e.g., video105)
python run_inference.py --video video105
```

---

## 📊 Dataset Attribution

This project utilizes data and inspiration from:
1. **[ThomasDubail/FIR-Image-Action-Localisation-Dataset](https://github.com/ThomasDubail/FIR-Image-Action-Localisation-Dataset)**: Annotated bounding boxes and temporal sync.
2. **[noahzhy/FIR-Image-Action-Dataset](https://github.com/noahzhy/FIR-Image-Action-Dataset)**: Original FIR sensor data and model architectures.

> **Note**: The full `dataset/` folder is excluded from this repository due to its large size (3.5 GB). Please download it from the links above and place it in the project root to use Dataset Mode.

---

## 📂 Project Structure

- `run_inference.py`: Main entry point (Live Camera + Dataset logic).
- `model_code/`: Contains the CNN/LSTM model architecture definitions.
- `demo_sleeping_frames.py`: Utility to extract specific action frames.
- `output_results/`: (Ignored) Destination for analysis charts and screenshots.

---

## ⚖️ License
This project is for educational and research purposes. Please refer to the origin datasets for their respective usage licenses.
