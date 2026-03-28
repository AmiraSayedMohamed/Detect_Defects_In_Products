<div align="center">

# 🍶 AI-Based Bottle Inspection & Quality Control System

[![YOLOv8](https://img.shields.io/badge/Model-YOLOv8-blue?logo=pytorch)](https://github.com/ultralytics/ultralytics)
[![Roboflow](https://img.shields.io/badge/Dataset-Roboflow-purple?logo=roboflow)](https://universe.roboflow.com/biang-suosk/bottle-defects-detection)
[![Raspberry Pi](https://img.shields.io/badge/Hardware-Raspberry%20Pi%205-red?logo=raspberry-pi)](https://www.raspberrypi.com/)
[![Python](https://img.shields.io/badge/Python-3.10%2B-yellow?logo=python)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **An intelligent, low-cost prototype for automated bottle inspection on a small-scale production line — powered by AI, computer vision, and Industry 4.0 principles.**

</div>

---

## 📋 Table of Contents

1. [Project Overview](#-project-overview)
2. [Key Features](#-key-features)
3. [Detected Defects](#-detected-defects)
4. [System Architecture](#-system-architecture)
5. [Hardware Components](#-hardware-components)
6. [Dataset](#-dataset)
7. [Model — YOLOv8](#-model--yolov8)
8. [Installation](#-installation)
9. [Usage](#-usage)
10. [Project Structure](#-project-structure)
11. [Results](#-results)
12. [Sustainability & Industry 4.0](#-sustainability--industry-40)
13. [Contributing](#-contributing)
14. [License](#-license)

---

## 🔍 Project Overview

Traditional manual inspection processes on production lines are often **time-consuming, error-prone, and inconsistent**, leading to increased waste, higher operational costs, and lower customer satisfaction.

This project addresses these challenges by presenting an **intelligent, automated quality control system** for bottle inspection. Using a **Raspberry Pi 5** paired with a camera module, capacitive sensing, and a **YOLOv8** object detection model, the system performs real-time defect detection without expensive industrial equipment.

| Aspect | Detail |
|---|---|
| **Goal** | Automate bottle quality control using AI & computer vision |
| **Core Model** | YOLOv8 (Object Detection) |
| **Platform** | Raspberry Pi 5 |
| **Dataset** | [Bottle Defects Detection — Roboflow](https://universe.roboflow.com/biang-suosk/bottle-defects-detection) |
| **Focus** | Low-cost, scalable, real-time inspection |

---

## ✨ Key Features

- 🤖 **Real-Time Defect Detection** — Continuous frame-by-frame inspection using YOLOv8
- 📷 **Camera-Based Vision** — High-resolution image capture integrated with Raspberry Pi 5
- 🔌 **Capacitive Sensing** — Detects bottle presence on the conveyor to trigger inspection
- 📋 **Defect Logging** — Timestamped logs with defect type, severity, and image snapshot
- 🚫 **Simulated PLC Rejection System** — Flags and virtually removes defective bottles from the line
- 🔔 **Maintenance Alerts** — Triggers notifications when defect rates exceed predefined thresholds
- 📊 **Dashboard-Ready Output** — Structured logs suitable for integration with monitoring dashboards
- ♻️ **Sustainability-Oriented** — Reduces material waste and supports energy-efficient manufacturing

---

## 🧪 Detected Defects

The system is trained and capable of detecting the following bottle defects:

| # | Defect Type | Description |
|---|---|---|
| 1 | **Cap Misalignment** | Bottle cap is improperly seated, tilted, or missing |
| 2 | **Missing Label** | Label is absent from the bottle body |
| 3 | **Incorrect Label** | Wrong label applied to the bottle |
| 4 | **Improper Fill Level** | Bottle is underfilled or overfilled beyond acceptable range |

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                   Production Line (Conveyor)                 │
└────────────────────────────┬────────────────────────────────┘
                             │ Bottle arrives
                             ▼
                  ┌─────────────────────┐
                  │  Capacitive Sensor   │  ◄─── Detects bottle presence
                  └────────┬────────────┘
                           │ Trigger
                           ▼
                  ┌─────────────────────┐
                  │  Raspberry Pi 5     │
                  │  + Camera Module    │  ◄─── Captures frame
                  └────────┬────────────┘
                           │ Image frame
                           ▼
                  ┌─────────────────────┐
                  │  YOLOv8 AI Model    │  ◄─── Runs inference
                  └────────┬────────────┘
                           │ Detection result
              ┌────────────┴────────────────┐
              │                             │
              ▼                             ▼
   ┌─────────────────────┐      ┌─────────────────────┐
   │   ✅ PASS           │      │   ❌ DEFECT FOUND    │
   │   Bottle accepted   │      │   Defect logged      │
   └─────────────────────┘      │   PLC rejection      │
                                │   Alert if threshold │
                                └─────────────────────┘
```

---

## 🔧 Hardware Components

| Component | Purpose |
|---|---|
| **Raspberry Pi 5** | Central processing unit — runs AI model and control logic |
| **Pi Camera Module** | Captures high-resolution images of each bottle |
| **Capacitive Sensor** | Detects bottle presence on the conveyor belt |
| **LED Indicators** | Visual pass/fail signals on the production line |
| **Conveyor Belt (prototype)** | Simulates small-scale production line movement |
| **Simulated PLC** | Software-based Programmable Logic Controller for rejection logic |

---

## 📦 Dataset

| Property | Value |
|---|---|
| **Source** | [Roboflow Universe](https://universe.roboflow.com/biang-suosk/bottle-defects-detection) |
| **Format** | YOLOv8 compatible (YOLO annotation format) |
| **Classes** | Cap defects, label defects, fill level defects |
| **Augmentations** | Rotation, flipping, brightness adjustment, noise injection |

To download the dataset using the Roboflow Python package:

```python
from roboflow import Roboflow

rf = Roboflow(api_key="YOUR_API_KEY")
project = rf.workspace("biang-suosk").project("bottle-defects-detection")
dataset = project.version(1).download("yolov8")
```

---

## 🧠 Model — YOLOv8

This project uses **YOLOv8** (You Only Look Once, version 8) by [Ultralytics](https://github.com/ultralytics/ultralytics) — one of the most efficient and accurate real-time object detection models available.

### Why YOLOv8?

- ⚡ **Fast inference** — suitable for real-time embedded deployment on Raspberry Pi
- 🎯 **High accuracy** — excellent mAP on small object datasets
- 🔄 **Easy fine-tuning** — supports transfer learning from pre-trained weights
- 📦 **Lightweight export** — supports ONNX, TFLite, and Edge TPU formats for edge deployment

### Training Configuration

```yaml
model: yolov8n.pt       # Nano variant — optimized for edge devices
data: dataset/data.yaml
epochs: 100
imgsz: 640
batch: 16
optimizer: AdamW
patience: 20            # Early stopping
```

### Training the Model

```bash
yolo detect train \
  model=yolov8n.pt \
  data=dataset/data.yaml \
  epochs=100 \
  imgsz=640 \
  batch=16
```

---

## ⚙️ Installation

### Prerequisites

- Python 3.10+
- pip
- (For Raspberry Pi) Raspberry Pi OS (64-bit recommended)

### 1. Clone the Repository

```bash
git clone https://github.com/AmiraSayedMohamed/Detect_Defects_In_Products.git
cd Detect_Defects_In_Products
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate        # Linux / macOS
# venv\Scripts\activate         # Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Download the Dataset

Follow the [Dataset](#-dataset) section above to download and place the dataset inside the `dataset/` folder.

---

## 🚀 Usage

### Run Inference on an Image

```bash
python detect.py --source path/to/bottle_image.jpg --weights runs/train/weights/best.pt
```

### Run Real-Time Inference from Camera

```bash
python detect.py --source 0 --weights runs/train/weights/best.pt
```

### Run the Full Inspection Pipeline (Raspberry Pi)

```bash
python inspection_pipeline.py
```

This script:
1. Waits for the capacitive sensor to detect a bottle
2. Captures a frame from the camera
3. Runs YOLOv8 inference
4. Logs the result with timestamp and defect type
5. Triggers the PLC rejection signal if a defect is found
6. Raises a maintenance alert if the defect rate exceeds the threshold

### View Defect Logs

```bash
cat logs/defect_log.csv
```

---

## 📁 Project Structure

```
Detect_Defects_In_Products/
│
├── dataset/                    # Training dataset (YOLOv8 format)
│   ├── images/
│   │   ├── train/
│   │   └── val/
│   ├── labels/
│   │   ├── train/
│   │   └── val/
│   └── data.yaml
│
├── models/                     # Saved model weights
│   └── best.pt
│
├── runs/                       # Training outputs (auto-generated)
│   └── train/
│       └── weights/
│
├── logs/                       # Defect logs
│   └── defect_log.csv
│
├── detect.py                   # Inference script
├── inspection_pipeline.py      # Full pipeline for Raspberry Pi
├── train.py                    # Training script
├── requirements.txt            # Python dependencies
└── README.md
```

---

## 📊 Results

> *(Results will be updated after training is complete.)*

| Metric | Value |
|---|---|
| **mAP@0.5** | — |
| **mAP@0.5:0.95** | — |
| **Precision** | — |
| **Recall** | — |
| **Inference Speed (Raspberry Pi 5)** | — |

---

## 🌱 Sustainability & Industry 4.0

This project aligns with the principles of **Industry 4.0** and **sustainable manufacturing**:

- ♻️ **Waste Reduction** — Accurate defect detection minimizes false rejects and product waste
- ⚡ **Energy Efficiency** — Lightweight YOLOv8 nano model reduces computational energy consumption
- 💰 **Low Cost** — Built on affordable hardware (Raspberry Pi 5), making AI inspection accessible to small producers
- 🔗 **Smart Manufacturing** — Demonstrates how AI-driven automation can replace error-prone manual inspection
- 📈 **Scalability** — The architecture can be scaled to larger production lines and extended to detect additional defect types
- 🛠️ **Proactive Maintenance** — Threshold-based alerts enable predictive maintenance, reducing downtime

---

## 🤝 Contributing

Contributions are welcome! To contribute:

1. Fork the repository
2. Create a new branch: `git checkout -b feature/your-feature-name`
3. Commit your changes: `git commit -m "Add your feature"`
4. Push to the branch: `git push origin feature/your-feature-name`
5. Open a Pull Request

Please follow the existing code style and include relevant documentation for new features.

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---

<div align="center">

**Built with ❤️ using AI, Computer Vision, and Raspberry Pi**

*Contributing to a smarter, more sustainable future in manufacturing.*

</div>
