# AI-Based Bottle Inspection & Quality Control System

![Bottle Inspection System](1-bottle.jpeg)

**A smart, low-cost AI-powered prototype for real-time defect detection on a small-scale production line.**

Built with **Raspberry Pi 5**, computer vision, capacitive sensing, and lightweight AI models — this system automatically detects **cap misalignment**, **missing/incorrect labels**, and **improper fill levels**. It logs defects, simulates PLC-based rejection, and triggers maintenance alerts when defect rates exceed thresholds.

---

## 🎥 Live Video Demo

<video width="100%" height="auto" controls autoplay loop muted playsinline>
  <source src="Video-Demo-Bottle.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>

> **The video above plays automatically** as you scroll (autoplay + loop + muted). It shows the complete real-time inspection workflow on the Raspberry Pi 5.

---

## ✨ Key Features

- **Real-time AI Inspection** using camera-based vision + optional deep learning model (`best.pt`)
- **Multi-Defect Detection**:
  - Cap misalignment
  - Missing or incorrect labels
  - Improper fill levels (via vision + capacitive sensing)
- **Automated Rejection** – simulated PLC-based reject mechanism
- **Defect Logging & Analytics** – records every failure with timestamps
- **Smart Alerts** – triggers maintenance notification when defect rate > threshold
- **Low-Cost & Scalable** – runs efficiently on Raspberry Pi 5
- **Industry 4.0 Ready** – supports smart manufacturing and sustainability goals

## 🚀 Why This Project Matters

Traditional manual inspection is **slow, error-prone, and inconsistent**.  
This system reduces waste, cuts operational costs, improves product quality, and supports **green manufacturing** by minimizing defective products that would otherwise be discarded.

---

## 📁 Project Files

| File                | Description |
|---------------------|-----------|
| **`detect3.py`**    | **Main detection script** – runs the full inspection pipeline (camera + AI + rejection logic) |
| **`best.pt`**       | Trained YOLO model (AI weights) – the brain of the vision system |
| **`quantization.py`** | Script to quantize the model for faster inference on Raspberry Pi 5 |
| `1-bottle.jpeg`     | Project showcase image |
| `Video-Demo-Bottle.mp4` | Full working demonstration video |

---

## 🛠️ How It Works (High-Level)

1. **Capture** – Raspberry Pi 5 camera streams live video of bottles on the conveyor.
2. **AI Analysis** – `detect3.py` runs inference with `best.pt` (or quantized version).
3. **Sensor Fusion** – Capacitive sensor confirms fill level.
4. **Decision** – System classifies bottle as **PASS** or **FAIL**.
5. **Action** – Simulated PLC rejects defective bottles + logs the event.
6. **Monitoring** – Dashboard/alert system activates if defect rate rises.

---

## 🧪 Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/YOUR-USERNAME/AI-Bottle-Inspection.git
cd AI-Bottle-Inspection

# 2. Install dependencies
pip install -r requirements.txt

# 3. (Optional) Quantize model for maximum speed on Pi 5
python quantization.py

# 4. Run the main inspection system
python detect3.py
