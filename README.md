# 🏏 CRICKET VISION AI
### AI-Powered Batting Posture Analysis & Coaching Dashboard

[![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.40+-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![MediaPipe](https://img.shields.io/badge/Google%20MediaPipe-Tasks%20API-00897B?style=for-the-badge&logo=google&logoColor=white)](https://developers.google.com/mediapipe)
[![OpenCV](https://img.shields.io/badge/OpenCV-5.0+-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)](https://opencv.org/)

**Cricket Vision AI** is an intelligent, real-time sports technology application designed to help beginner and amateur batsmen evaluate and refine their cricket batting stance at the crease. Combining **Google MediaPipe's high-speed Pose Landmarker**, **NumPy geometric vector mathematics**, and a **rule-based cricket biomechanics engine**, the application provides live skeletal tracking, joint angle diagnostics, and actionable coaching adjustments directly in your web browser.

---

## 🌟 Key Features

- **Real-Time Live Webcam Analysis**: Low-latency video streaming powered by WebRTC with integrated Start/Stop controls.
- **33-Keypoint Skeletal Overlay**: High-contrast, glowing coaching skeleton tracking joints from head to toe.
- **Mathematical Joint Angle Engine**: Pure vector dot-product angle calculations for knees, elbows, hips, torso tilt, and stance width.
- **Rule-Based Cricket Biomechanics Engine**:
  - **Knee Flexion Diagnostics**: Identifies straight/locked knees vs athletic springy knee bend (140°–168°).
  - **Torso & Spine Alignment**: Measures athletic forward lean toward the bowler (5°–22°) vs excessive crouching or standing too upright.
  - **Base Stability & Stance Width**: Evaluates ankle-to-shoulder spread ratio for balanced weight distribution.
  - **Head & Eye Alignment**: Checks horizontal eye levelness for depth perception and ball tracking.
  - **Arm & Elbow Readiness**: Detects stiff/locked arms vs soft, relaxed elbows for an unhindered backlift.
- **100% Real-Time Live Webcam Engine**: Continuous WebRTC video stream with thread-safe auto-refreshing dashboard metrics.
- **Batsman Orientation Support**: Switch effortlessly between **Right-Hand Batsman (RHB)** and **Left-Hand Batsman (LHB)** stances.
- **Auto-Clearing Stale State**: Metrics automatically reset to "Waiting for Batsman..." when leaving the camera frame.
- **Professional Sports Tech Dashboard**: Dark charcoal UI with vibrant cricket-green accents and responsive metrics cards.

---

## 🏗️ System Architecture & Processing Pipeline

```
┌─────────────────┐       ┌───────────────────────┐       ┌────────────────────────┐
│  Webcam Stream  │ ───►  │ MediaPipe Pose Model  │ ───►  │  33 Landmark Keypoints │
│   (Live/Snap)   │       │  (PoseLandmarker)     │       │   (Normalized x, y, z) │
└─────────────────┘       └───────────────────────┘       └────────────────────────┘
                                                                       │
                                                                       ▼
┌─────────────────┐       ┌───────────────────────┐       ┌────────────────────────┐
│  Coaching UI &  │ ◄───  │ Cricket Biomechanics  │ ◄───  │  Vector Joint Angles   │
│ Streamlit HUD   │       │  Rule-Based Engine    │       │     (NumPy Arccos)     │
└─────────────────┘       └───────────────────────┘       └────────────────────────┘
```

---

## 📁 Project Structure

```
Cricket_Posture_Analyzer/
│
├── app.py                      # Main Streamlit web application & dashboard
├── pose_detector.py            # MediaPipe Tasks wrapper, skeleton overlay & angle math
├── posture_analyzer.py         # Rule-based cricket biomechanics & feedback engine
├── pose_landmarker_full.task   # Pre-trained MediaPipe Pose Landmarker model asset
├── pose_landmarker_lite.task   # Lightweight fallback MediaPipe model asset
├── requirements.txt            # Python dependencies
├── README.md                   # Documentation
│
└── assets/
    ├── sample_cricket_stance.jpg            # Pre-loaded cricket batsman demo photo
    └── sample_cricket_stance_annotated.jpg  # Annotated reference output
```

---

## 🚀 Quick Start & Installation

### 1. Prerequisites
- Python 3.10 to 3.12 installed on your system.
- Webcam (built-in laptop or USB webcam).

### 2. Install Dependencies
Open your terminal in the project directory and run:

```bash
pip install -r requirements.txt
```

### 3. Launch the Application
Start the Streamlit coaching dashboard:

```bash
streamlit run app.py
```

Streamlit will open the application in your browser at `http://localhost:8501`.

---

## 🎯 How to Demonstrate in an AI Seminar

When presenting **Cricket Vision AI** to an audience or seminar panel, follow this demonstration script:

1. **Architecture Introduction**:
   - Point to the **How It Works** tab: Explain the distinction between **MediaPipe Pose Estimation** (detects body landmarks) and the **Rule-Based Cricket Biomechanics Engine** (interprets angles through cricket coaching principles).
2. **Live Camera Demonstration**:
   - Click **START** in the web interface to initialize the webcam stream.
   - Stand back 2–3 meters so your full body is visible to your laptop webcam.
   - **Scenario A (Locked Knees)**: Stand straight and rigid. Point out how the knee metrics turn amber, the overall score drops to ~55, and the dashboard advises: *"Flex knees slightly (aim for ~145°–160°) to establish a springy, athletic base"*.
   - **Scenario B (Athletic Batting Stance)**: Assume a proper cricket stance with soft knees, slight forward hip hinge, and level eyes. The dashboard immediately updates to **85–95/100 (Balanced Athletic Stance)**.
   - **Scenario C (Player Exits Frame)**: Step out of view of the camera; observe the dashboard immediately clearing measurements and resetting to *"Waiting for Batsman..."*.
3. **Orientation Toggle**:
   - Toggle from **Right-Hand Batsman** to **Left-Hand Batsman** in the sidebar to show how the system automatically swaps the lead front leg and back leg metrics.

---

## 📐 Biomechanical Thresholds Reference

| Metric | Ideal Cricket Range | Defect / Warning Range | Biomechanical Rationale |
| :--- | :--- | :--- | :--- |
| **Knee Flexion** | `140° – 168°` | `> 170°` (Locked)<br>`< 125°` (Crouched) | Locked knees inhibit weight transfer; slight bend stores elastic energy for quick footwork. |
| **Torso Spine Tilt** | `6° – 22°` | `< 5°` (Too upright)<br>`> 25°` (Over-leaning) | Slight forward hip hinge aligns the center of gravity over the balls of the feet. |
| **Stance Width Ratio** | `0.90x – 1.50x` | `< 0.80x` (Narrow)<br>`> 1.65x` (Overly wide) | Stance width (ankles vs shoulders) dictates stability and forward/backward stride freedom. |
| **Head Horizontal Tilt**| `< 8.0°` | `> 8.0°` (Tilted) | Level eyes parallel to the pitch maintain binocular depth perception to judge ball trajectory. |
| **Elbow Readiness** | `95° – 150°` | `> 165°` (Rigid) | Soft elbows allow a fluent, relaxed backlift without muscular tension. |

---

## ⚠️ Limitations & Educational Disclaimer

- **Educational Prototype**: This application provides heuristic guidance based on standard 2D/3D computer vision keypoints. It is designed for educational exploration and personal awareness.
- **Variability of Technique**: True cricket batting technique varies significantly across individual styles (e.g., Steve Smith's open stance vs Shivnarine Chanderpaul's crab stance vs classical side-on stance). Heuristic angles should not replace personalized coaching.
- **Camera Angle & Perspective**: Accuracy is optimal when the camera is positioned at waist height, directly facing the batsman side-on or front-on from 2–3 meters away.
- **Lighting & Clothing**: Loose or baggy clothing can occasionally introduce slight landmark jitter. Ensure good lighting and a clear background for best results.

