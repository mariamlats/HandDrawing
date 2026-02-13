# Hand Drawing → SVG → 3D

Webcam-based hand-tracking drawing application built with **OpenCV + MediaPipe**, featuring:

* Air drawing with your finger
* Smart stroke straightening & smoothing
* Pixel-based eraser (non-destructive)
* SVG vectorization
* Optional 3D model generation via Meshy API

# ✨ Features

## 🎨 Air Drawing

* Draw using your **index finger**
* Automatically stops drawing when gesture changes
* Adjustable brush size
* Multiple color palette (including brown)

## 🧽 Smart Pixel Eraser

* Erase using **index + middle finger pinch**
* Erasing does **not delete stroke geometry**
* You can repaint over erased areas

## 📐 Smart Straightening System

Each stroke is:

1. Smoothed using a moving average filter
2. Analyzed for linear intent using:

   * Stroke efficiency (length vs chord)
   * PCA linearity test
   * Bounding box aspect ratio
   * Turning angle constraint
   * Distance-to-line threshold
3. Snapped to a best-fit line **only if the stroke was likely intended to be straight**
4. Otherwise simplified using RDP (Ramer–Douglas–Peucker)

This preserves curves while correcting shaky straight lines.

## 🧩 Stroke Selection

* Toggle select mode
* Click near a stroke to select it
* Delete selected strokes

## 🖼 Export Options

* Save PNG
* Save SVG (via color quantization + contour extraction)

## 🧊 3D Generation (Optional)

* Sends PNG to Meshy Image-to-3D API
* Polls task progress
* Downloads `.glb` model
* Automatically opens the model locally

---

# 🎮 Controls

## ✋ Gestures

| Gesture              | Action   |
| -------------------- | -------- |
| Index finger up      | Draw     |
| Index + middle pinch | Erase    |
| Thumb + middle pinch | UI click |

---

## ⌨ Keyboard Shortcuts

| Key   | Action                   |
| ----- | ------------------------ |
| q     | Quit                     |
| m     | Toggle draw/select mode  |
| x     | Delete selected stroke   |
| s     | Save PNG                 |
| v     | Save SVG                 |
| 3     | Generate 3D              |
| t     | Toggle straighten ON/OFF |
| - / = | Brush size down/up       |
| [ / ] | Eraser size down/up      |
| c     | Clear everything         |
| r     | Reset erase mask         |

---

# 🧠 Technical Overview

## Stroke Representation

Each stroke is stored as:

```
{
  "color": (b,g,r),
  "thickness": int,
  "points": [(x,y), ...]
}
```

## Rendering Pipeline

1. Strokes rendered onto blank canvas
2. Current stroke preview rendered with highlight
3. Eraser mask applied as black holes
4. Canvas blended with camera feed

## Vectorization Pipeline

1. K-means color quantization
2. Per-color binary masks
3. Morphological closing
4. Contour extraction
5. Polygon simplification
6. SVG path generation

## Meshy API Flow

1. PNG converted to Base64 Data URI
2. POST request to Meshy API
3. Poll task status
4. Download GLB file
5. Auto-open locally

---

# 📦 Requirements

Python 3.9+

Install dependencies:

```
pip install opencv-python numpy mediapipe requests python-dotenv
```

---

# 🔐 Meshy Setup (Optional)

Create a `.env` file:

```
MESHY_API_KEY=YOUR_KEY_HERE
```

Without this key, 3D generation will not work.

---

# ▶ Run

```
python GUI_hand_drawing.py
```

Outputs are saved to:

```
outputs/
  drawing_<timestamp>.png
  vector_<timestamp>.svg
  meshy_<task_id>.glb
```
