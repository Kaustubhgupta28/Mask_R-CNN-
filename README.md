# 🎭 Mask R-CNN Instance Segmentation

> Detects & segments objects in images, videos and webcam feed with pixel-perfect masks using ResNet-50 FPN backbone pretrained on MS-COCO 2017.

🔗 **[Live Demo](https://mask-rcnn-detection.streamlit.app)** &nbsp;|&nbsp; ⭐ Star this repo if you find it useful!

---

![Python](https://img.shields.io/badge/Python-3.8+-blue?style=flat-square&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red?style=flat-square&logo=pytorch)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-ff4b4b?style=flat-square&logo=streamlit)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

---

## 📌 What is This?

A fully deployed web app that runs **Mask R-CNN** object detection and instance segmentation on any image or video. Unlike simple object detection that only draws bounding boxes, this app generates **pixel-level masks** for every individual object detected — each instance gets its own unique colored overlay.

Built end-to-end with Python, PyTorch, and Streamlit. No GPU required — runs entirely on CPU.

---

## ✨ Features

- 🖼️ **Image Detection** — Upload JPG, PNG, WEBP or paste a direct URL
- 🎬 **Video Detection** — Upload MP4/AVI/MOV, processes every frame and outputs annotated video
- 📷 **Webcam Support** — Take a photo or run live frame-by-frame detection (local mode)
- 🖼️ **Sample Images** — 6 built-in COCO dataset samples to try instantly
- 🎛️ **Real-time Controls** — Adjust confidence threshold, mask threshold, and transparency
- 👁️ **Toggle Overlays** — Show/hide masks, bounding boxes, and labels independently
- 📊 **Analytics Dashboard** — Class distribution chart, per-instance confidence scores, detection table
- ⬇️ **Download Results** — Save annotated images and videos directly

---

## 🧠 Model Details

| Property | Value |
|---|---|
| Architecture | Mask R-CNN |
| Backbone | ResNet-50 + FPN |
| Pretrained On | MS-COCO 2017 |
| Categories | 80 |
| Parameters | ~44M |
| Framework | PyTorch + Torchvision |

---

## 🗂️ Input Methods

| Method | Formats Supported |
|---|---|
| 📁 Upload Image | `.jpg` `.jpeg` `.png` `.webp` |
| 🎬 Upload Video | `.mp4` `.avi` `.mov` `.mkv` |
| 🔗 URL | Direct image or video URLs |
| 📷 Webcam | Browser camera / Live local feed |
| 🖼️ Sample Images | Built-in COCO dataset samples |

---

## 🛠️ Tech Stack

- **PyTorch + Torchvision** — Model inference
- **OpenCV** — Image/video processing, mask & box drawing
- **Streamlit** — Web UI and deployment
- **Matplotlib + Pandas** — Charts and analytics
- **Pillow** — Image handling
- **imageio + ffmpeg** — Video frame extraction and output

---

## 🚀 Run Locally

```bash
# 1. Clone the repo
git clone https://github.com/Kaustubhgupta28/Mask_R-CNN-.git
cd Mask_R-CNN-

# 2. Create virtual environment (recommended)
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app
streamlit run app.py
```

> **Note:** First run downloads model weights (~170MB). This only happens once and is cached automatically.

---

## 📁 Project Structure

```
Mask_R-CNN-/
├── app.py            # Main Streamlit app — UI, tabs, session state
├── model.py          # Model loading, inference, drawing logic
├── styles.css        # Custom CSS styling
├── requirements.txt  # Python dependencies
└── README.md
```

---

## 📸 80 COCO Object Categories

person, bicycle, car, motorcycle, airplane, bus, train, truck, boat, traffic light, fire hydrant, stop sign, bench, bird, cat, dog, horse, sheep, cow, elephant, bear, zebra, giraffe, backpack, umbrella, handbag, bottle, wine glass, cup, fork, knife, spoon, bowl, banana, apple, sandwich, orange, broccoli, pizza, donut, cake, chair, couch, bed, laptop, mouse, keyboard, cell phone, book, clock, vase, scissors, teddy bear and more...

---

## ⭐ Show Your Support

If you found this project helpful or interesting, please give it a ⭐ on GitHub — it really helps!

---

*Built with ❤️ using PyTorch and Streamlit*
