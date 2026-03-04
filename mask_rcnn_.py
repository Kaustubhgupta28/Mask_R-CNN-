# ============================================================
#   Mask R-CNN — Streamlit Web App
# ============================================================
import streamlit as st
import torch
import torchvision.transforms as T
from torchvision.models.detection import (
    maskrcnn_resnet50_fpn,
    MaskRCNN_ResNet50_FPN_Weights
)
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import requests
from io import BytesIO
import time
import os

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title = "Mask R-CNN Instance Segmentation",
    page_icon  = "🎭",
    layout     = "wide"
)

# ── COCO Labels ──────────────────────────────────────────────
COCO_CLASSES = {
    1:'person',      2:'bicycle',     3:'car',
    4:'motorcycle',  5:'airplane',    6:'bus',
    7:'train',       8:'truck',       9:'boat',
    10:'traffic light', 11:'fire hydrant', 13:'stop sign',
    14:'parking meter', 15:'bench',   16:'bird',
    17:'cat',        18:'dog',        19:'horse',
    20:'sheep',      21:'cow',        22:'elephant',
    23:'bear',       24:'zebra',      25:'giraffe',
    27:'backpack',   28:'umbrella',   31:'handbag',
    32:'tie',        33:'suitcase',   34:'frisbee',
    35:'skis',       36:'snowboard',  37:'sports ball',
    38:'kite',       39:'baseball bat', 40:'baseball glove',
    41:'skateboard', 42:'surfboard',  43:'tennis racket',
    44:'bottle',     46:'wine glass', 47:'cup',
    48:'fork',       49:'knife',      50:'spoon',
    51:'bowl',       52:'banana',     53:'apple',
    54:'sandwich',   55:'orange',     56:'broccoli',
    57:'carrot',     58:'hot dog',    59:'pizza',
    60:'donut',      61:'cake',       62:'chair',
    63:'couch',      64:'potted plant', 65:'bed',
    67:'dining table', 70:'toilet',   72:'tv',
    73:'laptop',     74:'mouse',      75:'remote',
    76:'keyboard',   77:'cell phone', 78:'microwave',
    79:'oven',       80:'toaster',    81:'sink',
    82:'refrigerator', 84:'book',     85:'clock',
    86:'vase',       87:'scissors',   88:'teddy bear',
    89:'hair drier', 90:'toothbrush'
}

COLORS = [
    (255, 99,  99),  (78, 205, 196),  (69, 183, 209),
    (150, 206, 180), (255, 234, 167), (221, 160, 221),
    (152, 216, 200), (241, 148, 138), (133, 193, 233),
    (240, 178, 122), (130, 224, 170), (255, 179, 102),
    (102, 255, 178), (255, 102, 178), (178, 102, 255),
]

# ── Load Model ───────────────────────────────────────────────
@st.cache_resource
def load_model():
    weights = MaskRCNN_ResNet50_FPN_Weights.DEFAULT
    model   = maskrcnn_resnet50_fpn(weights=weights)
    model.eval()
    return model

# ── Inference ────────────────────────────────────────────────
def run_inference(model, pil_image, score_threshold=0.5):
    transform = T.ToTensor()
    tensor    = transform(pil_image).unsqueeze(0)

    start = time.time()
    with torch.no_grad():
        predictions = model(tensor)
    elapsed = time.time() - start

    pred = predictions[0]
    keep = pred['scores'] >= score_threshold

    return {
        'boxes':          pred['boxes'][keep].cpu().numpy(),
        'labels':         pred['labels'][keep].cpu().numpy(),
        'scores':         pred['scores'][keep].cpu().numpy(),
        'masks':          pred['masks'][keep].cpu().numpy(),
        'inference_time': elapsed
    }

# ── Visualize ────────────────────────────────────────────────
def visualize_results(pil_image, results,
                      mask_threshold=0.5,
                      show_masks=True,
                      show_boxes=True,
                      show_labels=True):

    img_array = np.array(pil_image).copy()
    overlay   = img_array.copy()

    boxes  = results['boxes']
    labels = results['labels']
    scores = results['scores']
    masks  = results['masks']
    N      = len(boxes)

    for i in range(N):
        color        = COLORS[i % len(COLORS)]
        x1,y1,x2,y2 = boxes[i].astype(int)
        label_name   = COCO_CLASSES.get(int(labels[i]), 'unknown')
        score        = scores[i]

        if show_masks:
            binary  = (masks[i, 0] > mask_threshold)
            overlay[binary] = (
                overlay[binary] * 0.55 +
                np.array(color) * 0.45
            ).astype(np.uint8)
            mask_u8     = binary.astype(np.uint8) * 255
            contours, _ = cv2.findContours(
                mask_u8,
                cv2.RETR_TREE,
                cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(overlay, contours, -1, color, 2)

        if show_boxes:
            cv2.rectangle(overlay, (x1,y1), (x2,y2), color, 2)

        if show_labels:
            txt        = f'{label_name}: {score:.2f}'
            (tw,th), _ = cv2.getTextSize(
                txt, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            cv2.rectangle(overlay,
                (x1, y1-th-8),
                (x1+tw+6, y1), color, -1)
            cv2.putText(overlay, txt,
                (x1+3, y1-4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (0,0,0), 1, cv2.LINE_AA)

    return cv2.addWeighted(overlay, 0.75, img_array, 0.25, 0)


# ── UI ───────────────────────────────────────────────────────
st.title("🎭 Mask R-CNN Instance Segmentation")
st.markdown("Upload any image and detect objects with pixel-perfect masks!")

# Sidebar
st.sidebar.title("⚙️ Settings")
score_threshold = st.sidebar.slider(
    "Confidence Threshold", 0.1, 0.9, 0.5, 0.05)
show_masks  = st.sidebar.checkbox("Show Masks",  value=True)
show_boxes  = st.sidebar.checkbox("Show Boxes",  value=True)
show_labels = st.sidebar.checkbox("Show Labels", value=True)

st.sidebar.markdown("---")
st.sidebar.markdown("### 🧠 Model Info")
st.sidebar.markdown("- **Model**: Mask R-CNN")
st.sidebar.markdown("- **Backbone**: ResNet-50-FPN")
st.sidebar.markdown("- **Dataset**: MS-COCO (90 classes)")

# Load model
with st.spinner("⏳ Loading Mask R-CNN model..."):
    model = load_model()
st.success("✅ Model loaded!")

# Input options
input_type = st.radio(
    "Choose input type:",
    ["📁 Upload Image", "🌐 Image URL", "🖼️ Sample Image"]
)

pil_image = None

# Upload image
if input_type == "📁 Upload Image":
    uploaded = st.file_uploader(
        "Upload your image",
        type=['jpg', 'jpeg', 'png'])
    if uploaded:
        pil_image = Image.open(uploaded).convert('RGB')

# URL image
elif input_type == "🌐 Image URL":
    url = st.text_input("Paste image URL here:")
    if url:
        try:
            response  = requests.get(url, timeout=10)
            pil_image = Image.open(
                BytesIO(response.content)).convert('RGB')
        except:
            st.error("❌ Could not load image from URL")

# Sample image
elif input_type == "🖼️ Sample Image":
    sample = st.selectbox("Choose a sample:", [
        "People on street",
        "Animals",
        "Indoor scene"
    ])
    urls = {
        "People on street": "http://images.cocodataset.org/val2017/000000397133.jpg",
        "Animals":          "http://images.cocodataset.org/val2017/000000039769.jpg",
        "Indoor scene":     "http://images.cocodataset.org/val2017/000000037777.jpg",
    }
    try:
        response  = requests.get(urls[sample], timeout=10)
        pil_image = Image.open(
            BytesIO(response.content)).convert('RGB')
    except:
        st.error("❌ Could not load sample image")

# Run inference
if pil_image is not None:
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📸 Original Image")
        st.image(pil_image, use_column_width=True)

    with st.spinner("🔍 Running Mask R-CNN..."):
        results = run_inference(model, pil_image, score_threshold)
        output  = visualize_results(
            pil_image, results,
            show_masks  = show_masks,
            show_boxes  = show_boxes,
            show_labels = show_labels
        )

    with col2:
        st.subheader(f"🎭 Result — {len(results['boxes'])} objects")
        st.image(output, use_column_width=True)

    # Stats
    st.markdown("---")
    st.subheader("📊 Detection Results")

    col3, col4, col5 = st.columns(3)
    col3.metric("Total Objects",    len(results['boxes']))
    col4.metric("Inference Time",   f"{results['inference_time']*1000:.0f} ms")
    col5.metric("Threshold Used",   score_threshold)

    # Detection table
    if len(results['boxes']) > 0:
        st.markdown("### 🏷️ Detected Objects")
        for i, (label, score, box) in enumerate(zip(
                results['labels'],
                results['scores'],
                results['boxes'])):
            name = COCO_CLASSES.get(int(label), 'unknown')
            x1,y1,x2,y2 = box.astype(int)
            st.markdown(
                f"`[{i+1:02d}]` **{name}** — "
                f"score: `{score:.3f}` — "
                f"box: `({x1},{y1}) to ({x2},{y2})`"
            )

    # Download button
    output_pil = Image.fromarray(output)
    from io import BytesIO as BIO
    buf = BIO()
    output_pil.save(buf, format='PNG')
    st.download_button(
        label     = "💾 Download Result Image",
        data      = buf.getvalue(),
        file_name = "mask_rcnn_output.png",
        mime      = "image/png"
    )
```

---

**Update `requirements.txt` on GitHub:**
```
streamlit
torch
torchvision
opencv-python
matplotlib
Pillow
numpy
requests
```

---

## ✅ Steps to Deploy
```
1. Go to your GitHub repo
2. Click "Add file" → "Create new file"
3. Name it: app.py
4. Paste the code above
5. Click "Commit changes"
6. Update requirements.txt with new content
7. Go to: https://streamlit.io/cloud
8. Click "New app"
9. Connect your GitHub repo
10. Set main file: app.py
11. Click "Deploy"
12. Done! ✅
