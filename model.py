# ═══════════════════════════════════════════════════════════════
#  model.py — AI / ML Logic
#  Mask R-CNN: Model load, Inference, Drawing, Constants
# ═══════════════════════════════════════════════════════════════

import os
import torch
import torchvision.transforms as T
from torchvision.models.detection import maskrcnn_resnet50_fpn, MaskRCNN_ResNet50_FPN_Weights
import cv2
import numpy as np
from PIL import Image
import time
import streamlit as st

# Suppress OpenCV camera warnings on cloud
os.environ["OPENCV_LOG_LEVEL"] = "SILENT"

# ── Device ────────────────────────────────────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ── COCO Class Labels — 80 categories ────────────────────────
COCO_CLASSES = {
    1:'person',        2:'bicycle',        3:'car',            4:'motorcycle',
    5:'airplane',      6:'bus',            7:'train',          8:'truck',
    9:'boat',         10:'traffic light', 11:'fire hydrant',  13:'stop sign',
   14:'parking meter',15:'bench',         16:'bird',          17:'cat',
   18:'dog',          19:'horse',         20:'sheep',         21:'cow',
   22:'elephant',     23:'bear',          24:'zebra',         25:'giraffe',
   27:'backpack',     28:'umbrella',      31:'handbag',       32:'tie',
   33:'suitcase',     34:'frisbee',       35:'skis',          36:'snowboard',
   37:'sports ball',  38:'kite',          39:'baseball bat',  40:'baseball glove',
   41:'skateboard',   42:'surfboard',     43:'tennis racket', 44:'bottle',
   46:'wine glass',   47:'cup',           48:'fork',          49:'knife',
   50:'spoon',        51:'bowl',          52:'banana',        53:'apple',
   54:'sandwich',     55:'orange',        56:'broccoli',      57:'carrot',
   58:'hot dog',      59:'pizza',         60:'donut',         61:'cake',
   62:'chair',        63:'couch',         64:'potted plant',  65:'bed',
   67:'dining table', 70:'toilet',        72:'tv',            73:'laptop',
   74:'mouse',        75:'remote',        76:'keyboard',      77:'cell phone',
   78:'microwave',    79:'oven',          80:'toaster',       81:'sink',
   82:'refrigerator', 84:'book',          85:'clock',         86:'vase',
   87:'scissors',     88:'teddy bear',    89:'hair drier',    90:'toothbrush',
}

# ── Color Palette ─────────────────────────────────────────────
COLORS = [
    (255,  99,  99), ( 78, 205, 196), ( 69, 183, 209), (150, 206, 180),
    (255, 178, 102), (221, 160, 221), (152, 216, 200), (241, 148, 138),
    (133, 193, 233), (240, 178, 122), (130, 224, 170), (255, 102, 178),
    (178, 102, 255), (102, 178, 255), (255, 255, 102), (102, 255, 255),
    (200, 150, 255), (255, 200, 100), (100, 255, 200), (255, 100, 200),
]

# ── Sample Images ─────────────────────────────────────────────
SAMPLE_IMAGES = {
    "🐱 Cats on a couch":  "http://images.cocodataset.org/val2017/000000039769.jpg",
    "🚗 Street scene":     "http://images.cocodataset.org/val2017/000000397133.jpg",
    "👥 People & objects": "http://images.cocodataset.org/val2017/000000037777.jpg",
    "🍕 Kitchen scene":    "http://images.cocodataset.org/val2017/000000252219.jpg",
    "🌳 Outdoor scene":    "http://images.cocodataset.org/val2017/000000087038.jpg",
    "🏀 Sports scene":     "http://images.cocodataset.org/val2017/000000174482.jpg",
}

# ── Load Model ────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model():
    """
    Load pretrained Mask R-CNN (ResNet-50 FPN, MS-COCO 2017).
    Cached — loads only once per session.
    """
    weights = MaskRCNN_ResNet50_FPN_Weights.DEFAULT
    m = maskrcnn_resnet50_fpn(weights=weights)
    m.to(device)
    m.eval()
    return m

# ── Run Inference ─────────────────────────────────────────────
def run_inference(model, pil_image, score_threshold=0.5):
    """
    Run Mask R-CNN detection on a PIL image.
    Hard limit 512px to prevent cloud timeout/OOM.

    Args:
        model           : loaded model from load_model()
        pil_image       : PIL Image (RGB)
        score_threshold : minimum confidence score (0.0 - 1.0)

    Returns dict:
        boxes  — bounding box coordinates [x1,y1,x2,y2]
        labels — class IDs
        scores — confidence scores
        masks  — instance segmentation masks
        time   — inference time in seconds
    """
    # Hard resize limit inside inference — extra safety layer
    w, h = pil_image.size
    if max(w, h) > 512:
        scale = 512 / max(w, h)
        pil_image = pil_image.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

    tensor = T.ToTensor()(pil_image).to(device)
    t0 = time.time()
    with torch.inference_mode():   # faster than torch.no_grad()
        preds = model([tensor])
    elapsed = time.time() - t0

    pred = preds[0]
    keep = pred['scores'] >= score_threshold

    return {
        'boxes':  pred['boxes'][keep].cpu().numpy(),
        'labels': pred['labels'][keep].cpu().numpy(),
        'scores': pred['scores'][keep].cpu().numpy(),
        'masks':  pred['masks'][keep].cpu().numpy(),
        'time':   elapsed,
    }

# ── Draw Results ──────────────────────────────────────────────
def draw_results(pil_image, results,
                 mask_thr=0.5, show_masks=True,
                 show_boxes=True, show_labels=True, alpha=0.45):
    """
    Draw detection results (masks, boxes, labels) on image.

    Args:
        pil_image   : original PIL Image
        results     : output from run_inference()
        mask_thr    : binary threshold for mask
        show_masks  : draw colored instance masks
        show_boxes  : draw bounding boxes
        show_labels : draw class name + confidence label
        alpha       : mask overlay transparency

    Returns:
        final : annotated numpy image (RGB)
        N     : number of detected objects
    """
    img  = np.array(pil_image).copy()
    over = img.copy()
    N    = len(results['boxes'])

    for i in range(N):
        color        = COLORS[i % len(COLORS)]
        x1,y1,x2,y2 = results['boxes'][i].astype(int)
        lname        = COCO_CLASSES.get(int(results['labels'][i]), '?')
        score        = results['scores'][i]

        # Segmentation Mask
        if show_masks:
            bm = (results['masks'][i, 0] > mask_thr)
            over[bm] = (over[bm] * 0.55 + np.array(color) * 0.45).astype(np.uint8)
            mu8 = bm.astype(np.uint8) * 255
            ctrs, _ = cv2.findContours(mu8, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(over, ctrs, -1, color, 2)

        # Bounding Box
        if show_boxes:
            cv2.rectangle(over, (x1, y1), (x2, y2), color, 2)

        # Label Chip
        if show_labels:
            txt = f'{lname}: {score:.2f}'
            (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            cv2.rectangle(over, (x1, y1-th-8), (x1+tw+6, y1), color, -1)
            cv2.putText(over, txt, (x1+3, y1-4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)

    final = cv2.addWeighted(over, alpha+0.3, img, 1-(alpha+0.3), 0)
    return final, N

# ── Local Check ───────────────────────────────────────────────
def is_local():
    """
    Detect if running locally or on Streamlit Cloud.
    Uses environment variables only — no webcam probing.
    """
    if os.environ.get("HOME") == "/home/adminuser":
        return False
    if os.environ.get("STREAMLIT_SHARING_MODE"):
        return False
    if os.environ.get("IS_CLOUD"):
        return False
    return True
