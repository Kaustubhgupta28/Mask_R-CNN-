# ═══════════════════════════════════════════════════════════════
#  model.py — Sirf AI / ML Logic
#  Isme hai: Model load, Inference, Drawing, Constants
# ═══════════════════════════════════════════════════════════════

import torch
import torchvision.transforms as T
from torchvision.models.detection import maskrcnn_resnet50_fpn, MaskRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import cv2
import numpy as np
from PIL import Image
import time
import streamlit as st

# ─────────────────────────────────────────────────────────────
# DEVICE SETUP
# ─────────────────────────────────────────────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ─────────────────────────────────────────────────────────────
# COCO CLASS LABELS — 80 categories
# ─────────────────────────────────────────────────────────────
COCO_CLASSES = {
    1:'person',       2:'bicycle',       3:'car',           4:'motorcycle',
    5:'airplane',     6:'bus',           7:'train',         8:'truck',
    9:'boat',        10:'traffic light', 11:'fire hydrant', 13:'stop sign',
   14:'parking meter',15:'bench',       16:'bird',         17:'cat',
   18:'dog',         19:'horse',        20:'sheep',        21:'cow',
   22:'elephant',    23:'bear',         24:'zebra',        25:'giraffe',
   27:'backpack',    28:'umbrella',     31:'handbag',      32:'tie',
   33:'suitcase',    34:'frisbee',      35:'skis',         36:'snowboard',
   37:'sports ball', 38:'kite',         39:'baseball bat', 40:'baseball glove',
   41:'skateboard',  42:'surfboard',    43:'tennis racket',44:'bottle',
   46:'wine glass',  47:'cup',          48:'fork',         49:'knife',
   50:'spoon',       51:'bowl',         52:'banana',       53:'apple',
   54:'sandwich',    55:'orange',       56:'broccoli',     57:'carrot',
   58:'hot dog',     59:'pizza',        60:'donut',        61:'cake',
   62:'chair',       63:'couch',        64:'potted plant', 65:'bed',
   67:'dining table',70:'toilet',       72:'tv',           73:'laptop',
   74:'mouse',       75:'remote',       76:'keyboard',     77:'cell phone',
   78:'microwave',   79:'oven',         80:'toaster',      81:'sink',
   82:'refrigerator',84:'book',         85:'clock',        86:'vase',
   87:'scissors',    88:'teddy bear',   89:'hair drier',   90:'toothbrush',
}

# ─────────────────────────────────────────────────────────────
# COLOR PALETTE — har instance ke liye alag color
# ─────────────────────────────────────────────────────────────
COLORS = [
    (255, 99,  99),  (78,  205, 196), (69,  183, 209), (150, 206, 180),
    (255, 178, 102), (221, 160, 221), (152, 216, 200), (241, 148, 138),
    (133, 193, 233), (240, 178, 122), (130, 224, 170), (255, 102, 178),
    (178, 102, 255), (102, 178, 255), (255, 255, 102), (102, 255, 255),
    (200, 150, 255), (255, 200, 100), (100, 255, 200), (255, 100, 200),
]

# ─────────────────────────────────────────────────────────────
# SAMPLE IMAGES
# ─────────────────────────────────────────────────────────────
SAMPLE_IMAGES = {
    "🐱 Cats on a couch":  "http://images.cocodataset.org/val2017/000000039769.jpg",
    "🚗 Street scene":     "http://images.cocodataset.org/val2017/000000397133.jpg",
    "👥 People & objects": "http://images.cocodataset.org/val2017/000000037777.jpg",
    "🍕 Kitchen scene":    "http://images.cocodataset.org/val2017/000000252219.jpg",
    "🌳 Outdoor scene":    "http://images.cocodataset.org/val2017/000000087038.jpg",
    "🏀 Sports scene":     "http://images.cocodataset.org/val2017/000000174482.jpg",
}

# ─────────────────────────────────────────────────────────────
# LOAD MODEL — COCO pretrained Mask R-CNN
# ─────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model():
    """
    Pretrained Mask R-CNN model load karo.
    ResNet-50 FPN backbone, MS-COCO 2017 weights.
    @st.cache_resource — sirf ek baar load hoga.
    """
    weights = MaskRCNN_ResNet50_FPN_Weights.DEFAULT
    m = maskrcnn_resnet50_fpn(weights=weights)
    m.to(device)
    m.eval()
    return m

# ─────────────────────────────────────────────────────────────
# RUN INFERENCE — image pe detection chalao
# ─────────────────────────────────────────────────────────────
def run_inference(model, pil_image, score_threshold=0.5):
    """
    Mask R-CNN se objects detect karo.

    Args:
        model          : load_model() se mila model
        pil_image      : PIL Image (RGB)
        score_threshold: Minimum confidence (0.0 - 1.0)

    Returns dict with:
        boxes  — bounding box coordinates
        labels — class IDs
        scores — confidence scores
        masks  — instance segmentation masks
        time   — inference time in seconds
    """
    tensor = T.ToTensor()(pil_image).to(device)

    t0 = time.time()
    with torch.no_grad():
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

# ─────────────────────────────────────────────────────────────
# DRAW RESULTS — image pe masks, boxes, labels draw karo
# ─────────────────────────────────────────────────────────────
def draw_results(pil_image, results,
                 mask_thr=0.5, show_masks=True,
                 show_boxes=True, show_labels=True, alpha=0.45):
    """
    Detection results image pe draw karo.

    Args:
        pil_image  : original PIL Image
        results    : run_inference() ka output
        mask_thr   : mask binary threshold
        show_masks : colored mask dikhao
        show_boxes : bounding box dikhao
        show_labels: class + score label dikhao
        alpha      : mask transparency

    Returns:
        final  : annotated numpy image (RGB)
        N      : number of detected objects
    """
    img  = np.array(pil_image).copy()
    over = img.copy()
    N    = len(results['boxes'])

    for i in range(N):
        color        = COLORS[i % len(COLORS)]
        x1,y1,x2,y2 = results['boxes'][i].astype(int)
        lname        = COCO_CLASSES.get(int(results['labels'][i]), '?')
        score        = results['scores'][i]

        # ── Segmentation Mask ──────────────────────────────
        if show_masks:
            bm = (results['masks'][i, 0] > mask_thr)
            over[bm] = (over[bm] * 0.55 + np.array(color) * 0.45).astype(np.uint8)
            mu8 = bm.astype(np.uint8) * 255
            ctrs, _ = cv2.findContours(mu8, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(over, ctrs, -1, color, 2)

        # ── Bounding Box ───────────────────────────────────
        if show_boxes:
            cv2.rectangle(over, (x1, y1), (x2, y2), color, 2)

        # ── Label Chip ─────────────────────────────────────
        if show_labels:
            txt = f'{lname}: {score:.2f}'
            (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            cv2.rectangle(over, (x1, y1-th-8), (x1+tw+6, y1), color, -1)
            cv2.putText(over, txt, (x1+3, y1-4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)

    # Blend overlay with original
    final = cv2.addWeighted(over, alpha+0.3, img, 1-(alpha+0.3), 0)
    return final, N

# ─────────────────────────────────────────────────────────────
# LOCAL CHECK — webcam available hai ya nahi
# ─────────────────────────────────────────────────────────────
def is_local():
    """
    Check karo local computer pe chal raha hai ya Streamlit Cloud pe.
    Webcam try karke detect karta hai.
    """
    import os
    try:
        if os.environ.get("STREAMLIT_SHARING_MODE"):
            return False
        if os.environ.get("IS_CLOUD"):
            return False
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            cap.release()
            return True
        cap.release()
        return False
    except:
        return False
