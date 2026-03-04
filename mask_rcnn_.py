# ============================================================
#   Mask R-CNN — Complete Instance Segmentation Project
#   Run: python mask_rcnn.py
# ============================================================

# ── 1. Install packages ──────────────────────────────────────
import subprocess
import sys

def install(package):
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])

packages = [
    'torch==2.1.0',
    'torchvision==0.16.0',
    'opencv-python',
    'matplotlib',
    'Pillow',
    'numpy',
    'requests',
    'moviepy==1.0.3'
]

print("📦 Installing packages...")
for p in packages:
    install(p)
print("✅ All packages installed!\n")


# ── 2. Import libraries ──────────────────────────────────────
import torch
import torchvision
import torchvision.transforms as T
from torchvision.models.detection import (
    maskrcnn_resnet50_fpn,
    MaskRCNN_ResNet50_FPN_Weights
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn   import MaskRCNNPredictor

import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import requests
from io   import BytesIO
import os
import time
import warnings
warnings.filterwarnings('ignore')

# ── Device ───────────────────────────────────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🖥️  Device: {device}\n")


# ── 3. COCO Labels ───────────────────────────────────────────
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

# ── Color palette ─────────────────────────────────────────────
COLORS = [
    (255, 99,  99),  (78, 205, 196),  (69, 183, 209),
    (150, 206, 180), (255, 234, 167), (221, 160, 221),
    (152, 216, 200), (241, 148, 138), (133, 193, 233),
    (240, 178, 122), (130, 224, 170), (255, 179, 102),
    (102, 255, 178), (255, 102, 178), (178, 102, 255),
]
print("✅ COCO labels loaded!\n")


# ── 4. Load Model ────────────────────────────────────────────
def load_model():
    print("⬇️  Loading pretrained Mask R-CNN...")
    weights = MaskRCNN_ResNet50_FPN_Weights.DEFAULT
    model   = maskrcnn_resnet50_fpn(weights=weights)
    model.to(device)
    model.eval()
    print(f"✅ Model loaded!")
    print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}\n")
    return model

model = load_model()


# ── 5. Image Loading ─────────────────────────────────────────
def load_image_from_file(path):
    img = Image.open(path).convert('RGB')
    print(f"✅ Image loaded: {img.size}")
    return img

def load_image_from_url(url):
    response = requests.get(url, timeout=10)
    img      = Image.open(BytesIO(response.content)).convert('RGB')
    print(f"✅ Image loaded from URL: {img.size}")
    return img

def image_to_tensor(pil_image):
    return T.ToTensor()(pil_image).to(device)


# ── 6. Run Inference ─────────────────────────────────────────
def run_inference(model, pil_image, score_threshold=0.5):
    model.eval()
    tensor = image_to_tensor(pil_image)

    start = time.time()
    with torch.no_grad():
        predictions = model([tensor])
    elapsed = time.time() - start

    pred = predictions[0]
    keep = pred['scores'] >= score_threshold

    result = {
        'boxes':          pred['boxes'][keep].cpu().numpy(),
        'labels':         pred['labels'][keep].cpu().numpy(),
        'scores':         pred['scores'][keep].cpu().numpy(),
        'masks':          pred['masks'][keep].cpu().numpy(),
        'inference_time': elapsed
    }

    print(f"\n⚡ Inference time : {elapsed*1000:.1f} ms")
    print(f"🎯 Detections     : {len(result['boxes'])} objects")
    for i, (label, score) in enumerate(
            zip(result['labels'], result['scores'])):
        name = COCO_CLASSES.get(int(label), f'unknown({label})')
        print(f"   [{i+1:02d}] {name:<20s}  score={score:.3f}")

    return result


# ── 7. Visualize Results ─────────────────────────────────────
def visualize_results(pil_image, results,
                      mask_threshold=0.5,
                      show_masks=True,
                      show_boxes=True,
                      show_labels=True,
                      save_path='output_segmented.png'):

    img_array = np.array(pil_image).copy()
    overlay   = img_array.copy()

    boxes  = results['boxes']
    labels = results['labels']
    scores = results['scores']
    masks  = results['masks']
    N      = len(boxes)

    if N == 0:
        print("⚠️  No detections!")
        return

    for i in range(N):
        color      = COLORS[i % len(COLORS)]
        x1,y1,x2,y2 = boxes[i].astype(int)
        label_name = COCO_CLASSES.get(int(labels[i]), 'unknown')
        score      = scores[i]

        # Mask
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

        # Box
        if show_boxes:
            cv2.rectangle(overlay, (x1,y1), (x2,y2), color, 2)

        # Label
        if show_labels:
            txt        = f'{label_name}: {score:.2f}'
            (tw,th), _ = cv2.getTextSize(
                txt, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            cv2.rectangle(overlay,
                (x1, y1-th-8), (x1+tw+6, y1), color, -1)
            cv2.putText(overlay, txt,
                (x1+3, y1-4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (0,0,0), 1, cv2.LINE_AA)

    final = cv2.addWeighted(overlay, 0.75, img_array, 0.25, 0)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    axes[0].imshow(img_array)
    axes[0].set_title('Original Image', fontweight='bold')
    axes[0].axis('off')

    axes[1].imshow(final)
    axes[1].set_title(
        f'Mask R-CNN Output ({N} instances)',
        fontweight='bold')
    axes[1].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"\n💾 Saved: {save_path}")
    return final


# ── 8. Process Video ─────────────────────────────────────────
def process_video(video_path, output_path='output_video.mp4',
                  score_threshold=0.5, max_frames=None):

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f'Cannot open: {video_path}')

    fps    = int(cap.get(cv2.CAP_PROP_FPS))
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"\n🎬 Video: {width}×{height} @ {fps}fps ({total} frames)")

    writer = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps, (width, height)
    )

    frame_idx = 0
    model.eval()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        if max_frames and frame_idx >= max_frames: break

        rgb       = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_frame = Image.fromarray(rgb)
        results   = run_inference(model, pil_frame, score_threshold)

        annotated = np.array(pil_frame).copy()
        for i in range(len(results['boxes'])):
            color       = COLORS[i % len(COLORS)]
            x1,y1,x2,y2 = results['boxes'][i].astype(int)
            binary      = (results['masks'][i, 0] > 0.5)
            annotated[binary] = (
                annotated[binary] * 0.6 +
                np.array(color)   * 0.4
            ).astype(np.uint8)
            cv2.rectangle(annotated, (x1,y1), (x2,y2), color, 2)
            name = COCO_CLASSES.get(int(results['labels'][i]), 'unknown')
            txt  = f"{name}: {results['scores'][i]:.2f}"
            cv2.putText(annotated, txt, (x1, y1-5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        writer.write(cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))

        if frame_idx % 10 == 0:
            print(f"   Frame {frame_idx}/{total}")
        frame_idx += 1

    cap.release()
    writer.release()
    print(f"✅ Video saved: {output_path}")


# ── 9. Run on Image ──────────────────────────────────────────
print("\n" + "="*50)
print("  RUNNING ON IMAGE")
print("="*50)

# ── Option A: Local file ──────────────────────────
# my_image = load_image_from_file(r"C:\Users\Kaustubh Gupta\Downloads\your_image.jpg")

# ── Option B: URL ─────────────────────────────────
my_image = load_image_from_url(
    'http://images.cocodataset.org/val2017/000000039769.jpg')

my_results = run_inference(model, my_image, score_threshold=0.5)
visualize_results(my_image, my_results, save_path='output_segmented.png')


# ── 10. Run on Video ─────────────────────────────────────────
print("\n" + "="*50)
print("  RUNNING ON VIDEO")
print("="*50)

# ── Option A: Local video file ────────────────────
# process_video(
#     r"C:\Users\Kaustubh Gupta\Videos\your_video.mp4",
#     output_path    = 'output_video.mp4',
#     score_threshold = 0.5,
#     max_frames     = 50
# )

# ── Option B: Use sample video ────────────────────
print("📌 Uncomment process_video() above and add your video path to run!")


print("\n" + "="*50)
print("  ✅ PROJECT COMPLETE!")
print("="*50)
print("Output files saved:")
print("   📸 output_segmented.png")
print("   🎬 output_video.mp4 (if video was run)")
