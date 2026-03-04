#!/usr/bin/env python
# coding: utf-8

# # 🎭 Mask R-CNN — End-to-End Instance Segmentation
# ### Complete Jupyter Notebook: Setup → Load Model → Inference → Visualize → Evaluate
# ---

# ## 📦 Cell 1 — Install Dependencies

# In[15]:


get_ipython().system('pip uninstall torch torchvision torchaudio -y')


# In[16]:


get_ipython().system('pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cpu')


# In[1]:


import torch
print("PyTorch:", torch.__version__)


# In[2]:


get_ipython().system('pip install "numpy<2" --force-reinstall')


# In[1]:


import numpy
print("NumPy:", numpy.__version__)


# In[1]:


# Run this cell first — installs all required packages
get_ipython().system('pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118')
get_ipython().system('pip install opencv-python matplotlib Pillow numpy requests pycocotools')
print('✅ All packages installed!')


# ## 📚 Cell 2 — Import Libraries

# In[2]:


import torch
import torchvision
import numpy
import cv2
import matplotlib.pyplot as plt
from PIL import Image

print("✅ NumPy:", numpy.__version__)
print("✅ PyTorch:", torch.__version__)
print("✅ Torchvision:", torchvision.__version__)
print("✅ OpenCV:", cv2.__version__)
print("✅ All libraries working! Ready to run the project.")


# In[3]:


import torch
import torchvision
import torchvision.transforms as T
from torchvision.models.detection import maskrcnn_resnet50_fpn, MaskRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import to_rgb
from PIL import Image
import requests
from io import BytesIO
import os
import random
import time
import warnings
warnings.filterwarnings('ignore')

# ── Device setup ──────────────────────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'🖥️  Using device: {device}')
if torch.cuda.is_available():
    print(f'   GPU: {torch.cuda.get_device_name(0)}')
    print(f'   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')


# ## 🏷️ Cell 3 — COCO Class Labels

# In[4]:


# 90 COCO object categories (index 0 = background)
COCO_CLASSES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
    'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
    'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
    'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
    'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
    'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass',
    'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
    'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
    'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
    'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
    'hair drier', 'toothbrush'
]

# Color palette for instance masks (one color per instance)
def get_color_palette(n=20):
    colors = [
        (255, 99,  99),  (78, 205, 196),  (69, 183, 209),  (150, 206, 180),
        (255, 234, 167), (221, 160, 221), (152, 216, 200), (241, 148, 138),
        (133, 193, 233), (240, 178, 122), (130, 224, 170), (241, 148, 138),
        (255, 179, 102), (102, 255, 178), (255, 102, 178), (178, 102, 255),
        (102, 178, 255), (255, 255, 102), (102, 255, 255), (255, 102, 102),
    ]
    return [colors[i % len(colors)] for i in range(n)]

COLORS = get_color_palette(50)
print(f'✅ Loaded {len(COCO_CLASSES)-1} COCO classes')
print(f'   Sample classes: {COCO_CLASSES[1:8]}')


# ## 🧠 Cell 4 — Load Pretrained Mask R-CNN Model

# In[5]:


def load_pretrained_model():
    """
    Load Mask R-CNN with ResNet-50-FPN backbone,
    pretrained on MS-COCO 2017 (91 classes).
    Ready for inference out of the box.
    """
    print('⬇️  Loading pretrained Mask R-CNN (ResNet-50-FPN)...')
    weights = MaskRCNN_ResNet50_FPN_Weights.DEFAULT
    model = maskrcnn_resnet50_fpn(weights=weights)
    model.to(device)
    model.eval()  # Set to evaluation mode
    print('✅ Model loaded successfully!')
    print(f'   Parameters: {sum(p.numel() for p in model.parameters()):,}')
    return model


def load_custom_model(num_classes, checkpoint_path=None):
    """
    Load Mask R-CNN for fine-tuning on a custom dataset.
    Replaces the classification and mask heads.

    Args:
        num_classes (int): number of YOUR classes + 1 (background)
        checkpoint_path (str): path to saved .pth weights (optional)
    """
    # Start from COCO-pretrained backbone
    model = maskrcnn_resnet50_fpn(weights=MaskRCNN_ResNet50_FPN_Weights.DEFAULT)

    # Replace box head
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # Replace mask head
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_features_mask, hidden_layer=256, num_classes=num_classes
    )

    if checkpoint_path and os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print(f'✅ Loaded weights from {checkpoint_path}')

    model.to(device)
    return model


# Load pretrained model (91 COCO classes)
model = load_pretrained_model()


# ## 🖼️ Cell 5 — Image Loading Utilities (URL, File, Webcam)

# In[6]:


def load_image_from_url(url):
    """Load image from a URL."""
    response = requests.get(url, timeout=10)
    img = Image.open(BytesIO(response.content)).convert('RGB')
    return img


def load_image_from_file(path):
    """Load image from local file path."""
    img = Image.open(path).convert('RGB')
    return img


def load_image_from_array(numpy_bgr):
    """Convert OpenCV BGR array to PIL RGB image."""
    rgb = cv2.cvtColor(numpy_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def image_to_tensor(pil_image):
    """Convert PIL image to normalized tensor for Mask R-CNN."""
    transform = T.ToTensor()          # Converts to [0,1] float tensor
    tensor = transform(pil_image)     # Shape: [3, H, W]
    return tensor.to(device)


def show_image(pil_image, title='Input Image'):
    """Display PIL image in notebook."""
    plt.figure(figsize=(10, 7))
    plt.imshow(pil_image)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.axis('off')
    plt.tight_layout()
    plt.show()


# ── Test: Load a sample crowded street scene from the web ──
SAMPLE_URL = 'http://images.cocodataset.org/val2017/000000039769.jpg'
print('📥 Loading sample image from COCO dataset...')
sample_image = load_image_from_url(SAMPLE_URL)
print(f'   Image size: {sample_image.size} (W × H)')
show_image(sample_image, 'Sample Input Image (COCO val2017)')


# ## 🔮 Cell 6 — Run Inference

# In[19]:


COCO_CLASSES = {
    1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorcycle',
    5: 'airplane', 6: 'bus', 7: 'train', 8: 'truck', 9: 'boat',
    10: 'traffic light', 11: 'fire hydrant', 13: 'stop sign',
    14: 'parking meter', 15: 'bench', 16: 'bird', 17: 'cat',
    18: 'dog', 19: 'horse', 20: 'sheep', 21: 'cow', 22: 'elephant',
    23: 'bear', 24: 'zebra', 25: 'giraffe', 27: 'backpack',
    28: 'umbrella', 31: 'handbag', 32: 'tie', 33: 'suitcase',
    34: 'frisbee', 35: 'skis', 36: 'snowboard', 37: 'sports ball',
    38: 'kite', 39: 'baseball bat', 40: 'baseball glove',
    41: 'skateboard', 42: 'surfboard', 43: 'tennis racket',
    44: 'bottle', 46: 'wine glass', 47: 'cup', 48: 'fork',
    49: 'knife', 50: 'spoon', 51: 'bowl', 52: 'banana',
    53: 'apple', 54: 'sandwich', 55: 'orange', 56: 'broccoli',
    57: 'carrot', 58: 'hot dog', 59: 'pizza', 60: 'donut',
    61: 'cake', 62: 'chair', 63: 'couch', 64: 'potted plant',
    65: 'bed', 67: 'dining table', 70: 'toilet', 72: 'tv',
    73: 'laptop', 74: 'mouse', 75: 'remote', 76: 'keyboard',
    77: 'cell phone', 78: 'microwave', 79: 'oven', 80: 'toaster',
    81: 'sink', 82: 'refrigerator', 84: 'book', 85: 'clock',
    86: 'vase', 87: 'scissors', 88: 'teddy bear', 89: 'hair drier',
    90: 'toothbrush'
}
print("✅ COCO_CLASSES ready!")


# In[20]:


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

    print(f'⚡ Inference time : {elapsed*1000:.1f} ms')
    print(f'🎯 Detections     : {len(result["boxes"])} objects above threshold {score_threshold}')

    for i, (label, score) in enumerate(zip(result['labels'], result['scores'])):
        label_name = COCO_CLASSES.get(int(label), f'unknown({label})')
        print(f'   [{i+1:02d}] {label_name:<20s}  score={score:.3f}')

    return result

print("✅ run_inference ready!")


# ## 🎨 Cell 7 — Visualize Results (Masks + Boxes + Labels)

# In[21]:


def visualize_results(pil_image, results, mask_threshold=0.5,
                      show_masks=True, show_boxes=True, show_labels=True,
                      alpha=0.45, figsize=(14, 9)):
    """
    Visualize Mask R-CNN predictions on the image.

    Args:
        pil_image      : Original PIL image
        results        : Output from run_inference()
        mask_threshold : Binary threshold for mask pixels (default 0.5)
        show_masks     : Draw semi-transparent instance masks
        show_boxes     : Draw bounding boxes
        show_labels    : Draw class + score labels
        alpha          : Mask transparency (0=invisible, 1=solid)
        figsize        : Matplotlib figure size
    """
    img_array = np.array(pil_image).copy()   # H × W × 3, uint8
    overlay   = img_array.copy()

    boxes  = results['boxes']
    labels = results['labels']
    scores = results['scores']
    masks  = results['masks']   # (N, 1, H, W)
    N      = len(boxes)

    if N == 0:
        print('⚠️  No detections to display.')
        show_image(pil_image, 'No Detections')
        return

    # Draw each instance
    for i in range(N):
        color = COLORS[i % len(COLORS)]   # RGB tuple
        x1, y1, x2, y2 = boxes[i].astype(int)
        label_name = COCO_CLASSES[labels[i]]
        score      = scores[i]

        # ── Instance mask ─────────────────────────────
        if show_masks:
            binary_mask = (masks[i, 0] > mask_threshold)   # H × W bool
            overlay[binary_mask] = (
                overlay[binary_mask] * 0.55 +
                np.array(color)         * 0.45
            ).astype(np.uint8)

            # Mask contour
            mask_uint8  = binary_mask.astype(np.uint8) * 255
            contours, _ = cv2.findContours(
                mask_uint8, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(overlay, contours, -1, color, 2)

        # ── Bounding box ───────────────────────────────
        if show_boxes:
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)

        # ── Label chip ────────────────────────────────
        if show_labels:
            label_text = f'{label_name}: {score:.2f}'
            (tw, th), _ = cv2.getTextSize(
                label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            # Background rectangle
            cv2.rectangle(overlay,
                (x1, y1 - th - 8), (x1 + tw + 6, y1),
                color, -1)
            # Text
            cv2.putText(overlay, label_text,
                (x1 + 3, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (0, 0, 0), 1, cv2.LINE_AA)

    # Blend overlay with original
    final = cv2.addWeighted(overlay, alpha + 0.3, img_array, 1 - (alpha + 0.3), 0)

    # ── Plot ──────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    axes[0].imshow(img_array)
    axes[0].set_title('Original Image', fontweight='bold', fontsize=13)
    axes[0].axis('off')

    axes[1].imshow(final)
    axes[1].set_title(
        f'Mask R-CNN Output  ({N} instances detected)',
        fontweight='bold', fontsize=13)
    axes[1].axis('off')

    plt.suptitle('Instance Segmentation Result', fontsize=15, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig('output_segmented.png', dpi=150, bbox_inches='tight')
    plt.show()
    print('💾 Saved: output_segmented.png')

    return final


# ── Visualize ─────────────────────────────────────────
output = visualize_results(sample_image, results)


# ## 📂 Cell 8 — Run on YOUR OWN Image

# In[22]:


# ════════════════════════════════════════════════════════
#  OPTION A — Use a local file path
# ════════════════════════════════════════════════════════
# my_image = load_image_from_file('path/to/your/image.jpg')

# ════════════════════════════════════════════════════════
#  OPTION B — Paste an image URL
# ════════════════════════════════════════════════════════
# my_image = load_image_from_url('https://example.com/photo.jpg')

# ════════════════════════════════════════════════════════
#  OPTION C — Upload via Jupyter file browser
#             then use the filename below
# ════════════════════════════════════════════════════════
# from google.colab import files      # If running on Google Colab
# uploaded = files.upload()
# filename = list(uploaded.keys())[0]
# my_image = load_image_from_file(filename)

# ── For demo: use another COCO sample ─────────────────
CROWDED_URL = 'http://images.cocodataset.org/val2017/000000397133.jpg'
my_image = load_image_from_url(CROWDED_URL)

print(f'📸 Loaded image: {my_image.size}')
show_image(my_image, 'Your Input Image')

# ── Inference ─────────────────────────────────────────
my_results = run_inference(model, my_image, score_threshold=0.5)

# ── Visualize ─────────────────────────────────────────
visualize_results(my_image, my_results)


# ## 🎛️ Cell 9 — Interactive Controls (threshold, toggles)

# In[24]:


for threshold in [0.3, 0.5, 0.7, 0.9]:
    line = '='*50
    print(f'\n{line}')
    print(f'  Threshold = {threshold}')
    print(line)
    r = run_inference(model, sample_image, score_threshold=threshold)
    visualize_results(sample_image, r)


# ## 🎬 Cell 10 — Run on a Video File

# In[25]:


def process_video(video_path, output_path='output_video.mp4',
                  score_threshold=0.5, max_frames=None):
    """
    Run Mask R-CNN on every frame of a video.

    Args:
        video_path      : Path to input .mp4 / .avi file
        output_path     : Where to save the output video
        score_threshold : Minimum detection confidence
        max_frames      : Stop after N frames (None = entire video)
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f'Cannot open video: {video_path}')

    fps    = int(cap.get(cv2.CAP_PROP_FPS))
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f'🎬 Video: {width}×{height} @ {fps}fps  ({total} frames)')

    writer = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps, (width, height)
    )

    frame_idx = 0
    model.eval()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if max_frames and frame_idx >= max_frames:
            break

        pil_frame = load_image_from_array(frame)
        results   = run_inference(model, pil_frame, score_threshold)

        # Draw masks/boxes on frame
        annotated = np.array(pil_frame).copy()
        for i in range(len(results['boxes'])):
            color = COLORS[i % len(COLORS)]
            x1,y1,x2,y2 = results['boxes'][i].astype(int)
            binary_mask  = (results['masks'][i, 0] > 0.5)
            annotated[binary_mask] = (
                annotated[binary_mask] * 0.6 + np.array(color) * 0.4
            ).astype(np.uint8)
            cv2.rectangle(annotated, (x1,y1), (x2,y2), color, 2)
            label = f"{COCO_CLASSES[results['labels'][i]]}: {results['scores'][i]:.2f}"
            cv2.putText(annotated, label, (x1, y1-5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # Write BGR frame
        writer.write(cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))

        if frame_idx % 10 == 0:
            print(f'   Frame {frame_idx}/{total}  detections={len(results["boxes"])}')
        frame_idx += 1

    cap.release()
    writer.release()
    print(f'\n✅ Saved output video: {output_path}')


# ── Uncomment to run on your video ───────────────────
# process_video('your_video.mp4', output_path='output.mp4',
#               score_threshold=0.5, max_frames=100)
print('📌 Uncomment process_video() above and set your video path to run.')


# In[27]:


import urllib.request

print("⬇️ Downloading sample video...")
urllib.request.urlretrieve(
    'https://www.w3schools.com/html/mov_bbb.mp4',
    'sample_video.mp4'
)
print("✅ Downloaded!")

process_video(
    video_path      = 'sample_video.mp4',
    output_path     = 'output_video.mp4',
    score_threshold = 0.5,
    max_frames      = 30
)


# In[28]:


import os
print("📁 Your output video is here:")
print(os.path.abspath('output_video.mp4'))
print()
print("📁 Your segmented image is here:")
print(os.path.abspath('output_segmented.png'))


# In[29]:


import shutil

# ── Change this to wherever you want to save ──
save_folder = 'C:/Users/Kaustubh Gupta/Desktop/MaskRCNN_Project'

# Create folder if it doesn't exist
os.makedirs(save_folder, exist_ok=True)

# Copy all output files
shutil.copy('output_video.mp4',     f'{save_folder}/output_video.mp4')
shutil.copy('output_segmented.png', f'{save_folder}/output_segmented.png')

print("✅ Files saved to:", save_folder)
print("   - output_video.mp4")
print("   - output_segmented.png")


# ## 🏋️ Cell 11 — Fine-Tune on Your Custom Dataset

# In[30]:


from torch.utils.data import Dataset, DataLoader

# ── Custom Dataset Class ──────────────────────────────
class CustomMaskDataset(Dataset):
    """
    Template for your own dataset.
    Expected folder structure:
        data/
          images/  image_001.jpg  image_002.jpg ...
          masks/   image_001.png  image_002.png ...
          labels.txt   (one class per line)

    Each mask PNG: pixel value = instance class ID.
    """
    def __init__(self, image_dir, mask_dir, transforms=None):
        self.image_dir  = image_dir
        self.mask_dir   = mask_dir
        self.transforms = transforms
        self.images = sorted(os.listdir(image_dir))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name  = self.images[idx]
        mask_name = img_name.replace('.jpg', '.png').replace('.jpeg', '.png')

        # Load image
        image = Image.open(os.path.join(self.image_dir, img_name)).convert('RGB')

        # Load mask (H × W, each unique value = one instance)
        mask = np.array(Image.open(os.path.join(self.mask_dir, mask_name)))

        instance_ids = np.unique(mask)
        instance_ids = instance_ids[instance_ids != 0]   # exclude background

        masks, boxes, labels = [], [], []
        for inst_id in instance_ids:
            m = (mask == inst_id).astype(np.uint8)
            pos = np.where(m)
            x1, y1 = np.min(pos[1]), np.min(pos[0])
            x2, y2 = np.max(pos[1]), np.max(pos[0])
            if x2 > x1 and y2 > y1:   # valid box
                masks.append(m)
                boxes.append([x1, y1, x2, y2])
                labels.append(1)       # ← change to actual class ID

        target = {
            'boxes':   torch.FloatTensor(boxes)                              if boxes else torch.zeros((0,4)),
            'masks':   torch.as_tensor(np.array(masks), dtype=torch.uint8)  if masks else torch.zeros((0, mask.shape[0], mask.shape[1]), dtype=torch.uint8),
            'labels':  torch.LongTensor(labels)                              if labels else torch.zeros(0, dtype=torch.long),
            'image_id': torch.tensor([idx]),
            'area':    torch.FloatTensor([(b[2]-b[0])*(b[3]-b[1]) for b in boxes]) if boxes else torch.zeros(0),
            'iscrowd': torch.zeros(len(boxes), dtype=torch.int64),
        }

        if self.transforms:
            image = self.transforms(image)
        else:
            image = T.ToTensor()(image)

        return image, target


# ── Training Function ─────────────────────────────────
def train_one_epoch(model, optimizer, data_loader, device, epoch):
    model.train()
    total_loss = 0.0

    for batch_idx, (images, targets) in enumerate(data_loader):
        images  = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # Forward pass — returns dict of losses during training
        loss_dict = model(images, targets)

        # Sum all losses
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        total_loss += losses.item()

        if (batch_idx + 1) % 10 == 0:
            breakdown = '  '.join([f'{k}={v.item():.3f}' for k, v in loss_dict.items()])
            print(f'  Epoch {epoch}  Batch {batch_idx+1}/{len(data_loader)}  {breakdown}')

    return total_loss / len(data_loader)


def train_model(num_classes, train_img_dir, train_mask_dir,
                num_epochs=10, batch_size=2, lr=0.005,
                checkpoint_dir='checkpoints'):
    """
    Full training loop for custom fine-tuning.

    Args:
        num_classes    : Number of YOUR classes + 1 (background)
        train_img_dir  : Path to training images folder
        train_mask_dir : Path to training masks folder
        num_epochs     : Total epochs
        batch_size     : Images per batch (2 recommended for 8GB GPU)
        lr             : Initial learning rate
        checkpoint_dir : Folder to save .pth checkpoints
    """
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Dataset & DataLoader
    dataset = CustomMaskDataset(train_img_dir, train_mask_dir)
    loader  = DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        collate_fn=lambda x: tuple(zip(*x)),   # required for variable-size targets
        num_workers=2
    )

    # Build model
    model = load_custom_model(num_classes)
    model.train()

    # Optimizer & scheduler
    params    = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=0.0005)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    loss_history = []

    print(f'🏋️  Starting training: {num_epochs} epochs, {len(dataset)} images')
    print(f'   Classes: {num_classes}  |  Batch: {batch_size}  |  LR: {lr}\n')

    for epoch in range(1, num_epochs + 1):
        epoch_loss = train_one_epoch(model, optimizer, loader, device, epoch)
        scheduler.step()
        loss_history.append(epoch_loss)

        # Save checkpoint
        ckpt_path = os.path.join(checkpoint_dir, f'maskrcnn_epoch{epoch:02d}.pth')
        torch.save(model.state_dict(), ckpt_path)
        print(f'✅ Epoch {epoch:02d}  avg_loss={epoch_loss:.4f}  → saved {ckpt_path}\n')

    # Plot loss curve
    plt.figure(figsize=(8, 4))
    plt.plot(range(1, num_epochs+1), loss_history, 'o-', color='#FF6B6B', linewidth=2)
    plt.xlabel('Epoch'); plt.ylabel('Loss')
    plt.title('Training Loss Curve', fontweight='bold')
    plt.grid(True, alpha=0.3); plt.tight_layout()
    plt.savefig('training_loss.png', dpi=150)
    plt.show()

    return model, loss_history


# ── HOW TO USE — uncomment to fine-tune ──────────────
# trained_model, history = train_model(
#     num_classes    = 3,                      # e.g., background + cat + dog
#     train_img_dir  = 'data/images/',
#     train_mask_dir = 'data/masks/',
#     num_epochs     = 10,
#     batch_size     = 2,
#     lr             = 0.005
# )
print('📌 Fine-tuning template ready. Fill in your dataset paths and run train_model().')


# In[32]:


import os

print("📁 Files in PennFudanPed dataset:")
for item in os.listdir('PennFudanPed'):
    full_path = os.path.join('PennFudanPed', item)
    if os.path.isdir(full_path):
        count = len(os.listdir(full_path))
        print(f"   📂 {item}/  ({count} files)")
    else:
        print(f"   📄 {item}")


# In[33]:


import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from PIL import Image
import numpy as np
import os

class PennFudanDataset(Dataset):
    def __init__(self, root, transforms=None):
        self.root       = root
        self.transforms = transforms
        self.imgs  = sorted(os.listdir(os.path.join(root, 'PNGImages')))
        self.masks = sorted(os.listdir(os.path.join(root, 'PedMasks')))

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        # Load image
        img_path  = os.path.join(self.root, 'PNGImages', self.imgs[idx])
        mask_path = os.path.join(self.root, 'PedMasks',  self.masks[idx])

        img  = Image.open(img_path).convert('RGB')
        mask = np.array(Image.open(mask_path))

        # Each unique color = one person instance
        instance_ids = np.unique(mask)
        instance_ids = instance_ids[instance_ids != 0]  # remove background

        masks, boxes, labels = [], [], []
        for inst_id in instance_ids:
            m   = (mask == inst_id).astype(np.uint8)
            pos = np.where(m)
            x1  = int(np.min(pos[1]))
            y1  = int(np.min(pos[0]))
            x2  = int(np.max(pos[1]))
            y2  = int(np.max(pos[0]))
            if x2 > x1 and y2 > y1:
                masks.append(m)
                boxes.append([x1, y1, x2, y2])
                labels.append(1)  # 1 = person

        target = {
            'boxes':    torch.FloatTensor(boxes),
            'masks':    torch.as_tensor(np.array(masks), dtype=torch.uint8),
            'labels':   torch.LongTensor(labels),
            'image_id': torch.tensor([idx]),
            'area':     torch.FloatTensor(
                            [(b[2]-b[0])*(b[3]-b[1]) for b in boxes]),
            'iscrowd':  torch.zeros(len(boxes), dtype=torch.int64),
        }

        img = T.ToTensor()(img)
        return img, target


# Load dataset
dataset    = PennFudanDataset('PennFudanPed')
val_size   = 10
train_size = len(dataset) - val_size
train_set, val_set = torch.utils.data.random_split(
    dataset, [train_size, val_size])

train_loader = DataLoader(
    train_set, batch_size=2, shuffle=True,
    collate_fn=lambda x: tuple(zip(*x)))

print(f"✅ Dataset ready!")
print(f"   Total images : {len(dataset)}")
print(f"   Train        : {train_size}")
print(f"   Validation   : {val_size}")


# In[34]:


from torchvision.models.detection import maskrcnn_resnet50_fpn, MaskRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

# 2 classes = background + person
NUM_CLASSES = 2

# Load pretrained model
ft_model = maskrcnn_resnet50_fpn(
    weights=MaskRCNN_ResNet50_FPN_Weights.DEFAULT)

# Replace box head
in_features = ft_model.roi_heads.box_predictor.cls_score.in_features
ft_model.roi_heads.box_predictor = FastRCNNPredictor(
    in_features, NUM_CLASSES)

# Replace mask head
in_features_mask = ft_model.roi_heads.mask_predictor.conv5_mask.in_channels
ft_model.roi_heads.mask_predictor = MaskRCNNPredictor(
    in_features_mask, 256, NUM_CLASSES)

ft_model.to(device)

# Optimizer
optimizer = torch.optim.SGD(
    [p for p in ft_model.parameters() if p.requires_grad],
    lr=0.005, momentum=0.9, weight_decay=0.0005)

scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer, step_size=3, gamma=0.1)

print("✅ Model ready for fine-tuning!")
print(f"   Device     : {device}")
print(f"   Classes    : {NUM_CLASSES} (background + person)")
print(f"   Optimizer  : SGD  lr=0.005")


# In[35]:


import time
import matplotlib.pyplot as plt

NUM_EPOCHS   = 3
loss_history = []

print(f"🏋️ Starting fine-tuning for {NUM_EPOCHS} epochs...\n")

for epoch in range(1, NUM_EPOCHS + 1):
    ft_model.train()
    epoch_loss = 0.0
    start_time = time.time()

    for batch_idx, (images, targets) in enumerate(train_loader):
        images  = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()}
                   for t in targets]

        loss_dict = ft_model(images, targets)
        losses    = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        epoch_loss += losses.item()

        if (batch_idx + 1) % 5 == 0:
            print(f"   Epoch {epoch} | Batch {batch_idx+1}/{len(train_loader)} "
                  f"| Loss: {losses.item():.4f}")

    scheduler.step()
    avg_loss   = epoch_loss / len(train_loader)
    epoch_time = time.time() - start_time
    loss_history.append(avg_loss)
    print(f"\n✅ Epoch {epoch} complete — avg loss: {avg_loss:.4f}  "
          f"time: {epoch_time:.1f}s\n")

# Plot loss curve
plt.figure(figsize=(7, 4))
plt.plot(range(1, NUM_EPOCHS+1), loss_history,
         'o-', color='#FF6B6B', linewidth=2, markersize=8)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Fine-tuning Loss Curve', fontweight='bold')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('finetuning_loss.png', dpi=150)
plt.show()
print("💾 Loss curve saved: finetuning_loss.png")


# In[36]:


import os

save_folder = 'C:/Users/Kaustubh Gupta/Desktop/MaskRCNN_Project'
os.makedirs(save_folder, exist_ok=True)

save_path = f'{save_folder}/maskrcnn_finetuned.pth'
torch.save(ft_model.state_dict(), save_path)

size_mb = os.path.getsize(save_path) / 1e6
print(f"✅ Fine-tuned model saved!")
print(f"   Path : {save_path}")
print(f"   Size : {size_mb:.1f} MB")


# ## 📊 Cell 12 — Evaluation (COCO mAP)

# In[37]:


def evaluate_on_batch(model, image_list, score_threshold=0.5):
    """
    Quick evaluation: run inference on multiple images,
    print detection counts and average confidence per class.

    Args:
        image_list      : List of PIL images
        score_threshold : Detection threshold
    """
    class_counts = {}
    class_scores = {}

    for i, img in enumerate(image_list):
        r = run_inference(model, img, score_threshold)
        for label, score in zip(r['labels'], r['scores']):
            name = COCO_CLASSES[label]
            class_counts[name] = class_counts.get(name, 0) + 1
            class_scores.setdefault(name, []).append(score)

    print('\n' + '='*45)
    print(f'  Evaluation Summary — {len(image_list)} images')
    print('='*45)
    print(f'{"Class":<20} {"Count":>6}  {"Avg Score":>10}')
    print('-'*45)
    for cls in sorted(class_counts, key=class_counts.get, reverse=True):
        avg_score = np.mean(class_scores[cls])
        print(f'{cls:<20} {class_counts[cls]:>6}  {avg_score:>10.3f}')
    print('='*45)


# ── Evaluate on a few COCO samples ────────────────────
EVAL_URLS = [
    'http://images.cocodataset.org/val2017/000000039769.jpg',
    'http://images.cocodataset.org/val2017/000000397133.jpg',
    'http://images.cocodataset.org/val2017/000000037777.jpg',
]
print('📥 Loading evaluation images...')
eval_images = [load_image_from_url(u) for u in EVAL_URLS]
evaluate_on_batch(model, eval_images, score_threshold=0.5)


# ## 🧪 Cell 13 — Save & Load Model Checkpoint

# In[38]:


# ── Save model weights ────────────────────────────────
def save_model(model, path='maskrcnn_weights.pth'):
    torch.save(model.state_dict(), path)
    size_mb = os.path.getsize(path) / 1e6
    print(f'💾 Model saved → {path}  ({size_mb:.1f} MB)')


# ── Load model weights ────────────────────────────────
def load_model_weights(model, path='maskrcnn_weights.pth'):
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    print(f'✅ Weights loaded from {path}')
    return model


# ── Export to TorchScript (for deployment) ────────────
def export_torchscript(model, path='maskrcnn_scripted.pt'):
    model.eval()
    scripted = torch.jit.script(model)
    scripted.save(path)
    print(f'🚀 TorchScript model saved → {path}')


# Save pretrained model weights
save_model(model, 'maskrcnn_coco_pretrained.pth')

# Reload them
model = load_pretrained_model()
model = load_model_weights(model, 'maskrcnn_coco_pretrained.pth')
print('\n✅ Save/load cycle complete!')


# ## 📋 Cell 14 — Quick Reference Summary

# In[39]:


summary = """
╔══════════════════════════════════════════════════════════════╗
║          MASK R-CNN QUICK REFERENCE                          ║
╠══════════════════════════════════════════════════════════════╣
║  LOAD MODEL                                                  ║
║    model = load_pretrained_model()    # COCO pretrained      ║
║    model = load_custom_model(n_cls)   # Fine-tune            ║
╠══════════════════════════════════════════════════════════════╣
║  INPUT OPTIONS                                               ║
║    img = load_image_from_file('img.jpg')                     ║
║    img = load_image_from_url('https://...')                  ║
║    img = load_image_from_array(cv2_bgr_array)                ║
╠══════════════════════════════════════════════════════════════╣
║  INFERENCE                                                   ║
║    results = run_inference(model, img, score_threshold=0.5)  ║
║    # results keys: boxes, labels, scores, masks              ║
╠══════════════════════════════════════════════════════════════╣
║  VISUALIZE                                                   ║
║    visualize_results(img, results,                           ║
║        show_masks=True, show_boxes=True, show_labels=True)   ║
╠══════════════════════════════════════════════════════════════╣
║  VIDEO                                                       ║
║    process_video('input.mp4', 'output.mp4')                  ║
╠══════════════════════════════════════════════════════════════╣
║  FINE-TUNE                                                   ║
║    train_model(num_classes, img_dir, mask_dir, epochs=10)    ║
╠══════════════════════════════════════════════════════════════╣
║  SAVE / LOAD                                                 ║
║    save_model(model, 'my_model.pth')                         ║
║    model = load_model_weights(model, 'my_model.pth')         ║
╚══════════════════════════════════════════════════════════════╝
"""
print(summary)


# In[48]:


my_image = load_image_from_file(r"C:\Users\Kaustubh Gupta\Downloads\picnic-7055653_1280.jpg")
print("📸 Running Mask R-CNN on your image...")
my_results = run_inference(model, my_image, score_threshold=0.5)

# Show only image — no array data
visualize_results(my_image, my_results)
plt.show()


# In[49]:


get_ipython().system('pip install moviepy gitpython')

from moviepy.editor import VideoFileClip
import os

# ── Step 1: Check original video size ────────────
size_mb = os.path.getsize(r"C:\Users\Kaustubh Gupta\Videos\Screen Recordings\Screen Recording 2026-03-05 013216.mp4") / 1e6
print(f"📁 Original video size: {size_mb:.1f} MB")

# ── Step 2: Compress video to under 25MB ─────────
print("\n🔄 Compressing video...")

clip = VideoFileClip('output_video.mp4')

# Reduce size based on original size
if size_mb > 100:
    clip = clip.resize(width=320)
    fps  = 8
    bitrate = '200k'
elif size_mb > 50:
    clip = clip.resize(width=400)
    fps  = 8
    bitrate = '300k'
elif size_mb > 25:
    clip = clip.resize(width=480)
    fps  = 10
    bitrate = '400k'
else:
    clip = clip.resize(width=640)
    fps  = 15
    bitrate = '500k'

# Take max 10 seconds to keep size small
clip = clip.subclip(0, min(10, clip.duration))

# Save compressed video
clip.write_videofile(
    'output_video_github.mp4',
    fps     = fps,
    codec   = 'libx264',
    bitrate = bitrate
)

# ── Step 3: Check compressed size ────────────────
new_size = os.path.getsize('output_video_github.mp4') / 1e6
print(f"\n✅ Compressed video size: {new_size:.1f} MB")

if new_size < 25:
    print("✅ Ready to upload to GitHub directly!")
else:
    print("⚠️ Still too large — compressing more...")

    # Compress even more
    clip2 = VideoFileClip('output_video_github.mp4')
    clip2 = clip2.resize(width=240)
    clip2 = clip2.subclip(0, min(5, clip2.duration))
    clip2.write_videofile(
        'output_video_github.mp4',
        fps     = 6,
        codec   = 'libx264',
        bitrate = '150k'
    )
    final_size = os.path.getsize('output_video_github.mp4') / 1e6
    print(f"✅ Final size: {final_size:.1f} MB")


# In[50]:


get_ipython().system('pip uninstall moviepy -y')
get_ipython().system('pip install moviepy==1.0.3')


# In[8]:


from moviepy.editor import VideoFileClip
import os
import matplotlib.pyplot as plt
import cv2
from IPython.display import Video, display

# ── Your video path ───────────────────────────────
video_path = r"C:\Users\Kaustubh Gupta\Videos\Screen Recordings\Screen Recording 2026-03-05 013216.mp4"

# ── Step 1: Check original video size ────────────
size_mb = os.path.getsize(video_path) / 1e6
print(f"📁 Original video size: {size_mb:.1f} MB")

# ── Step 2: Compress video ────────────────────────
print("🔄 Compressing video...")

clip = VideoFileClip(video_path)

if size_mb > 100:
    clip = clip.resize(width=320)
    fps     = 8
    bitrate = '200k'
elif size_mb > 50:
    clip = clip.resize(width=400)
    fps     = 8
    bitrate = '300k'
elif size_mb > 25:
    clip = clip.resize(width=480)
    fps     = 10
    bitrate = '400k'
else:
    clip = clip.resize(width=640)
    fps     = 15
    bitrate = '500k'

clip.write_videofile(
    'output_video_small.mp4',
    fps     = fps,
    codec   = 'libx264',
    bitrate = bitrate
)

new_size = os.path.getsize('output_video_small.mp4') / 1e6
print(f"✅ Compressed size: {new_size:.1f} MB")

# ── Step 3: Show 6 frames from video ─────────────
print("🎬 Showing video frames...")

cap   = cv2.VideoCapture('output_video_small.mp4')
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
axes = axes.flatten()

positions = [0, total//5, 2*total//5,
             3*total//5, 4*total//5, total-1]

for i, pos in enumerate(positions):
    cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
    ret, frame = cap.read()
    if ret:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        axes[i].imshow(frame_rgb)
        axes[i].set_title(f'Frame {pos}', fontweight='bold')
        axes[i].axis('off')

cap.release()
plt.suptitle('🎬 Mask R-CNN Video Output',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

# ── Step 4: Play video inside Jupyter ────────────
print("▶️  Playing video...")
display(Video('output_video_small.mp4', width=640))


# In[ ]:




