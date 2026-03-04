import streamlit as st
import torch
import torchvision
import torchvision.transforms as T
from torchvision.models.detection import maskrcnn_resnet50_fpn, MaskRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
import requests
from io import BytesIO
import os
import time
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="🎭 Mask R-CNN Instance Segmentation",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.main-title {
    font-size: 2.4rem; font-weight: 900;
    background: linear-gradient(135deg, #667eea, #764ba2);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin-bottom: 0;
}
.subtitle { color: #888; font-size: 1rem; margin-top: 4px; }
.metric-card {
    background: white; border-radius: 14px;
    padding: 18px 22px; border: 1px solid #e5e7eb;
    box-shadow: 0 2px 10px rgba(0,0,0,0.06);
    text-align: center;
}
.metric-val { font-size: 2rem; font-weight: 800; color: #764ba2; }
.metric-lbl { font-size: 0.8rem; color: #888; margin-top: 2px; }
.info-box {
    background: #f0f4ff; border-left: 4px solid #667eea;
    border-radius: 8px; padding: 14px 18px;
    font-size: 0.88rem; color: #333; margin: 10px 0;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
COCO_CLASSES = {
    1:'person',      2:'bicycle',      3:'car',          4:'motorcycle',
    5:'airplane',    6:'bus',          7:'train',        8:'truck',
    9:'boat',       10:'traffic light',11:'fire hydrant',13:'stop sign',
   14:'parking meter',15:'bench',     16:'bird',        17:'cat',
   18:'dog',        19:'horse',       20:'sheep',       21:'cow',
   22:'elephant',   23:'bear',        24:'zebra',       25:'giraffe',
   27:'backpack',   28:'umbrella',    31:'handbag',     32:'tie',
   33:'suitcase',   34:'frisbee',     35:'skis',        36:'snowboard',
   37:'sports ball',38:'kite',        39:'baseball bat',40:'baseball glove',
   41:'skateboard', 42:'surfboard',   43:'tennis racket',44:'bottle',
   46:'wine glass', 47:'cup',         48:'fork',        49:'knife',
   50:'spoon',      51:'bowl',        52:'banana',      53:'apple',
   54:'sandwich',   55:'orange',      56:'broccoli',    57:'carrot',
   58:'hot dog',    59:'pizza',       60:'donut',       61:'cake',
   62:'chair',      63:'couch',       64:'potted plant',65:'bed',
   67:'dining table',70:'toilet',     72:'tv',          73:'laptop',
   74:'mouse',      75:'remote',      76:'keyboard',    77:'cell phone',
   78:'microwave',  79:'oven',        80:'toaster',     81:'sink',
   82:'refrigerator',84:'book',       85:'clock',       86:'vase',
   87:'scissors',   88:'teddy bear', 89:'hair drier',  90:'toothbrush',
}

COLORS = [
    (255,99,99),   (78,205,196),  (69,183,209),  (150,206,180),
    (255,178,102), (221,160,221), (152,216,200), (241,148,138),
    (133,193,233), (240,178,122), (130,224,170), (255,102,178),
    (178,102,255), (102,178,255), (255,255,102), (102,255,255),
    (200,150,255), (255,200,100), (100,255,200), (255,100,200),
]

SAMPLE_IMAGES = {
    "🐱 Cats on a couch":    "http://images.cocodataset.org/val2017/000000039769.jpg",
    "🚗 Street scene":       "http://images.cocodataset.org/val2017/000000397133.jpg",
    "👥 People & objects":   "http://images.cocodataset.org/val2017/000000037777.jpg",
    "🍕 Kitchen scene":      "http://images.cocodataset.org/val2017/000000252219.jpg",
    "🌳 Outdoor scene":      "http://images.cocodataset.org/val2017/000000087038.jpg",
    "🏀 Sports scene":       "http://images.cocodataset.org/val2017/000000174482.jpg",
}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ─────────────────────────────────────────────────────────────────────────────
# MODEL (cached so it loads only once)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model():
    weights = MaskRCNN_ResNet50_FPN_Weights.DEFAULT
    m = maskrcnn_resnet50_fpn(weights=weights)
    m.to(device)
    m.eval()
    return m

# ─────────────────────────────────────────────────────────────────────────────
# INFERENCE
# ─────────────────────────────────────────────────────────────────────────────
def run_inference(model, pil_image, score_threshold=0.5):
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

# ─────────────────────────────────────────────────────────────────────────────
# VISUALISE
# ─────────────────────────────────────────────────────────────────────────────
def draw_results(pil_image, results,
                 mask_thr=0.5, show_masks=True,
                 show_boxes=True, show_labels=True, alpha=0.45):
    img  = np.array(pil_image).copy()
    over = img.copy()
    N    = len(results['boxes'])

    for i in range(N):
        color  = COLORS[i % len(COLORS)]
        x1,y1,x2,y2 = results['boxes'][i].astype(int)
        lname  = COCO_CLASSES.get(int(results['labels'][i]), '?')
        score  = results['scores'][i]

        if show_masks:
            bm = (results['masks'][i, 0] > mask_thr)
            over[bm] = (over[bm]*0.55 + np.array(color)*0.45).astype(np.uint8)
            mu8 = bm.astype(np.uint8)*255
            ctrs,_ = cv2.findContours(mu8, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(over, ctrs, -1, color, 2)

        if show_boxes:
            cv2.rectangle(over, (x1,y1), (x2,y2), color, 2)

        if show_labels:
            txt = f'{lname}: {score:.2f}'
            (tw,th),_ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            cv2.rectangle(over, (x1, y1-th-8), (x1+tw+6, y1), color, -1)
            cv2.putText(over, txt, (x1+3, y1-4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,0,0), 1, cv2.LINE_AA)

    final = cv2.addWeighted(over, alpha+0.3, img, 1-(alpha+0.3), 0)
    return final, N

# ─────────────────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<p class="main-title">🎭 Mask R-CNN Instance Segmentation</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">ResNet-50 FPN backbone · Pretrained on MS-COCO 2017 · 80 object categories</p>', unsafe_allow_html=True)
st.markdown("---")

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Detection Settings")

    score_thr = st.slider("🎯 Confidence Threshold", 0.10, 0.95, 0.50, 0.05,
        help="Lower = more detections (may include false positives)")

    mask_thr  = st.slider("🎨 Mask Threshold",      0.10, 0.90, 0.50, 0.05,
        help="Threshold for converting soft masks to binary")

    alpha     = st.slider("🌫️ Mask Transparency",   0.10, 0.90, 0.45, 0.05)

    st.markdown("---")
    st.subheader("👁️ Display Options")
    show_masks  = st.checkbox("Show Segmentation Masks", value=True)
    show_boxes  = st.checkbox("Show Bounding Boxes",     value=True)
    show_labels = st.checkbox("Show Class Labels",       value=True)

    st.markdown("---")
    st.subheader("ℹ️ System Info")
    st.markdown(f"**Device:** `{str(device).upper()}`")
    st.markdown(f"**PyTorch:** `{torch.__version__}`")
    st.markdown(f"**Classes:** `{len(COCO_CLASSES)}` COCO categories")
    if torch.cuda.is_available():
        st.markdown(f"**GPU:** `{torch.cuda.get_device_name(0)}`")

# ─────────────────────────────────────────────────────────────────────────────
# IMAGE INPUT
# ─────────────────────────────────────────────────────────────────────────────
st.subheader("📥 Step 1 — Choose Your Image")

tab1, tab2, tab3 = st.tabs(["📁 Upload File", "🔗 Image URL", "🖼️ Sample Images"])

pil_image = None

with tab1:
    uploaded = st.file_uploader(
        "Upload a JPG or PNG image", type=["jpg","jpeg","png"],
        help="Any photo works — street scenes, animals, kitchens, sports, etc."
    )
    if uploaded:
        pil_image = Image.open(uploaded).convert("RGB")
        st.success(f"✅ Loaded: {uploaded.name}  ({pil_image.size[0]}×{pil_image.size[1]}px)")

with tab2:
    url = st.text_input("Paste any image URL:",
        placeholder="https://example.com/photo.jpg")
    if url:
        try:
            with st.spinner("Downloading..."):
                r = requests.get(url, timeout=10)
                pil_image = Image.open(BytesIO(r.content)).convert("RGB")
            st.success(f"✅ Downloaded  ({pil_image.size[0]}×{pil_image.size[1]}px)")
        except Exception as e:
            st.error(f"❌ Failed to load image: {e}")

with tab3:
    st.markdown("Pick one of these COCO validation images:")
    choice = st.selectbox("Select sample:", list(SAMPLE_IMAGES.keys()))
    if st.button("📥 Load Sample Image"):
        try:
            with st.spinner("Loading..."):
                r = requests.get(SAMPLE_IMAGES[choice], timeout=10)
                pil_image = Image.open(BytesIO(r.content)).convert("RGB")
            st.success(f"✅ Loaded  ({pil_image.size[0]}×{pil_image.size[1]}px)")
        except Exception as e:
            st.error(f"❌ Failed: {e}")

# ─────────────────────────────────────────────────────────────────────────────
# INFERENCE + RESULTS
# ─────────────────────────────────────────────────────────────────────────────
if pil_image is not None:
    st.markdown("---")
    st.subheader("🔮 Step 2 — Run Detection")

    col_img, col_btn = st.columns([3, 1])
    with col_img:
        st.image(pil_image, caption="Input Image", use_container_width=True)
    with col_btn:
        st.markdown("<br><br>", unsafe_allow_html=True)
        run = st.button("🚀 Run Mask R-CNN", type="primary",
                        use_container_width=True)
        st.markdown("""
        <div class="info-box">
        Model will detect & segment all objects it recognises from 80 COCO categories.
        </div>
        """, unsafe_allow_html=True)

    if run:
        # Load model
        with st.spinner("🧠 Loading Mask R-CNN model (first time may take ~30s)..."):
            model = load_model()

        # Run inference
        with st.spinner("⚡ Running inference..."):
            results = run_inference(model, pil_image, score_thr)

        # Draw results
        with st.spinner("🎨 Drawing results..."):
            output_img, n_det = draw_results(
                pil_image, results,
                mask_thr=mask_thr,
                show_masks=show_masks,
                show_boxes=show_boxes,
                show_labels=show_labels,
                alpha=alpha
            )

        # ── Metrics row ───────────────────────────────────────────────────────
        st.markdown("---")
        st.subheader("📊 Results")

        mc1, mc2, mc3, mc4 = st.columns(4)
        mc1.markdown(f"""<div class="metric-card">
            <div class="metric-val">{n_det}</div>
            <div class="metric-lbl">Objects Detected</div></div>""",
            unsafe_allow_html=True)
        mc2.markdown(f"""<div class="metric-card">
            <div class="metric-val">{results['time']*1000:.0f}ms</div>
            <div class="metric-lbl">Inference Time</div></div>""",
            unsafe_allow_html=True)
        mc3.markdown(f"""<div class="metric-card">
            <div class="metric-val">{score_thr:.2f}</div>
            <div class="metric-lbl">Confidence Threshold</div></div>""",
            unsafe_allow_html=True)
        mc4.markdown(f"""<div class="metric-card">
            <div class="metric-val">{str(device).upper()}</div>
            <div class="metric-lbl">Device Used</div></div>""",
            unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Side-by-side images ───────────────────────────────────────────────
        left, right = st.columns(2)
        with left:
            st.markdown("**🖼️ Original Image**")
            st.image(pil_image, use_container_width=True)
        with right:
            st.markdown(f"**🎭 Segmentation Output  —  {n_det} instances**")
            st.image(output_img, use_container_width=True)

        # ── Download button ───────────────────────────────────────────────────
        out_pil = Image.fromarray(output_img)
        buf = BytesIO()
        out_pil.save(buf, format="PNG")
        st.download_button(
            "⬇️ Download Segmented Image",
            data=buf.getvalue(),
            file_name="mask_rcnn_output.png",
            mime="image/png",
            use_container_width=True
        )

        # ── Detection table ───────────────────────────────────────────────────
        if n_det > 0:
            st.markdown("---")
            st.subheader("📋 Detection Details")

            rows = []
            for i in range(n_det):
                lid   = int(results['labels'][i])
                lname = COCO_CLASSES.get(lid, f'class_{lid}')
                score = float(results['scores'][i])
                box   = results['boxes'][i].astype(int)
                w     = int(box[2]-box[0])
                h     = int(box[3]-box[1])
                rows.append({
                    "#":        i+1,
                    "Class":    lname,
                    "Score":    f"{score:.3f}",
                    "Width px": w,
                    "Height px":h,
                    "Box [x1,y1,x2,y2]": f"[{box[0]}, {box[1]}, {box[2]}, {box[3]}]"
                })

            df = pd.DataFrame(rows)
            st.dataframe(df, use_container_width=True, hide_index=True)

            # ── Class distribution bar chart ──────────────────────────────────
            st.subheader("📈 Class Distribution")
            class_counts = {}
            for i in range(n_det):
                name = COCO_CLASSES.get(int(results['labels'][i]), '?')
                class_counts[name] = class_counts.get(name, 0) + 1

            fig, ax = plt.subplots(figsize=(max(6, len(class_counts)*1.2), 3.5))
            bars = ax.bar(class_counts.keys(), class_counts.values(),
                          color='#764ba2', edgecolor='white', linewidth=0.8)
            for bar, val in zip(bars, class_counts.values()):
                ax.text(bar.get_x() + bar.get_width()/2,
                        bar.get_height() + 0.05, str(val),
                        ha='center', va='bottom', fontweight='bold', fontsize=11)
            ax.set_xlabel("Detected Class", fontsize=11)
            ax.set_ylabel("Count", fontsize=11)
            ax.set_title("Instances per Class", fontsize=13, fontweight='bold')
            ax.set_ylim(0, max(class_counts.values()) + 1)
            plt.xticks(rotation=30, ha='right', fontsize=10)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

            # ── Score distribution ────────────────────────────────────────────
            if n_det > 1:
                st.subheader("📉 Confidence Score Distribution")
                fig2, ax2 = plt.subplots(figsize=(8, 3))
                scores_list = results['scores'].tolist()
                labels_list = [COCO_CLASSES.get(int(l),'?')
                               for l in results['labels']]
                colors_bar  = [f'#{COLORS[i%len(COLORS)][0]:02x}'
                                f'{COLORS[i%len(COLORS)][1]:02x}'
                                f'{COLORS[i%len(COLORS)][2]:02x}'
                               for i in range(n_det)]
                ax2.barh(range(n_det), scores_list,
                         color=colors_bar, edgecolor='white')
                ax2.set_yticks(range(n_det))
                ax2.set_yticklabels(
                    [f"#{i+1} {labels_list[i]}" for i in range(n_det)],
                    fontsize=9)
                ax2.set_xlabel("Confidence Score")
                ax2.set_xlim(0, 1.05)
                ax2.axvline(score_thr, color='red', linestyle='--',
                            linewidth=1.5, label=f'Threshold={score_thr}')
                ax2.legend(fontsize=9)
                ax2.set_title("Per-Instance Confidence", fontweight='bold')
                plt.tight_layout()
                st.pyplot(fig2)
                plt.close()

        else:
            st.warning("⚠️ No objects detected. Try lowering the **Confidence Threshold** in the sidebar.")

# ─────────────────────────────────────────────────────────────────────────────
# ABOUT SECTION
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
col_a, col_b = st.columns(2)

with col_a:
    with st.expander("ℹ️ About Mask R-CNN"):
        st.markdown("""
**Mask R-CNN** extends Faster R-CNN by adding a branch for predicting
segmentation masks on each Region of Interest (RoI).

**Pipeline:**
1. **Backbone** — ResNet-50 extracts feature maps
2. **FPN** — Feature Pyramid Network combines multi-scale features
3. **RPN** — Region Proposal Network suggests candidate boxes
4. **RoI Align** — Pools features for each proposal
5. **Heads** — Three parallel branches predict:
   - Class label
   - Bounding box offset
   - Instance segmentation mask

**Paper:** He et al., *Mask R-CNN*, ICCV 2017
        """)

with col_b:
    with st.expander("📦 Model Specifications"):
        st.markdown(f"""
| Property | Value |
|---|---|
| Architecture | Mask R-CNN |
| Backbone | ResNet-50 + FPN |
| Pretrained on | MS-COCO 2017 |
| COCO categories | 80 |
| Parameters | ~44 million |
| Input | Any resolution PIL image |
| Output | Boxes · Masks · Labels · Scores |
| Device | `{str(device).upper()}` |
| PyTorch | `{torch.__version__}` |
        """)

with st.expander("🏷️ All 80 COCO Classes"):
    classes_list = list(COCO_CLASSES.values())
    cols = st.columns(5)
    for i, cls in enumerate(classes_list):
        cols[i % 5].markdown(f"• {cls}")

st.markdown("---")
st.markdown(
    "<p style='text-align:center;color:#aaa;font-size:0.8rem;'>"
    "🎭 Mask R-CNN Instance Segmentation · Built with Streamlit · "
    "Model: torchvision pretrained on MS-COCO 2017"
    "</p>",
    unsafe_allow_html=True
)
