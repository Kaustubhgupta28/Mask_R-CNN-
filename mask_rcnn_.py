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
import threading
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
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;900&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.main-title {
    font-size: 2.6rem; font-weight: 900;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin-bottom: 0; line-height: 1.2;
}
.subtitle {
    color: #888; font-size: 1rem; margin-top: 6px; margin-bottom: 0;
}
.metric-card {
    background: white; border-radius: 16px;
    padding: 20px 24px; border: 1px solid #e5e7eb;
    box-shadow: 0 4px 15px rgba(0,0,0,0.07);
    text-align: center; margin: 6px 0;
    transition: transform 0.2s;
}
.metric-card:hover { transform: translateY(-2px); }
.metric-val { font-size: 2rem; font-weight: 800; color: #764ba2; line-height: 1.2; }
.metric-lbl { font-size: 0.78rem; color: #9ca3af; margin-top: 4px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; }

.info-box {
    background: linear-gradient(135deg, #f0f4ff, #f5f0ff);
    border-left: 4px solid #667eea;
    border-radius: 10px; padding: 14px 18px;
    font-size: 0.88rem; color: #333; margin: 10px 0;
}
.webcam-box {
    background: linear-gradient(135deg, #fff0f8, #f0f8ff);
    border: 2px dashed #764ba2;
    border-radius: 16px; padding: 20px;
    text-align: center; margin: 10px 0;
}
.live-badge {
    display: inline-block;
    background: #ef4444; color: white;
    border-radius: 999px; padding: 3px 12px;
    font-size: 0.75rem; font-weight: 700;
    letter-spacing: 0.08em; margin-left: 8px;
    animation: pulse 1.5s ease-in-out infinite;
}
@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.5; }
}
.section-header {
    font-size: 1.2rem; font-weight: 800;
    color: #374151; margin: 20px 0 10px 0;
    display: flex; align-items: center; gap: 8px;
}
.tag {
    display: inline-block; background: #f3f0ff;
    color: #764ba2; border-radius: 6px;
    padding: 2px 10px; font-size: 0.78rem;
    font-weight: 600; margin: 2px;
}
.footer {
    text-align: center; color: #aaa;
    font-size: 0.8rem; padding: 20px 0;
}
div[data-testid="stTabs"] button {
    font-weight: 600 !important;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
COCO_CLASSES = {
    1:'person',      2:'bicycle',       3:'car',           4:'motorcycle',
    5:'airplane',    6:'bus',           7:'train',         8:'truck',
    9:'boat',       10:'traffic light', 11:'fire hydrant', 13:'stop sign',
   14:'parking meter',15:'bench',      16:'bird',         17:'cat',
   18:'dog',        19:'horse',        20:'sheep',        21:'cow',
   22:'elephant',   23:'bear',         24:'zebra',        25:'giraffe',
   27:'backpack',   28:'umbrella',     31:'handbag',      32:'tie',
   33:'suitcase',   34:'frisbee',      35:'skis',         36:'snowboard',
   37:'sports ball',38:'kite',         39:'baseball bat', 40:'baseball glove',
   41:'skateboard', 42:'surfboard',    43:'tennis racket',44:'bottle',
   46:'wine glass', 47:'cup',          48:'fork',         49:'knife',
   50:'spoon',      51:'bowl',         52:'banana',       53:'apple',
   54:'sandwich',   55:'orange',       56:'broccoli',     57:'carrot',
   58:'hot dog',    59:'pizza',        60:'donut',        61:'cake',
   62:'chair',      63:'couch',        64:'potted plant', 65:'bed',
   67:'dining table',70:'toilet',      72:'tv',           73:'laptop',
   74:'mouse',      75:'remote',       76:'keyboard',     77:'cell phone',
   78:'microwave',  79:'oven',         80:'toaster',      81:'sink',
   82:'refrigerator',84:'book',        85:'clock',        86:'vase',
   87:'scissors',   88:'teddy bear',   89:'hair drier',   90:'toothbrush',
}

COLORS = [
    (255,99,99),   (78,205,196),  (69,183,209),  (150,206,180),
    (255,178,102), (221,160,221), (152,216,200), (241,148,138),
    (133,193,233), (240,178,122), (130,224,170), (255,102,178),
    (178,102,255), (102,178,255), (255,255,102), (102,255,255),
    (200,150,255), (255,200,100), (100,255,200), (255,100,200),
]

SAMPLE_IMAGES = {
    "🐱 Cats on a couch":   "http://images.cocodataset.org/val2017/000000039769.jpg",
    "🚗 Street scene":      "http://images.cocodataset.org/val2017/000000397133.jpg",
    "👥 People & objects":  "http://images.cocodataset.org/val2017/000000037777.jpg",
    "🍕 Kitchen scene":     "http://images.cocodataset.org/val2017/000000252219.jpg",
    "🌳 Outdoor scene":     "http://images.cocodataset.org/val2017/000000087038.jpg",
    "🏀 Sports scene":      "http://images.cocodataset.org/val2017/000000174482.jpg",
}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ─────────────────────────────────────────────────────────────────────────────
# MODEL
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
# DRAW RESULTS
# ─────────────────────────────────────────────────────────────────────────────
def draw_results(pil_image, results,
                 mask_thr=0.5, show_masks=True,
                 show_boxes=True, show_labels=True, alpha=0.45):
    img  = np.array(pil_image).copy()
    over = img.copy()
    N    = len(results['boxes'])

    for i in range(N):
        color        = COLORS[i % len(COLORS)]
        x1,y1,x2,y2 = results['boxes'][i].astype(int)
        lname        = COCO_CLASSES.get(int(results['labels'][i]), '?')
        score        = results['scores'][i]

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
            cv2.rectangle(over, (x1,y1-th-8), (x1+tw+6,y1), color, -1)
            cv2.putText(over, txt, (x1+3,y1-4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,0,0), 1, cv2.LINE_AA)

    final = cv2.addWeighted(over, alpha+0.3, img, 1-(alpha+0.3), 0)
    return final, N

# ─────────────────────────────────────────────────────────────────────────────
# SHOW RESULTS (charts + table) — reusable function
# ─────────────────────────────────────────────────────────────────────────────
def show_result_details(results, n_det, score_thr):
    if n_det == 0:
        st.warning("⚠️ Koi object detect nahi hua. Confidence Threshold kam karo.")
        return

    # Metrics
    mc1, mc2, mc3, mc4 = st.columns(4)
    mc1.markdown(f'<div class="metric-card"><div class="metric-val">{n_det}</div><div class="metric-lbl">Objects Detected</div></div>', unsafe_allow_html=True)
    mc2.markdown(f'<div class="metric-card"><div class="metric-val">{results["time"]*1000:.0f}ms</div><div class="metric-lbl">Inference Time</div></div>', unsafe_allow_html=True)
    mc3.markdown(f'<div class="metric-card"><div class="metric-val">{score_thr:.2f}</div><div class="metric-lbl">Confidence Threshold</div></div>', unsafe_allow_html=True)
    mc4.markdown(f'<div class="metric-card"><div class="metric-val">{str(device).upper()}</div><div class="metric-lbl">Device</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Detection Table
    st.markdown('<div class="section-header">📋 Detection Details</div>', unsafe_allow_html=True)
    rows = []
    for i in range(n_det):
        lid   = int(results['labels'][i])
        lname = COCO_CLASSES.get(lid, f'class_{lid}')
        score = float(results['scores'][i])
        box   = results['boxes'][i].astype(int)
        rows.append({
            "#": i+1,
            "Class": lname,
            "Score": f"{score:.3f}",
            "Width px": int(box[2]-box[0]),
            "Height px": int(box[3]-box[1]),
            "Box [x1,y1,x2,y2]": f"[{box[0]},{box[1]},{box[2]},{box[3]}]"
        })
    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)

    # Charts side by side
    ch1, ch2 = st.columns(2)

    # Chart 1 — Class distribution
    with ch1:
        st.markdown('<div class="section-header">📊 Class Distribution</div>', unsafe_allow_html=True)
        class_counts = {}
        for i in range(n_det):
            name = COCO_CLASSES.get(int(results['labels'][i]), '?')
            class_counts[name] = class_counts.get(name, 0) + 1
        fig, ax = plt.subplots(figsize=(5, 3.5))
        bars = ax.bar(class_counts.keys(), class_counts.values(),
                      color='#764ba2', edgecolor='white', linewidth=0.8)
        for bar, val in zip(bars, class_counts.values()):
            ax.text(bar.get_x()+bar.get_width()/2,
                    bar.get_height()+0.05, str(val),
                    ha='center', va='bottom', fontweight='bold', fontsize=10)
        ax.set_ylabel("Count")
        ax.set_title("Instances per Class", fontweight='bold')
        ax.set_ylim(0, max(class_counts.values())+1.5)
        plt.xticks(rotation=30, ha='right', fontsize=9)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    # Chart 2 — Confidence scores
    with ch2:
        if n_det > 1:
            st.markdown('<div class="section-header">📉 Confidence Scores</div>', unsafe_allow_html=True)
            scores_list  = results['scores'].tolist()
            labels_list  = [COCO_CLASSES.get(int(l),'?') for l in results['labels']]
            colors_bar   = [f'#{COLORS[i%len(COLORS)][0]:02x}{COLORS[i%len(COLORS)][1]:02x}{COLORS[i%len(COLORS)][2]:02x}' for i in range(n_det)]
            fig2, ax2 = plt.subplots(figsize=(5, 3.5))
            ax2.barh(range(n_det), scores_list, color=colors_bar, edgecolor='white')
            ax2.set_yticks(range(n_det))
            ax2.set_yticklabels([f"#{i+1} {labels_list[i]}" for i in range(n_det)], fontsize=8)
            ax2.set_xlabel("Confidence Score")
            ax2.set_xlim(0, 1.05)
            ax2.axvline(score_thr, color='red', linestyle='--', linewidth=1.5, label=f'Threshold={score_thr}')
            ax2.legend(fontsize=8)
            ax2.set_title("Per-Instance Confidence", fontweight='bold')
            plt.tight_layout()
            st.pyplot(fig2)
            plt.close()

# ─────────────────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<p class="main-title">🎭 Mask R-CNN Instance Segmentation</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">ResNet-50 FPN backbone &nbsp;·&nbsp; MS-COCO 2017 pretrained &nbsp;·&nbsp; 80 object categories &nbsp;·&nbsp; Image + Webcam support</p>', unsafe_allow_html=True)
st.markdown("---")

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Detection Settings")

    score_thr = st.slider("🎯 Confidence Threshold", 0.10, 0.95, 0.50, 0.05,
        help="Kitna confident hona chahiye model — low = zyada detections")
    mask_thr  = st.slider("🎨 Mask Threshold",       0.10, 0.90, 0.50, 0.05,
        help="Mask ke pixels ka cutoff")
    alpha     = st.slider("🌫️ Mask Transparency",    0.10, 0.90, 0.45, 0.05,
        help="Mask kitna transparent ho")

    st.markdown("---")
    st.markdown("## 👁️ Display Options")
    show_masks  = st.checkbox("Show Segmentation Masks", value=True)
    show_boxes  = st.checkbox("Show Bounding Boxes",     value=True)
    show_labels = st.checkbox("Show Class Labels",       value=True)

    st.markdown("---")
    st.markdown("## 🖥️ System Info")
    st.markdown(f"**Device:** `{str(device).upper()}`")
    st.markdown(f"**PyTorch:** `{torch.__version__}`")
    st.markdown(f"**Classes:** `{len(COCO_CLASSES)}` COCO")
    if torch.cuda.is_available():
        st.success(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        st.info("Running on CPU")

# ─────────────────────────────────────────────────────────────────────────────
# MAIN TABS — 4 input modes
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">📥 Step 1 — Choose Input Source</div>', unsafe_allow_html=True)

tab1, tab2, tab3, tab4 = st.tabs([
    "📁 Upload Image",
    "🔗 Image URL",
    "🖼️ Sample Images",
    "📷 Webcam"
])

pil_image = None

# ── TAB 1: File Upload ────────────────────────────────────────────────────────
with tab1:
    uploaded = st.file_uploader(
        "Koi bhi JPG ya PNG image upload karo",
        type=["jpg","jpeg","png"],
        help="Street scene, animals, kitchen, sports — kuch bhi!"
    )
    if uploaded:
        pil_image = Image.open(uploaded).convert("RGB")
        st.success(f"✅ Image load ho gayi: {uploaded.name} ({pil_image.size[0]}×{pil_image.size[1]}px)")
        st.image(pil_image, caption="Uploaded Image", use_container_width=True)

# ── TAB 2: URL ────────────────────────────────────────────────────────────────
with tab2:
    url = st.text_input("Image URL yahan paste karo:",
        placeholder="https://example.com/photo.jpg")
    if url:
        try:
            with st.spinner("Downloading image..."):
                r = requests.get(url, timeout=10)
                pil_image = Image.open(BytesIO(r.content)).convert("RGB")
            st.success(f"✅ Downloaded! ({pil_image.size[0]}×{pil_image.size[1]}px)")
            st.image(pil_image, use_container_width=True)
        except Exception as e:
            st.error(f"❌ Image load nahi hui: {e}")

# ── TAB 3: Sample Images ──────────────────────────────────────────────────────
with tab3:
    st.markdown("Neeche se koi sample image choose karo:")
    choice = st.selectbox("Sample image:", list(SAMPLE_IMAGES.keys()))
    if st.button("📥 Load Sample Image", use_container_width=True):
        try:
            with st.spinner("Loading..."):
                r = requests.get(SAMPLE_IMAGES[choice], timeout=10)
                pil_image = Image.open(BytesIO(r.content)).convert("RGB")
            st.success(f"✅ Loaded! ({pil_image.size[0]}×{pil_image.size[1]}px)")
            st.image(pil_image, use_container_width=True)
        except Exception as e:
            st.error(f"❌ Failed: {e}")

# ── TAB 4: WEBCAM ─────────────────────────────────────────────────────────────
with tab4:
    st.markdown("""
    <div class="webcam-box">
        <h3>📷 Webcam Detection</h3>
        <p>Do modes available hain:</p>
        <p>
            <span class="tag">📸 Single Photo Mode</span> — Cloud + Local dono pe kaam karta hai<br><br>
            <span class="tag">🎥 Live Video Mode</span> — Sirf Local pe kaam karta hai
        </p>
    </div>
    """, unsafe_allow_html=True)

    webcam_mode = st.radio(
        "Webcam mode choose karo:",
        ["📸 Single Photo (Cloud + Local)", "🎥 Live Video (Sirf Local)"],
        horizontal=True
    )

    # ── WEBCAM MODE 1: Single Photo (st.camera_input) ────────────────────────
    if webcam_mode == "📸 Single Photo (Cloud + Local)":
        st.markdown("""
        <div class="info-box">
        📸 <b>Single Photo Mode</b> — Camera se ek photo lo, usse detect karo.<br>
        Streamlit Cloud aur Local dono pe kaam karta hai!
        </div>
        """, unsafe_allow_html=True)

        camera_img = st.camera_input("📷 Camera se photo lo")

        if camera_img is not None:
            pil_image = Image.open(camera_img).convert("RGB")
            st.success(f"✅ Photo li gayi! ({pil_image.size[0]}×{pil_image.size[1]}px)")

            # Auto detect on camera capture
            with st.spinner("🧠 Model load ho raha hai..."):
                model = load_model()
            with st.spinner("⚡ Detecting objects..."):
                results = run_inference(model, pil_image, score_thr)
            with st.spinner("🎨 Drawing..."):
                output_img, n_det = draw_results(
                    pil_image, results,
                    mask_thr=mask_thr,
                    show_masks=show_masks,
                    show_boxes=show_boxes,
                    show_labels=show_labels,
                    alpha=alpha
                )

            # Show side by side
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**📷 Original Photo**")
                st.image(pil_image, use_container_width=True)
            with c2:
                st.markdown(f"**🎭 Detected — {n_det} objects**")
                st.image(output_img, use_container_width=True)

            # Download
            out_pil = Image.fromarray(output_img)
            buf = BytesIO()
            out_pil.save(buf, format="PNG")
            st.download_button(
                "⬇️ Download Result",
                data=buf.getvalue(),
                file_name="webcam_detection.png",
                mime="image/png",
                use_container_width=True
            )

            st.markdown("---")
            show_result_details(results, n_det, score_thr)

    # ── WEBCAM MODE 2: Live Video ─────────────────────────────────────────────
    else:
        st.markdown("""
        <div class="info-box">
        🎥 <b>Live Video Mode</b> — Real-time detection har frame pe.<br>
        ⚠️ <b>Sirf Local computer pe kaam karta hai</b> — Streamlit Cloud pe nahi.<br>
        Local pe chalane ke liye: <code>streamlit run mask_rcnn_.py</code>
        </div>
        """, unsafe_allow_html=True)

        col_start, col_stop = st.columns(2)
        start_live = col_start.button("▶️ Start Live Detection", type="primary", use_container_width=True)
        stop_live  = col_stop.button("⏹️ Stop",                                  use_container_width=True)

        # Session state for live detection
        if 'live_running' not in st.session_state:
            st.session_state.live_running = False
        if start_live:
            st.session_state.live_running = True
        if stop_live:
            st.session_state.live_running = False

        if st.session_state.live_running:
            st.markdown('<span class="live-badge">● LIVE</span>', unsafe_allow_html=True)

            # Placeholders for live update
            frame_placeholder = st.empty()
            info_placeholder  = st.empty()

            with st.spinner("🧠 Model load ho raha hai..."):
                model = load_model()

            # Try to open webcam
            cap = cv2.VideoCapture(0)

            if not cap.isOpened():
                st.error("❌ Webcam nahi mila! Check karo ki webcam connected hai aur local pe chal raha hai.")
                st.session_state.live_running = False
            else:
                st.success("✅ Webcam connected! Live detection shuru...")
                frame_count = 0

                while st.session_state.live_running:
                    ret, frame = cap.read()
                    if not ret:
                        st.error("❌ Camera se frame nahi aaya.")
                        break

                    # Convert BGR to RGB
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    pil_frame = Image.fromarray(frame_rgb)

                    # Run inference
                    results = run_inference(model, pil_frame, score_thr)

                    # Draw results
                    output_frame, n_det = draw_results(
                        pil_frame, results,
                        mask_thr=mask_thr,
                        show_masks=show_masks,
                        show_boxes=show_boxes,
                        show_labels=show_labels,
                        alpha=alpha
                    )

                    # Show frame
                    frame_placeholder.image(
                        output_frame,
                        caption=f"🎥 Live — Frame #{frame_count} — {n_det} objects detected",
                        use_container_width=True
                    )

                    # Show live stats
                    info_placeholder.markdown(f"""
                    | Metric | Value |
                    |---|---|
                    | 🎯 Objects | **{n_det}** |
                    | ⚡ Time | **{results['time']*1000:.0f}ms** |
                    | 📸 Frame | **#{frame_count}** |
                    | 🖥️ Device | **{str(device).upper()}** |
                    """)

                    frame_count += 1

                    # Stop button check
                    if not st.session_state.live_running:
                        break

                cap.release()
                st.info("⏹️ Live detection band ho gayi.")

        else:
            # Show instructions when not running
            st.markdown("""
            <div style='text-align:center; padding:40px; background:#f9fafb; border-radius:16px; border:2px dashed #e5e7eb;'>
                <h2>🎥</h2>
                <p style='color:#888; font-size:1.1rem;'>
                    <b>▶️ Start Live Detection</b> button dabao<br>
                    webcam se real-time detection shuru ho jaayegi
                </p>
                <br>
                <p style='color:#aaa; font-size:0.85rem;'>
                    ⚠️ Local computer pe <code>streamlit run mask_rcnn_.py</code> se chalao
                </p>
            </div>
            """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — Run Detection (for Upload/URL/Sample tabs)
# ─────────────────────────────────────────────────────────────────────────────
if pil_image is not None:
    st.markdown("---")
    st.markdown('<div class="section-header">🔮 Step 2 — Run Detection</div>', unsafe_allow_html=True)

    run = st.button("🚀 Run Mask R-CNN", type="primary", use_container_width=True)

    if run:
        with st.spinner("🧠 Model load ho raha hai (pehli baar ~30s lag sakta hai)..."):
            model = load_model()
        with st.spinner("⚡ Inference chal rahi hai..."):
            results = run_inference(model, pil_image, score_thr)
        with st.spinner("🎨 Results draw ho rahe hain..."):
            output_img, n_det = draw_results(
                pil_image, results,
                mask_thr=mask_thr,
                show_masks=show_masks,
                show_boxes=show_boxes,
                show_labels=show_labels,
                alpha=alpha
            )

        # Side by side
        st.markdown("---")
        left, right = st.columns(2)
        with left:
            st.markdown("**🖼️ Original Image**")
            st.image(pil_image, use_container_width=True)
        with right:
            st.markdown(f"**🎭 Segmentation Output — {n_det} instances**")
            st.image(output_img, use_container_width=True)

        # Download
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

        st.markdown("---")
        show_result_details(results, n_det, score_thr)

# ─────────────────────────────────────────────────────────────────────────────
# ABOUT SECTION
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
col_a, col_b = st.columns(2)

with col_a:
    with st.expander("ℹ️ Mask R-CNN kya hai?"):
        st.markdown("""
**Mask R-CNN** ek deep learning model hai jo:
- Har object ko alag alag **detect** karta hai
- Har object ke aas paas **box** draw karta hai
- Har object ka **pixel-level mask** banata hai

**Kaise kaam karta hai:**
1. **ResNet-50** — image se features nikalti hai
2. **FPN** — alag alag sizes ki features combine karta hai
3. **RPN** — object ke possible locations suggest karta hai
4. **RoI Align** — har region ke features pool karta hai
5. **3 Heads** — class, box, aur mask predict karte hain

**Paper:** He et al., *Mask R-CNN*, ICCV 2017
        """)

with col_b:
    with st.expander("📦 Model Details"):
        st.markdown(f"""
| Property | Value |
|---|---|
| Architecture | Mask R-CNN |
| Backbone | ResNet-50 + FPN |
| Pretrained on | MS-COCO 2017 |
| Categories | 80 |
| Parameters | ~44 million |
| Device | `{str(device).upper()}` |
| PyTorch | `{torch.__version__}` |
| Webcam (Cloud) | ✅ Single Photo |
| Webcam (Local) | ✅ Live Video |
        """)

with st.expander("📷 Webcam Guide — Kaise use karein?"):
    st.markdown("""
### 📸 Single Photo Mode (Cloud + Local)
1. **Webcam** tab pe jao
2. **Single Photo** select karo
3. Camera allow karo browser mein
4. Photo lo — **automatic detect** ho jaayega!

---

### 🎥 Live Video Mode (Sirf Local)
1. Apne computer pe yeh command chalao:
```bash
pip install streamlit torch torchvision opencv-python
streamlit run mask_rcnn_.py
```
2. Browser mein `http://localhost:8501` khulega
3. **Webcam** tab → **Live Video** → **▶️ Start** dabao
4. Real-time detection shuru! **⏹️ Stop** se band karo

---

### ⚠️ Important Notes:
- Live Video mode **Streamlit Cloud pe nahi** chalega
- Single Photo mode **dono jagah** chalega
- Camera permission browser mein **Allow** karna zaruri hai
    """)

with st.expander("🏷️ All 80 COCO Classes"):
    cols = st.columns(5)
    for i, cls in enumerate(COCO_CLASSES.values()):
        cols[i % 5].markdown(f"<span class='tag'>{cls}</span>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div class="footer">
    🎭 Mask R-CNN Instance Segmentation &nbsp;·&nbsp;
    Built with Streamlit &nbsp;·&nbsp;
    Model: torchvision pretrained on MS-COCO 2017 &nbsp;·&nbsp;
    📷 Image + Webcam Support
</div>
""", unsafe_allow_html=True)
