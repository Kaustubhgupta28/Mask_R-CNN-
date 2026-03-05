# ═══════════════════════════════════════════════════════════════
#  app.py — Sirf Streamlit UI
#  Yeh main file hai jo Streamlit chalata hai.
#  AI logic → model.py se aata hai
#  Design   → styles.css se aata hai
#
#  Chalane ke liye:
#      streamlit run app.py
# ═══════════════════════════════════════════════════════════════

import streamlit as st
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
import requests
import pandas as pd
from io import BytesIO
import cv2
import os

# ── Apni files import karo ────────────────────────────────────
from model import (
    load_model, run_inference, draw_results,
    is_local, COCO_CLASSES, COLORS, SAMPLE_IMAGES, device
)
import torch

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="🎭 Mask R-CNN Instance Segmentation",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────────────────────
# NEURAL NETWORK ANIMATED CSS
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;900&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* ── DARK BACKGROUND ── */
.stApp {
    background: #020917 !important;
    color: #e2e8f0 !important;
}

/* ── NEURAL CANVAS ANIMATION ── */
#neural-canvas {
    position: fixed;
    top: 0; left: 0;
    width: 100vw; height: 100vh;
    z-index: 0;
    pointer-events: none;
}

/* ── MAIN CONTENT ABOVE CANVAS ── */
.main .block-container {
    position: relative;
    z-index: 1;
}

/* ── SIDEBAR ── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #040d1f 0%, #050f24 100%) !important;
    border-right: 1px solid rgba(0,212,255,0.2) !important;
}
section[data-testid="stSidebar"] * { color: #94a3b8 !important; }
section[data-testid="stSidebar"] h2, 
section[data-testid="stSidebar"] .sidebar-title { color: #00d4ff !important; }

/* ── HEADER 3D ── */
.neural-header {
    background: linear-gradient(135deg, rgba(0,20,50,0.95), rgba(5,15,40,0.95));
    border: 1px solid rgba(0,212,255,0.3);
    border-radius: 20px;
    padding: 36px 44px;
    margin-bottom: 24px;
    position: relative;
    overflow: hidden;
    transform-style: preserve-3d;
    box-shadow:
        0 0 40px rgba(0,212,255,0.15),
        0 0 80px rgba(0,100,255,0.08),
        inset 0 1px 0 rgba(0,212,255,0.2);
    animation: headerFloat 6s ease-in-out infinite;
}
@keyframes headerFloat {
    0%,100% { transform: translateY(0px) rotateX(0deg); box-shadow: 0 0 40px rgba(0,212,255,0.15), 0 20px 60px rgba(0,100,255,0.1); }
    50%      { transform: translateY(-6px) rotateX(1deg); box-shadow: 0 0 60px rgba(0,212,255,0.25), 0 30px 80px rgba(0,100,255,0.15); }
}
.neural-header::before {
    content: '';
    position: absolute;
    top: -50%; left: -50%;
    width: 200%; height: 200%;
    background: conic-gradient(transparent, rgba(0,212,255,0.03), transparent 30%);
    animation: rotate 8s linear infinite;
}
@keyframes rotate { to { transform: rotate(360deg); } }

.neural-header::after {
    content: '';
    position: absolute;
    top: 0; left: -100%;
    width: 60%; height: 100%;
    background: linear-gradient(90deg, transparent, rgba(0,212,255,0.05), transparent);
    animation: shimmer 4s ease-in-out infinite;
}
@keyframes shimmer { 0%{left:-100%} 100%{left:200%} }

/* ── TITLE ── */
.main-title {
    font-size: 2.8rem;
    font-weight: 900;
    background: linear-gradient(135deg, #00d4ff 0%, #7b2fff 50%, #00d4ff 100%);
    background-size: 200% auto;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    animation: titleShine 3s linear infinite;
    margin-bottom: 0;
    line-height: 1.2;
    text-shadow: none;
}
@keyframes titleShine { to { background-position: 200% center; } }
.subtitle { color: #64748b; font-size: 1rem; margin-top: 8px; }

/* ── GLOWING BADGES ── */
.badge {
    display: inline-block;
    background: rgba(0,212,255,0.08);
    color: #00d4ff;
    border: 1px solid rgba(0,212,255,0.3);
    border-radius: 999px;
    padding: 4px 14px;
    font-size: 0.78rem;
    font-weight: 700;
    margin: 3px;
    box-shadow: 0 0 10px rgba(0,212,255,0.15);
    animation: badgePulse 2s ease-in-out infinite;
}
@keyframes badgePulse {
    0%,100% { box-shadow: 0 0 8px rgba(0,212,255,0.15); }
    50%      { box-shadow: 0 0 18px rgba(0,212,255,0.4); }
}

/* ── GLOWING CARDS ── */
.metric-card {
    background: rgba(0,20,50,0.8);
    border-radius: 16px;
    padding: 20px 24px;
    border: 1px solid rgba(0,212,255,0.2);
    box-shadow: 0 0 20px rgba(0,212,255,0.08);
    text-align: center;
    margin: 6px 0;
    transition: all 0.3s ease;
    animation: cardGlow 3s ease-in-out infinite;
}
.metric-card:hover {
    border-color: rgba(0,212,255,0.6);
    box-shadow: 0 0 30px rgba(0,212,255,0.3);
    transform: translateY(-4px);
}
@keyframes cardGlow {
    0%,100% { box-shadow: 0 0 15px rgba(0,212,255,0.08); }
    50%      { box-shadow: 0 0 25px rgba(0,212,255,0.18); }
}
.metric-val { font-size: 2rem; font-weight: 800; color: #00d4ff; line-height: 1.2; }
.metric-lbl { font-size: 0.78rem; color: #475569; margin-top: 4px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; }

/* ── TABS ── */
div[data-testid="stTabs"] button {
    background: rgba(0,20,50,0.8) !important;
    border-radius: 10px 10px 0 0 !important;
    color: #64748b !important;
    font-weight: 600 !important;
    border: 1px solid rgba(0,212,255,0.15) !important;
    transition: all 0.3s !important;
}
div[data-testid="stTabs"] button:hover {
    color: #00d4ff !important;
    border-color: rgba(0,212,255,0.4) !important;
}
div[data-testid="stTabs"] button[aria-selected="true"] {
    background: linear-gradient(135deg, rgba(0,212,255,0.15), rgba(123,47,255,0.15)) !important;
    color: #00d4ff !important;
    border-color: rgba(0,212,255,0.5) !important;
    box-shadow: 0 0 15px rgba(0,212,255,0.2) !important;
}

/* ── FILE UPLOADER ── */
div[data-testid="stFileUploader"] {
    background: rgba(0,20,50,0.6) !important;
    border: 2px dashed rgba(0,212,255,0.3) !important;
    border-radius: 14px !important;
    transition: all 0.3s !important;
}
div[data-testid="stFileUploader"]:hover {
    border-color: rgba(0,212,255,0.7) !important;
    box-shadow: 0 0 20px rgba(0,212,255,0.15) !important;
}

/* ── EXPANDERS ── */
div[data-testid="stExpander"] {
    background: rgba(0,20,50,0.7) !important;
    border: 1px solid rgba(0,212,255,0.2) !important;
    border-radius: 12px !important;
    transition: all 0.3s !important;
}
div[data-testid="stExpander"]:hover {
    border-color: rgba(0,212,255,0.4) !important;
    box-shadow: 0 0 15px rgba(0,212,255,0.1) !important;
}

/* ── BUTTONS ── */
div[data-testid="stButton"] button[kind="primary"] {
    background: linear-gradient(135deg, #00d4ff, #7b2fff) !important;
    border: none !important;
    border-radius: 12px !important;
    font-weight: 700 !important;
    color: white !important;
    box-shadow: 0 0 20px rgba(0,212,255,0.4) !important;
    transition: all 0.3s !important;
    animation: btnPulse 2s ease-in-out infinite !important;
}
div[data-testid="stButton"] button[kind="primary"]:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 0 35px rgba(0,212,255,0.6) !important;
}
@keyframes btnPulse {
    0%,100% { box-shadow: 0 0 20px rgba(0,212,255,0.4); }
    50%      { box-shadow: 0 0 35px rgba(0,212,255,0.7); }
}

/* ── TEXT COLOR FIX ── */
.stMarkdown, p, label, .stText { color: #94a3b8 !important; }
h1, h2, h3, h4 { color: #e2e8f0 !important; }

/* ── SELECTBOX ── */
div[data-testid="stSelectbox"] > div > div {
    background: rgba(0,20,50,0.8) !important;
    border: 1px solid rgba(0,212,255,0.3) !important;
    border-radius: 10px !important;
    color: #e2e8f0 !important;
}

/* ── SLIDERS ── */
div[data-testid="stSlider"] div[role="slider"] {
    background: #00d4ff !important;
    box-shadow: 0 0 10px rgba(0,212,255,0.6) !important;
}

/* ── INFO BOX ── */
.info-box {
    background: rgba(0,212,255,0.06);
    border-left: 3px solid #00d4ff;
    border-radius: 10px;
    padding: 14px 18px;
    color: #94a3b8;
    margin: 10px 0;
    box-shadow: 0 0 15px rgba(0,212,255,0.08);
}
.cloud-box {
    background: rgba(0,20,50,0.8);
    border: 1px solid rgba(0,212,255,0.3);
    border-radius: 16px;
    padding: 30px;
    text-align: center;
    margin: 10px 0;
    box-shadow: 0 0 20px rgba(0,212,255,0.1);
}
.local-box {
    background: rgba(0,20,50,0.8);
    border: 2px dashed rgba(0,212,255,0.4);
    border-radius: 16px;
    padding: 20px;
    text-align: center;
    margin: 10px 0;
}

/* ── LIVE BADGE ── */
.live-badge {
    display: inline-block;
    background: #ef4444;
    color: white;
    border-radius: 999px;
    padding: 3px 14px;
    font-size: 0.75rem;
    font-weight: 700;
    box-shadow: 0 0 15px rgba(239,68,68,0.6);
    animation: blink 1.2s ease-in-out infinite;
}
@keyframes blink { 0%,100%{opacity:1} 50%{opacity:0.4} }

/* ── TAGS ── */
.tag {
    display: inline-block;
    background: rgba(0,212,255,0.08);
    color: #00d4ff;
    border: 1px solid rgba(0,212,255,0.25);
    border-radius: 6px;
    padding: 2px 10px;
    font-size: 0.78rem;
    font-weight: 600;
    margin: 2px;
}

/* ── FOOTER ── */
.footer {
    text-align: center;
    color: #334155;
    font-size: 0.8rem;
    padding: 20px 0;
}

/* ── FLOATING PARTICLES ── */
.particle {
    position: fixed;
    width: 3px; height: 3px;
    background: #00d4ff;
    border-radius: 50%;
    pointer-events: none;
    z-index: 0;
    animation: floatUp linear infinite;
    box-shadow: 0 0 6px #00d4ff;
}
@keyframes floatUp {
    0%   { transform: translateY(100vh) scale(0); opacity:0; }
    10%  { opacity: 1; }
    90%  { opacity: 0.6; }
    100% { transform: translateY(-10vh) scale(1); opacity: 0; }
}

/* ── DATAFRAME ── */
div[data-testid="stDataFrame"] {
    border: 1px solid rgba(0,212,255,0.2) !important;
    border-radius: 12px !important;
    overflow: hidden !important;
}
</style>

<!-- Neural Network Canvas -->
<canvas id="neural-canvas"></canvas>

<!-- Floating Particles -->
<div id="particles-container"></div>

<script>
// ── Neural Network Animation ──────────────────────────────
(function() {
    const canvas = document.getElementById('neural-canvas');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    
    function resize() {
        canvas.width = window.innerWidth;
        canvas.height = window.innerHeight;
    }
    resize();
    window.addEventListener('resize', resize);

    const nodes = Array.from({length: 40}, () => ({
        x: Math.random() * canvas.width,
        y: Math.random() * canvas.height,
        vx: (Math.random() - 0.5) * 0.4,
        vy: (Math.random() - 0.5) * 0.4,
        r: Math.random() * 2.5 + 1.5,
        pulse: Math.random() * Math.PI * 2
    }));

    function draw() {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        
        nodes.forEach(n => {
            n.x += n.vx;
            n.y += n.vy;
            n.pulse += 0.02;
            if (n.x < 0 || n.x > canvas.width)  n.vx *= -1;
            if (n.y < 0 || n.y > canvas.height) n.vy *= -1;
        });

        // Draw connections
        nodes.forEach((a, i) => {
            nodes.slice(i + 1).forEach(b => {
                const d = Math.hypot(a.x - b.x, a.y - b.y);
                if (d < 130) {
                    const alpha = Math.floor((1 - d / 130) * 60).toString(16).padStart(2, '0');
                    ctx.beginPath();
                    ctx.strokeStyle = '#00d4ff' + alpha;
                    ctx.lineWidth = 0.6;
                    ctx.moveTo(a.x, a.y);
                    ctx.lineTo(b.x, b.y);
                    ctx.stroke();
                }
            });
        });

        // Draw nodes
        nodes.forEach(n => {
            const glow = Math.sin(n.pulse) * 0.5 + 0.5;
            ctx.beginPath();
            ctx.arc(n.x, n.y, n.r + glow, 0, Math.PI * 2);
            ctx.fillStyle = '#00d4ff';
            ctx.shadowBlur = 12 + glow * 8;
            ctx.shadowColor = '#00d4ff';
            ctx.fill();
        });

        requestAnimationFrame(draw);
    }
    draw();
})();

// ── Floating Particles ────────────────────────────────────
(function() {
    const container = document.getElementById('particles-container');
    if (!container) return;
    
    for (let i = 0; i < 25; i++) {
        const p = document.createElement('div');
        p.className = 'particle';
        p.style.left = Math.random() * 100 + 'vw';
        p.style.animationDuration = (Math.random() * 10 + 8) + 's';
        p.style.animationDelay = (Math.random() * 10) + 's';
        p.style.width = p.style.height = (Math.random() * 3 + 1) + 'px';
        p.style.opacity = Math.random() * 0.6 + 0.2;
        container.appendChild(p);
    }
})();
</script>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# CHECK LOCAL / CLOUD
# ─────────────────────────────────────────────────────────────
RUNNING_LOCAL = is_local()

# ─────────────────────────────────────────────────────────────
# RESULT DETAILS — charts + table (reusable)
# ─────────────────────────────────────────────────────────────
def show_result_details(results, n_det, score_thr):
    if n_det == 0:
        st.warning("⚠️ Koi object detect nahi hua. Sidebar mein Confidence Threshold kam karo.")
        return

    # ── 4 Metric Cards ────────────────────────────────────────
    mc1, mc2, mc3, mc4 = st.columns(4)
    mc1.markdown(f'<div class="metric-card"><div class="metric-val">{n_det}</div><div class="metric-lbl">Objects Detected</div></div>', unsafe_allow_html=True)
    mc2.markdown(f'<div class="metric-card"><div class="metric-val">{results["time"]*1000:.0f}ms</div><div class="metric-lbl">Inference Time</div></div>', unsafe_allow_html=True)
    mc3.markdown(f'<div class="metric-card"><div class="metric-val">{score_thr:.2f}</div><div class="metric-lbl">Confidence Threshold</div></div>', unsafe_allow_html=True)
    mc4.markdown(f'<div class="metric-card"><div class="metric-val">{str(device).upper()}</div><div class="metric-lbl">Device</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Detection Table ───────────────────────────────────────
    st.markdown("#### 📋 Detection Details")
    rows = []
    for i in range(n_det):
        lid   = int(results['labels'][i])
        lname = COCO_CLASSES.get(lid, f'class_{lid}')
        score = float(results['scores'][i])
        box   = results['boxes'][i].astype(int)
        rows.append({
            "#":          i + 1,
            "Class":      lname,
            "Score":      f"{score:.3f}",
            "Width px":   int(box[2] - box[0]),
            "Height px":  int(box[3] - box[1]),
            "Box [x1,y1,x2,y2]": f"[{box[0]},{box[1]},{box[2]},{box[3]}]"
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # ── Charts ────────────────────────────────────────────────
    ch1, ch2 = st.columns(2)

    with ch1:
        st.markdown("#### 📊 Class Distribution")
        class_counts = {}
        for i in range(n_det):
            name = COCO_CLASSES.get(int(results['labels'][i]), '?')
            class_counts[name] = class_counts.get(name, 0) + 1
        fig, ax = plt.subplots(figsize=(5, 3.5))
        fig.patch.set_facecolor('#fff8fd')
        bars = ax.bar(class_counts.keys(), class_counts.values(),
                      color='#c084fc', edgecolor='white', linewidth=0.8)
        for bar, val in zip(bars, class_counts.values()):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.05, str(val),
                    ha='center', va='bottom', fontweight='bold', color='#6d28d9')
        ax.set_facecolor('#fff8fd')
        ax.set_ylabel("Count", color='#9d4edd')
        ax.set_title("Instances per Class", fontweight='bold', color='#6d28d9')
        ax.set_ylim(0, max(class_counts.values()) + 1.5)
        plt.xticks(rotation=30, ha='right', fontsize=9, color='#9d4edd')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with ch2:
        if n_det > 1:
            st.markdown("#### 📉 Confidence Scores")
            scores_list = results['scores'].tolist()
            labels_list = [COCO_CLASSES.get(int(l), '?') for l in results['labels']]
            colors_bar  = [
                f'#{COLORS[i % len(COLORS)][0]:02x}'
                f'{COLORS[i % len(COLORS)][1]:02x}'
                f'{COLORS[i % len(COLORS)][2]:02x}'
                for i in range(n_det)
            ]
            fig2, ax2 = plt.subplots(figsize=(5, 3.5))
            fig2.patch.set_facecolor('#fff8fd')
            ax2.set_facecolor('#fff8fd')
            ax2.barh(range(n_det), scores_list, color=colors_bar, edgecolor='white')
            ax2.set_yticks(range(n_det))
            ax2.set_yticklabels(
                [f"#{i+1} {labels_list[i]}" for i in range(n_det)], fontsize=8)
            ax2.set_xlabel("Confidence Score", color='#9d4edd')
            ax2.set_xlim(0, 1.05)
            ax2.axvline(score_thr, color='#e879f9', linestyle='--',
                        linewidth=1.5, label=f'Threshold={score_thr}')
            ax2.legend(fontsize=8)
            ax2.set_title("Per-Instance Confidence", fontweight='bold', color='#6d28d9')
            plt.tight_layout()
            st.pyplot(fig2)
            plt.close()

# ─────────────────────────────────────────────────────────────
# HEADER — Glass Morphism
# ─────────────────────────────────────────────────────────────
env_icon  = "🖥️ Local Mode" if RUNNING_LOCAL else "☁️ Cloud Mode"
env_color = "#16a34a"       if RUNNING_LOCAL else "#7c3aed"

# HEADER — 3D Neural Style
env_icon  = "🖥️ Local Mode" if RUNNING_LOCAL else "☁️ Cloud Mode"
st.markdown(f"""
<div class="neural-header">
    <p class="main-title">🎭 Mask R-CNN</p>
    <p class="main-title" style="font-size:1.5rem; margin-top:-6px;">Instance Segmentation</p>
    <p class="subtitle">ResNet-50 FPN &nbsp;·&nbsp; MS-COCO 2017 Pretrained &nbsp;·&nbsp; 80 Object Categories &nbsp;·&nbsp; Image + Webcam</p>
    <div style="margin-top:16px; display:flex; gap:8px; flex-wrap:wrap;">
        <span class="badge">🧠 ResNet-50 FPN</span>
        <span class="badge">📦 MS-COCO 2017</span>
        <span class="badge">🏷️ 80 Categories</span>
        <span class="badge">{env_icon}</span>
    </div>
</div>
""", unsafe_allow_html=True)
st.markdown("---")

# ─────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    score_thr = st.slider("🎯 Confidence Threshold", 0.10, 0.95, 0.50, 0.05)
    mask_thr  = st.slider("🎨 Mask Threshold",       0.10, 0.90, 0.50, 0.05)
    alpha     = st.slider("🌫️ Mask Transparency",    0.10, 0.90, 0.45, 0.05)

    st.markdown("---")
    st.markdown("## 👁️ Display")
    show_masks  = st.checkbox("Show Masks",  value=True)
    show_boxes  = st.checkbox("Show Boxes",  value=True)
    show_labels = st.checkbox("Show Labels", value=True)

    st.markdown("---")
    st.markdown("## 🖥️ System")
    st.markdown(f"**Device:** `{str(device).upper()}`")
    st.markdown(f"**PyTorch:** `{torch.__version__}`")
    st.markdown(f"**Mode:** {'🖥️ Local' if RUNNING_LOCAL else '☁️ Cloud'}")
    if torch.cuda.is_available():
        st.success(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        st.info("CPU mode")

# ─────────────────────────────────────────────────────────────
# INPUT TABS
# ─────────────────────────────────────────────────────────────
st.markdown("### 📥 Input Source Choose Karo")

tab1, tab2, tab3, tab4 = st.tabs([
    "📁 Upload Image",
    "🔗 Image URL",
    "🖼️ Sample Images",
    "📷 Webcam",
])

pil_image = None

# ── TAB 1: Upload ─────────────────────────────────────────────
with tab1:
    uploaded = st.file_uploader("JPG / PNG upload karo", type=["jpg","jpeg","png"])
    if uploaded:
        pil_image = Image.open(uploaded).convert("RGB")
        st.success(f"✅ {uploaded.name} ({pil_image.size[0]}×{pil_image.size[1]}px)")
        st.image(pil_image, use_container_width=True)

# ── TAB 2: URL ────────────────────────────────────────────────
with tab2:
    url = st.text_input("Image URL paste karo:", placeholder="https://example.com/photo.jpg")
    if url:
        try:
            with st.spinner("Downloading..."):
                r = requests.get(url, timeout=10)
                pil_image = Image.open(BytesIO(r.content)).convert("RGB")
            st.success(f"✅ Downloaded! ({pil_image.size[0]}×{pil_image.size[1]}px)")
            st.image(pil_image, use_container_width=True)
        except Exception as e:
            st.error(f"❌ Load nahi hui: {e}")

# ── TAB 3: Sample ─────────────────────────────────────────────
with tab3:
    choice = st.selectbox("Sample choose karo:", list(SAMPLE_IMAGES.keys()))
    if st.button("📥 Load Sample", use_container_width=True):
        try:
            with st.spinner("Loading..."):
                r = requests.get(SAMPLE_IMAGES[choice], timeout=10)
                pil_image = Image.open(BytesIO(r.content)).convert("RGB")
            st.success(f"✅ Loaded! ({pil_image.size[0]}×{pil_image.size[1]}px)")
            st.image(pil_image, use_container_width=True)
        except Exception as e:
            st.error(f"❌ Failed: {e}")

# ── TAB 4: WEBCAM ─────────────────────────────────────────────
with tab4:

    # ══ CLOUD MODE — sirf Single Photo ════════════════════════
    if not RUNNING_LOCAL:
        st.markdown("""
        <div class="cloud-box">
            <h2>📸 Webcam Photo Mode</h2>
            <p style="color:#6d28d9; font-size:1rem;">
                Camera se photo lo — automatic object detection hoga!<br>
                <b>Cloud pe Live Video nahi chalta</b> — local guide neeche hai.
            </p>
        </div>
        """, unsafe_allow_html=True)

        camera_img = st.camera_input("📷 Photo lo — detection automatic hoga!")

        if camera_img is not None:
            pil_cam = Image.open(camera_img).convert("RGB")
            st.success(f"✅ Photo li gayi! ({pil_cam.size[0]}×{pil_cam.size[1]}px)")

            with st.spinner("🧠 Model load ho raha hai..."):
                model = load_model()
            with st.spinner("⚡ Objects detect ho rahe hain..."):
                cam_results = run_inference(model, pil_cam, score_thr)
            with st.spinner("🎨 Drawing results..."):
                cam_output, cam_n = draw_results(
                    pil_cam, cam_results,
                    mask_thr=mask_thr, show_masks=show_masks,
                    show_boxes=show_boxes, show_labels=show_labels, alpha=alpha
                )

            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**📷 Original Photo**")
                st.image(pil_cam, use_container_width=True)
            with c2:
                st.markdown(f"**🎭 Detected — {cam_n} objects**")
                st.image(cam_output, use_container_width=True)

            buf = BytesIO()
            Image.fromarray(cam_output).save(buf, format="PNG")
            st.download_button("⬇️ Download Result", buf.getvalue(),
                               "webcam_result.png", "image/png",
                               use_container_width=True)
            st.markdown("---")
            show_result_details(cam_results, cam_n, score_thr)

        st.markdown("---")
        with st.expander("🎥 Live Video kaise chalayein? (Local Guide)"):
            st.markdown("""
            ### Live Video sirf local computer pe kaam karta hai

            #### Step 1 — Libraries install karo
            ```bash
            pip install streamlit torch torchvision opencv-python pillow requests pandas matplotlib
            ```
            #### Step 2 — GitHub se teeno files download karo
            - `app.py`
            - `model.py`
            - `styles.css`

            #### Step 3 — Command chalao
            ```bash
            streamlit run app.py
            ```
            #### Step 4
            Browser mein `http://localhost:8501` khulega → **Webcam → Live Video → ▶️ Start**
            """)

    # ══ LOCAL MODE — Photo + Live Video dono ══════════════════
    else:
        st.markdown("""
        <div class="info-box">
        🖥️ <b>Local mode detect hua!</b> Dono webcam modes available hain.
        </div>
        """, unsafe_allow_html=True)

        webcam_mode = st.radio(
            "Mode choose karo:",
            ["📸 Single Photo", "🎥 Live Video"],
            horizontal=True
        )

        # ── Single Photo ──────────────────────────────────────
        if webcam_mode == "📸 Single Photo":
            camera_img = st.camera_input("📷 Photo lo!")
            if camera_img is not None:
                pil_cam = Image.open(camera_img).convert("RGB")
                st.success(f"✅ Photo li gayi! ({pil_cam.size[0]}×{pil_cam.size[1]}px)")
                with st.spinner("🧠 Loading model..."):
                    model = load_model()
                with st.spinner("⚡ Detecting..."):
                    cam_results = run_inference(model, pil_cam, score_thr)
                with st.spinner("🎨 Drawing..."):
                    cam_output, cam_n = draw_results(
                        pil_cam, cam_results,
                        mask_thr=mask_thr, show_masks=show_masks,
                        show_boxes=show_boxes, show_labels=show_labels, alpha=alpha
                    )
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown("**📷 Original**")
                    st.image(pil_cam, use_container_width=True)
                with c2:
                    st.markdown(f"**🎭 Detected — {cam_n} objects**")
                    st.image(cam_output, use_container_width=True)
                buf = BytesIO()
                Image.fromarray(cam_output).save(buf, format="PNG")
                st.download_button("⬇️ Download", buf.getvalue(),
                                   "photo_result.png", "image/png",
                                   use_container_width=True)
                st.markdown("---")
                show_result_details(cam_results, cam_n, score_thr)

        # ── Live Video ────────────────────────────────────────
        else:
            st.markdown("""
            <div class="local-box">
                <h3>🎥 Live Video Detection</h3>
                <p style="color:#6d28d9;">Webcam se real-time object detection</p>
            </div>
            """, unsafe_allow_html=True)

            col1, col2 = st.columns(2)
            start_btn = col1.button("▶️ Start Live Detection",
                                    type="primary", use_container_width=True)
            stop_btn  = col2.button("⏹️ Stop", use_container_width=True)

            if 'live_on' not in st.session_state:
                st.session_state.live_on = False
            if start_btn:
                st.session_state.live_on = True
            if stop_btn:
                st.session_state.live_on = False

            if st.session_state.live_on:
                st.markdown('<span class="live-badge">● LIVE</span>', unsafe_allow_html=True)
                frame_ph = st.empty()
                stats_ph = st.empty()

                with st.spinner("🧠 Model load ho raha hai..."):
                    model = load_model()

                cap = cv2.VideoCapture(0)
                if not cap.isOpened():
                    st.error("❌ Webcam nahi khula!")
                    st.session_state.live_on = False
                else:
                    st.success("✅ Webcam connected!")
                    frame_count = 0
                    while st.session_state.live_on:
                        ret, frame = cap.read()
                        if not ret:
                            break
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        pil_f = Image.fromarray(frame_rgb)
                        res = run_inference(model, pil_f, score_thr)
                        out_f, nd = draw_results(
                            pil_f, res,
                            mask_thr=mask_thr, show_masks=show_masks,
                            show_boxes=show_boxes, show_labels=show_labels, alpha=alpha
                        )
                        frame_ph.image(out_f,
                            caption=f"🎥 Frame #{frame_count} — {nd} objects",
                            use_container_width=True)
                        detected = list(set([
                            COCO_CLASSES.get(int(l), '?') for l in res['labels']
                        ]))
                        stats_ph.markdown(f"""
                        | | |
                        |---|---|
                        | 🎯 Objects | **{nd}** |
                        | ⚡ Time | **{res['time']*1000:.0f}ms** |
                        | 📸 Frame | **#{frame_count}** |
                        | 🏷️ Classes | **{', '.join(detected) if detected else 'None'}** |
                        """)
                        frame_count += 1
                        if not st.session_state.live_on:
                            break
                    cap.release()
                    st.info("⏹️ Live detection band ho gayi.")
            else:
                st.markdown("""
                <div style='text-align:center; padding:50px; background:#fdf4ff;
                            border-radius:16px; border:2px dashed #e9d5ff;'>
                    <h2 style='color:#c084fc;'>🎥</h2>
                    <p style='color:#9d4edd; font-size:1.1rem;'>
                        <b>▶️ Start Live Detection</b> dabao<br>
                        real-time webcam detection shuru ho jaayegi!
                    </p>
                </div>
                """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# RUN BUTTON — Upload / URL / Sample ke liye
# ─────────────────────────────────────────────────────────────
if pil_image is not None:
    st.markdown("---")
    st.markdown("### 🔮 Step 2 — Run Detection")

    if st.button("🚀 Run Mask R-CNN", type="primary", use_container_width=True):
        with st.spinner("🧠 Model load ho raha hai..."):
            model = load_model()
        with st.spinner("⚡ Inference chal rahi hai..."):
            results = run_inference(model, pil_image, score_thr)
        with st.spinner("🎨 Results draw ho rahe hain..."):
            output_img, n_det = draw_results(
                pil_image, results,
                mask_thr=mask_thr, show_masks=show_masks,
                show_boxes=show_boxes, show_labels=show_labels, alpha=alpha
            )

        st.markdown("---")
        left, right = st.columns(2)
        with left:
            st.markdown("**🖼️ Original Image**")
            st.image(pil_image, use_container_width=True)
        with right:
            st.markdown(f"**🎭 Result — {n_det} instances**")
            st.image(output_img, use_container_width=True)

        buf = BytesIO()
        Image.fromarray(output_img).save(buf, format="PNG")
        st.download_button("⬇️ Download Result", buf.getvalue(),
                           "mask_rcnn_output.png", "image/png",
                           use_container_width=True)
        st.markdown("---")
        show_result_details(results, n_det, score_thr)

# ─────────────────────────────────────────────────────────────
# ABOUT SECTION
# ─────────────────────────────────────────────────────────────
st.markdown("---")
c1, c2 = st.columns(2)

with c1:
    with st.expander("ℹ️ Mask R-CNN kya hai?"):
        st.markdown("""
**Mask R-CNN** ek deep learning model hai jo:
- Objects ko **detect** karta hai (80 categories)
- Har object ke around **bounding box** draw karta hai
- Har object ka **pixel-level mask** banata hai

**Pipeline:**
1. ResNet-50 → features extract karta hai
2. FPN → multi-scale features combine karta hai
3. RPN → object regions suggest karta hai
4. RoI Align → features pool karta hai
5. 3 Heads → class + box + mask predict karte hain

**Paper:** He et al., *Mask R-CNN*, ICCV 2017
        """)

with c2:
    with st.expander("📦 Model Info"):
        st.markdown(f"""
| Property | Value |
|---|---|
| Model | Mask R-CNN |
| Backbone | ResNet-50 + FPN |
| Pretrained | MS-COCO 2017 |
| Categories | 80 |
| Parameters | ~44 million |
| Device | `{str(device).upper()}` |
| Webcam Cloud | ✅ Single Photo |
| Webcam Local | ✅ Photo + Live Video |
        """)

with st.expander("🏷️ All 80 COCO Classes"):
    cols = st.columns(5)
    for i, cls in enumerate(COCO_CLASSES.values()):
        cols[i % 5].markdown(f"<span class='tag'>{cls}</span>",
                             unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div class="footer">
    🎭 Mask R-CNN Instance Segmentation &nbsp;·&nbsp;
    Built with Streamlit &nbsp;·&nbsp;
    MS-COCO 2017 Pretrained &nbsp;·&nbsp;
    📷 Cloud Photo + 🎥 Local Live Video
</div>
""", unsafe_allow_html=True)
