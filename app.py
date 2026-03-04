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
# CSS LOAD — styles.css se
# ─────────────────────────────────────────────────────────────
def load_css():
    css_path = os.path.join(os.path.dirname(__file__), "styles.css")
    with open(css_path, "r") as f:
        css = f.read()
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)

load_css()

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

st.markdown(f"""
<div class="glass-header">
    <p class="main-title">🎭 Mask R-CNN</p>
    <p class="main-title" style="font-size:1.6rem; margin-top:-8px;">
        Instance Segmentation
    </p>
    <p class="subtitle">
        ResNet-50 FPN &nbsp;·&nbsp; MS-COCO 2017 Pretrained
        &nbsp;·&nbsp; 80 Object Categories &nbsp;·&nbsp; Image + Webcam
    </p>
    <div style="margin-top: 16px; display: flex; gap: 10px; flex-wrap: wrap;">
        <span style="background: rgba(168,85,247,0.15); color: #7c3aed;
                     border: 1px solid rgba(168,85,247,0.3);
                     border-radius: 999px; padding: 4px 14px;
                     font-size: 0.78rem; font-weight: 700;">
            🧠 ResNet-50 FPN
        </span>
        <span style="background: rgba(236,72,153,0.12); color: #be185d;
                     border: 1px solid rgba(236,72,153,0.25);
                     border-radius: 999px; padding: 4px 14px;
                     font-size: 0.78rem; font-weight: 700;">
            📦 MS-COCO 2017
        </span>
        <span style="background: rgba(129,140,248,0.15); color: #4338ca;
                     border: 1px solid rgba(129,140,248,0.3);
                     border-radius: 999px; padding: 4px 14px;
                     font-size: 0.78rem; font-weight: 700;">
            🏷️ 80 Categories
        </span>
        <span style="background: rgba(34,197,94,0.12); color: {env_color};
                     border: 1px solid rgba(34,197,94,0.25);
                     border-radius: 999px; padding: 4px 14px;
                     font-size: 0.78rem; font-weight: 700;">
            {env_icon}
        </span>
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
