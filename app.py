# ═══════════════════════════════════════════════════════════════
#  app.py — Mask R-CNN Instance Segmentation
#  Run: streamlit run app.py
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
import tempfile
import time

from model import (
    load_model, run_inference, draw_results,
    is_local, COCO_CLASSES, COLORS, SAMPLE_IMAGES, device
)
import torch

st.set_page_config(
    page_title="🎭 Mask R-CNN Instance Segmentation",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Session State Init ────────────────────────────────────────
for key, val in [('pil_image',None),('result_img',None),('results',None),('n_det',0),('live_on',False)]:
    if key not in st.session_state:
        st.session_state[key] = val

# ── CSS + JS ──────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;900&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp { background: #020917 !important; color: #e2e8f0 !important; }
#neural-canvas { position:fixed;top:0;left:0;width:100vw;height:100vh;z-index:0;pointer-events:none; }
.main .block-container { position:relative;z-index:1;transform-style:preserve-3d;will-change:transform; }
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg,#040d1f 0%,#050f24 100%) !important;
    border-right: 1px solid rgba(0,212,255,0.2) !important;
    box-shadow: 4px 0 24px rgba(0,0,0,0.6),8px 0 48px rgba(0,0,0,0.3),inset -1px 0 0 rgba(0,212,255,0.08) !important;
}
section[data-testid="stSidebar"]::before {
    content:'';position:absolute;top:0;left:0;right:0;height:2px;
    background:linear-gradient(90deg,transparent,rgba(0,212,255,0.5),transparent);pointer-events:none;
}
section[data-testid="stSidebar"] * { color:#94a3b8 !important; }
section[data-testid="stSidebar"] h2 { color:#00d4ff !important; }
.neural-header {
    background:linear-gradient(135deg,rgba(0,20,50,0.95),rgba(5,15,40,0.95));
    border:1px solid rgba(0,212,255,0.3);border-radius:20px;padding:36px 44px;margin-bottom:24px;
    position:relative;overflow:hidden;
    box-shadow:0 0 40px rgba(0,212,255,0.15),0 0 80px rgba(0,100,255,0.08),inset 0 1px 0 rgba(0,212,255,0.2);
    animation:headerFloat 6s ease-in-out infinite;
}
@keyframes headerFloat {
    0%,100%{transform:translateY(0px);box-shadow:0 0 40px rgba(0,212,255,0.15),0 20px 60px rgba(0,100,255,0.1);}
    50%{transform:translateY(-6px);box-shadow:0 0 60px rgba(0,212,255,0.25),0 30px 80px rgba(0,100,255,0.15);}
}
.neural-header::before {
    content:'';position:absolute;top:-50%;left:-50%;width:200%;height:200%;
    background:conic-gradient(transparent,rgba(0,212,255,0.03),transparent 30%);
    animation:rotate 8s linear infinite;
}
@keyframes rotate{to{transform:rotate(360deg);}}
.neural-header::after {
    content:'';position:absolute;top:0;left:-100%;width:60%;height:100%;
    background:linear-gradient(90deg,transparent,rgba(0,212,255,0.05),transparent);
    animation:shimmer 4s ease-in-out infinite;
}
@keyframes shimmer{0%{left:-100%}100%{left:200%}}
.main-title {
    font-size:2.8rem;font-weight:900;
    background:linear-gradient(135deg,#00d4ff 0%,#7b2fff 50%,#00d4ff 100%);
    background-size:200% auto;-webkit-background-clip:text;-webkit-text-fill-color:transparent;
    animation:titleShine 3s linear infinite;margin-bottom:0;line-height:1.2;
}
@keyframes titleShine{to{background-position:200% center;}}
.subtitle{color:#64748b;font-size:1rem;margin-top:8px;}
.badge {
    display:inline-block;background:rgba(0,212,255,0.08);color:#00d4ff;
    border:1px solid rgba(0,212,255,0.3);border-radius:999px;
    padding:4px 14px;font-size:0.78rem;font-weight:700;margin:3px;
    animation:badgePulse 2s ease-in-out infinite;
}
@keyframes badgePulse{0%,100%{box-shadow:0 0 8px rgba(0,212,255,0.15);}50%{box-shadow:0 0 18px rgba(0,212,255,0.4);}}
.metric-card {
    background:rgba(0,20,50,0.8);border-radius:16px;padding:20px 24px;
    border:1px solid rgba(0,212,255,0.2);text-align:center;margin:6px 0;
    position:relative;overflow:hidden;transform-style:preserve-3d;
    transition:transform 0.35s ease,box-shadow 0.35s ease;will-change:transform;
    animation:cardGlow 3s ease-in-out infinite;
}
.metric-card::after {
    content:'';position:absolute;inset:0;border-radius:16px;
    background:linear-gradient(135deg,rgba(255,255,255,0.06) 0%,transparent 50%,rgba(0,212,255,0.04) 100%);
    pointer-events:none;
}
.metric-card:hover {
    transform:translateY(-6px) rotateX(6deg) rotateY(-3deg);
    box-shadow:0 20px 40px rgba(0,0,0,0.4),0 0 30px rgba(0,212,255,0.25),inset 0 1px 0 rgba(0,212,255,0.3) !important;
}
@keyframes cardGlow{0%,100%{box-shadow:0 0 15px rgba(0,212,255,0.08);}50%{box-shadow:0 0 25px rgba(0,212,255,0.18);}}
.metric-val{font-size:2rem;font-weight:800;color:#00d4ff;line-height:1.2;}
.metric-lbl{font-size:0.78rem;color:#475569;margin-top:4px;font-weight:600;text-transform:uppercase;letter-spacing:0.05em;}
div[data-testid="stTabs"] button {
    background:rgba(0,20,50,0.8) !important;border-radius:10px 10px 0 0 !important;
    color:#64748b !important;font-weight:600 !important;border:1px solid rgba(0,212,255,0.15) !important;
    box-shadow:0 4px 8px rgba(0,0,0,0.35),inset 0 1px 0 rgba(0,212,255,0.1) !important;transition:all 0.2s ease !important;
}
div[data-testid="stTabs"] button:hover {
    color:#00d4ff !important;transform:translateY(-2px) !important;
    box-shadow:0 8px 16px rgba(0,0,0,0.4),0 0 12px rgba(0,212,255,0.2) !important;
}
div[data-testid="stTabs"] button[aria-selected="true"] {
    background:linear-gradient(135deg,rgba(0,212,255,0.15),rgba(123,47,255,0.15)) !important;
    color:#00d4ff !important;transform:translateY(-3px) !important;
    box-shadow:0 8px 20px rgba(0,0,0,0.45),0 0 20px rgba(0,212,255,0.25),inset 0 1px 0 rgba(0,212,255,0.4) !important;
}
div[data-testid="stFileUploader"] {
    background:rgba(0,20,50,0.6) !important;border:2px dashed rgba(0,212,255,0.3) !important;
    border-radius:14px !important;box-shadow:0 4px 20px rgba(0,0,0,0.4),inset 0 1px 0 rgba(0,212,255,0.06) !important;transition:all 0.3s !important;
}
div[data-testid="stFileUploader"]:hover{border-color:rgba(0,212,255,0.7) !important;box-shadow:0 0 20px rgba(0,212,255,0.15) !important;}
div[data-testid="stFileUploader"] > div,div[data-testid="stFileUploader"] > div > div,
div[data-testid="stFileUploader"] section,div[data-testid="stFileUploader"] section > div {
    background:rgba(0,15,40,0.9) !important;background-color:rgba(0,15,40,0.9) !important;border-radius:10px !important;border:none !important;
}
div[data-testid="stFileUploader"] span,div[data-testid="stFileUploader"] p,div[data-testid="stFileUploader"] small{color:#64748b !important;}
div[data-testid="stFileUploader"] button {
    background:linear-gradient(135deg,rgba(0,212,255,0.15),rgba(123,47,255,0.15)) !important;
    color:#00d4ff !important;border:1px solid rgba(0,212,255,0.4) !important;border-radius:8px !important;
}
div[data-testid="stExpander"] {
    background:rgba(0,20,50,0.7) !important;border:1px solid rgba(0,212,255,0.2) !important;border-radius:12px !important;
    box-shadow:0 4px 16px rgba(0,0,0,0.35),inset 0 1px 0 rgba(0,212,255,0.08) !important;transition:transform 0.25s ease,box-shadow 0.25s ease !important;
}
div[data-testid="stExpander"]:hover{transform:translateY(-2px) !important;border-color:rgba(0,212,255,0.4) !important;box-shadow:0 8px 24px rgba(0,0,0,0.45),0 0 16px rgba(0,212,255,0.12) !important;}
div[data-testid="stButton"] button[kind="primary"] {
    background:linear-gradient(135deg,#00d4ff,#7b2fff) !important;border:none !important;border-radius:12px !important;
    font-weight:700 !important;color:white !important;
    box-shadow:0 6px 0 rgba(0,100,150,0.6),0 8px 16px rgba(0,0,0,0.4),0 0 20px rgba(0,212,255,0.4) !important;
    transition:transform 0.1s ease,box-shadow 0.1s ease !important;
}
div[data-testid="stButton"] button[kind="primary"]:hover{transform:translateY(-3px) !important;box-shadow:0 9px 0 rgba(0,100,150,0.5),0 12px 24px rgba(0,0,0,0.5),0 0 35px rgba(0,212,255,0.6) !important;}
div[data-testid="stButton"] button[kind="primary"]:active{transform:translateY(4px) !important;box-shadow:0 2px 0 rgba(0,100,150,0.6),0 2px 6px rgba(0,0,0,0.3),0 0 10px rgba(0,212,255,0.3) !important;animation:none !important;}
div[data-testid="stButton"] button:not([kind="primary"]){box-shadow:0 4px 0 rgba(0,50,80,0.7),0 6px 12px rgba(0,0,0,0.3) !important;border-radius:10px !important;transition:transform 0.1s,box-shadow 0.1s !important;}
div[data-testid="stButton"] button:not([kind="primary"]):hover{transform:translateY(-2px) !important;box-shadow:0 6px 0 rgba(0,50,80,0.6),0 10px 20px rgba(0,0,0,0.3) !important;}
div[data-testid="stButton"] button:not([kind="primary"]):active{transform:translateY(3px) !important;box-shadow:0 1px 0 rgba(0,50,80,0.7) !important;}
.stMarkdown,p,label,.stText{color:#94a3b8 !important;}
h1,h2,h3,h4{color:#e2e8f0 !important;}
div[data-testid="stSelectbox"] > div > div{background:rgba(0,20,50,0.8) !important;border:1px solid rgba(0,212,255,0.3) !important;border-radius:10px !important;color:#e2e8f0 !important;}
div[data-testid="stSlider"] div[role="slider"]{background:#00d4ff !important;box-shadow:0 0 10px rgba(0,212,255,0.6) !important;}
.info-box{background:rgba(0,212,255,0.06);border-left:3px solid #00d4ff;border-radius:10px;padding:14px 18px;color:#94a3b8;margin:10px 0;box-shadow:0 0 15px rgba(0,212,255,0.08);}
.warn-box{background:rgba(255,170,0,0.06);border-left:3px solid #ffaa00;border-radius:10px;padding:14px 18px;color:#94a3b8;margin:10px 0;}
.cloud-box{background:rgba(0,20,50,0.8);border:1px solid rgba(0,212,255,0.3);border-radius:16px;padding:30px;text-align:center;margin:10px 0;box-shadow:0 0 20px rgba(0,212,255,0.1);}
.local-box{background:rgba(0,20,50,0.8);border:2px dashed rgba(0,212,255,0.4);border-radius:16px;padding:20px;text-align:center;margin:10px 0;}
.live-badge{display:inline-block;background:#ef4444;color:white;border-radius:999px;padding:3px 14px;font-size:0.75rem;font-weight:700;box-shadow:0 0 15px rgba(239,68,68,0.6);animation:blink 1.2s ease-in-out infinite;}
@keyframes blink{0%,100%{opacity:1}50%{opacity:0.4}}
.tag{display:inline-block;background:rgba(0,212,255,0.08);color:#00d4ff;border:1px solid rgba(0,212,255,0.25);border-radius:6px;padding:2px 10px;font-size:0.78rem;font-weight:600;margin:2px;}
.footer{text-align:center;color:#334155;font-size:0.8rem;padding:20px 0;}
.particle{position:fixed;width:3px;height:3px;background:#00d4ff;border-radius:50%;pointer-events:none;z-index:0;animation:floatUp linear infinite;box-shadow:0 0 6px #00d4ff;}
@keyframes floatUp{0%{transform:translateY(100vh) scale(0);opacity:0;}10%{opacity:1;}90%{opacity:0.6;}100%{transform:translateY(-10vh) scale(1);opacity:0;}}
div[data-testid="stDataFrame"]{border:1px solid rgba(0,212,255,0.2) !important;border-radius:12px !important;overflow:hidden !important;}
.slider-desc{background:rgba(0,212,255,0.04);border-left:2px solid rgba(0,212,255,0.35);border-radius:0 6px 6px 0;padding:7px 11px;margin:-4px 0 14px 0;font-size:0.76rem;color:#4e6a85;line-height:1.55;}
.slider-desc b{color:#00a8cc;}
</style>
<canvas id="neural-canvas"></canvas>
<div id="particles-container"></div>
<script>
(function(){
    const canvas=document.getElementById('neural-canvas');if(!canvas)return;
    const ctx=canvas.getContext('2d');
    function resize(){canvas.width=window.innerWidth;canvas.height=window.innerHeight;}
    resize();window.addEventListener('resize',resize);
    const nodes=Array.from({length:40},()=>({x:Math.random()*canvas.width,y:Math.random()*canvas.height,vx:(Math.random()-0.5)*0.4,vy:(Math.random()-0.5)*0.4,r:Math.random()*2.5+1.5,pulse:Math.random()*Math.PI*2}));
    function draw(){
        ctx.clearRect(0,0,canvas.width,canvas.height);
        nodes.forEach(n=>{n.x+=n.vx;n.y+=n.vy;n.pulse+=0.02;if(n.x<0||n.x>canvas.width)n.vx*=-1;if(n.y<0||n.y>canvas.height)n.vy*=-1;});
        nodes.forEach((a,i)=>{nodes.slice(i+1).forEach(b=>{const d=Math.hypot(a.x-b.x,a.y-b.y);if(d<130){const al=Math.floor((1-d/130)*60).toString(16).padStart(2,'0');ctx.beginPath();ctx.strokeStyle='#00d4ff'+al;ctx.lineWidth=0.6;ctx.moveTo(a.x,a.y);ctx.lineTo(b.x,b.y);ctx.stroke();}});});
        nodes.forEach(n=>{const g=Math.sin(n.pulse)*0.5+0.5;ctx.beginPath();ctx.arc(n.x,n.y,n.r+g,0,Math.PI*2);ctx.fillStyle='#00d4ff';ctx.shadowBlur=12+g*8;ctx.shadowColor='#00d4ff';ctx.fill();});
        requestAnimationFrame(draw);
    }
    draw();
})();
(function(){
    const c=document.getElementById('particles-container');if(!c)return;
    for(let i=0;i<25;i++){const p=document.createElement('div');p.className='particle';p.style.left=Math.random()*100+'vw';p.style.animationDuration=(Math.random()*10+8)+'s';p.style.animationDelay=(Math.random()*10)+'s';p.style.width=p.style.height=(Math.random()*3+1)+'px';p.style.opacity=Math.random()*0.6+0.2;c.appendChild(p);}
})();
(function(){
    const MAX=2.5;let raf=null;
    document.addEventListener('mousemove',function(e){if(raf)cancelAnimationFrame(raf);raf=requestAnimationFrame(function(){const cx=window.innerWidth/2,cy=window.innerHeight/2;const dx=(e.clientX-cx)/cx,dy=(e.clientY-cy)/cy;const rx=(-dy*MAX).toFixed(2),ry=(dx*MAX).toFixed(2);const c=document.querySelector('.main .block-container');if(c){c.style.transition='transform 0.12s ease-out';c.style.transform=`perspective(1200px) rotateX(${rx}deg) rotateY(${ry}deg)`;}});});
    document.addEventListener('mouseleave',function(){const c=document.querySelector('.main .block-container');if(c){c.style.transition='transform 0.5s ease';c.style.transform='perspective(1200px) rotateX(0deg) rotateY(0deg)';}});
})();
</script>
""", unsafe_allow_html=True)

RUNNING_LOCAL = is_local()

# ── Video Processing Helper ───────────────────────────────────
def process_video(video_path, model, original_filename="video.mp4"):
    try:
        import imageio
    except ImportError:
        st.error("❌ imageio not found. Please add `imageio` and `imageio-ffmpeg` to requirements.txt.")
        return None, {}, 0
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps    = cap.get(cv2.CAP_PROP_FPS) or 25
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    if total_frames == 0 or width == 0:
        st.error("❌ Could not open the video file. It may be corrupted or in an unsupported format.")
        return None, {}, 0
    st.markdown(f'<div class="metric-card" style="text-align:left;padding:14px 20px;margin:10px 0;"><span style="color:#64748b;font-size:0.85rem;">📹 <b style="color:#00d4ff;">{total_frames}</b> frames &nbsp;·&nbsp; 🎞️ <b style="color:#00d4ff;">{fps:.1f}</b> FPS &nbsp;·&nbsp; 📐 <b style="color:#00d4ff;">{width}×{height}</b>px</span></div>', unsafe_allow_html=True)
    out_path      = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
    progress_bar  = st.progress(0, text="⏳ Processing frames...")
    preview_ph    = st.empty()
    stats_ph      = st.empty()
    frame_idx=0; total_det=0; all_classes={}; start_time=time.time(); output_frames=[]
    try:
        reader = imageio.get_reader(video_path, format='ffmpeg')
    except Exception as e:
        st.error(f"❌ Failed to read video: {e}")
        return None, {}, 0
    for frame_rgb in reader:
        pil_f     = Image.fromarray(frame_rgb)
        res       = run_inference(model, pil_f, score_thr)
        out_f, nd = draw_results(pil_f, res, mask_thr=mask_thr, show_masks=show_masks, show_boxes=show_boxes, show_labels=show_labels, alpha=alpha)
        output_frames.append(out_f)
        total_det += nd
        for lbl in res['labels']:
            cname = COCO_CLASSES.get(int(lbl), '?')
            all_classes[cname] = all_classes.get(cname, 0) + 1
        frame_idx += 1
        elapsed = time.time() - start_time
        eta = (elapsed/frame_idx)*(total_frames-frame_idx) if frame_idx > 0 else 0
        progress_bar.progress(min(frame_idx/max(total_frames,1),1.0), text=f"⏳ Processing frame {frame_idx} of {total_frames} — ETA: {eta:.0f}s")
        if frame_idx % 10 == 0:
            preview_ph.image(out_f, caption=f"Frame #{frame_idx} — {nd} object(s) detected", width="stretch")
            stats_ph.markdown(f"| | |\n|---|---|\n| 🎯 Frame | **{frame_idx} / {total_frames}** |\n| 🔍 This frame | **{nd}** |\n| 📊 Total detections | **{total_det}** |\n| ⏱️ Elapsed | **{elapsed:.1f}s** |\n| 🏷️ Classes | **{', '.join(list(all_classes.keys())[:5])}{'...' if len(all_classes)>5 else ''}** |")
    reader.close()
    with st.spinner("💾 Saving output video..."):
        try:
            writer = imageio.get_writer(out_path, format='ffmpeg', fps=fps, codec='libx264', output_params=['-pix_fmt','yuv420p'])
            for frm in output_frames: writer.append_data(frm)
            writer.close()
        except Exception as e:
            st.error(f"❌ Failed to save output video: {e}")
            return None, all_classes, total_det
    progress_bar.progress(1.0, text="✅ Processing complete!")
    return out_path, all_classes, total_det

def show_video_summary(out_path, all_classes, total_det, total_frames, filename="output.mp4"):
    st.markdown("---")
    st.markdown("### 📊 Video Detection Summary")
    sc1,sc2,sc3 = st.columns(3)
    sc1.markdown(f'<div class="metric-card"><div class="metric-val">{total_frames}</div><div class="metric-lbl">Total Frames</div></div>', unsafe_allow_html=True)
    sc2.markdown(f'<div class="metric-card"><div class="metric-val">{total_det}</div><div class="metric-lbl">Total Detections</div></div>', unsafe_allow_html=True)
    sc3.markdown(f'<div class="metric-card"><div class="metric-val">{total_det//max(total_frames,1)}</div><div class="metric-lbl">Avg per Frame</div></div>', unsafe_allow_html=True)
    if all_classes:
        st.markdown("#### 🏷️ Classes Detected")
        st.dataframe(pd.DataFrame(sorted(all_classes.items(),key=lambda x:-x[1]),columns=["Class","Total Detections"]), width="stretch", hide_index=True)
    if out_path and os.path.exists(out_path):
        st.markdown("---")
        with open(out_path,"rb") as f: video_bytes=f.read()
        st.download_button("⬇️ Download Output Video", video_bytes, f"mask_rcnn_{filename}", "video/mp4", width="stretch", type="primary")
        try: os.unlink(out_path)
        except: pass

def show_result_details(results, n_det, score_thr):
    if n_det == 0:
        st.warning("⚠️ No objects detected. Try lowering the Confidence Threshold.")
        return
    mc1,mc2,mc3,mc4 = st.columns(4)
    mc1.markdown(f'<div class="metric-card"><div class="metric-val">{n_det}</div><div class="metric-lbl">Objects Detected</div></div>', unsafe_allow_html=True)
    mc2.markdown(f'<div class="metric-card"><div class="metric-val">{results["time"]*1000:.0f}ms</div><div class="metric-lbl">Inference Time</div></div>', unsafe_allow_html=True)
    mc3.markdown(f'<div class="metric-card"><div class="metric-val">{score_thr:.2f}</div><div class="metric-lbl">Confidence Threshold</div></div>', unsafe_allow_html=True)
    mc4.markdown(f'<div class="metric-card"><div class="metric-val">{str(device).upper()}</div><div class="metric-lbl">Device</div></div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("#### 📋 Detection Details")
    rows=[]
    for i in range(n_det):
        lid=int(results['labels'][i]); lname=COCO_CLASSES.get(lid,f'class_{lid}')
        score=float(results['scores'][i]); box=results['boxes'][i].astype(int)
        rows.append({"#":i+1,"Class":lname,"Score":f"{score:.3f}","Width (px)":int(box[2]-box[0]),"Height (px)":int(box[3]-box[1]),"Bounding Box [x1,y1,x2,y2]":f"[{box[0]},{box[1]},{box[2]},{box[3]}]"})
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
    ch1,ch2 = st.columns(2)
    with ch1:
        st.markdown("#### 📊 Class Distribution")
        class_counts={}
        for i in range(n_det):
            name=COCO_CLASSES.get(int(results['labels'][i]),'?')
            class_counts[name]=class_counts.get(name,0)+1
        fig,ax=plt.subplots(figsize=(5,3.5))
        fig.patch.set_facecolor('#020917');ax.set_facecolor('#0a1628')
        bars=ax.bar(class_counts.keys(),class_counts.values(),color='#00d4ff',edgecolor='#003d52')
        for bar,val in zip(bars,class_counts.values()):
            ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+0.05,str(val),ha='center',va='bottom',fontweight='bold',color='#00d4ff')
        ax.set_ylabel("Count",color='#64748b');ax.set_title("Instances per Class",fontweight='bold',color='#e2e8f0')
        ax.set_ylim(0,max(class_counts.values())+1.5);ax.tick_params(colors='#64748b')
        for s in ['top','right']:ax.spines[s].set_visible(False)
        for s in ['bottom','left']:ax.spines[s].set_color('#1e3a5f')
        plt.xticks(rotation=30,ha='right',fontsize=9,color='#94a3b8');plt.tight_layout();st.pyplot(fig);plt.close()
    with ch2:
        if n_det > 1:
            st.markdown("#### 📉 Confidence Scores")
            scores_list=results['scores'].tolist()
            labels_list=[COCO_CLASSES.get(int(l),'?') for l in results['labels']]
            colors_bar=[f'#{COLORS[i%len(COLORS)][0]:02x}{COLORS[i%len(COLORS)][1]:02x}{COLORS[i%len(COLORS)][2]:02x}' for i in range(n_det)]
            fig2,ax2=plt.subplots(figsize=(5,3.5))
            fig2.patch.set_facecolor('#020917');ax2.set_facecolor('#0a1628')
            ax2.barh(range(n_det),scores_list,color=colors_bar,edgecolor='#001428')
            ax2.set_yticks(range(n_det));ax2.set_yticklabels([f"#{i+1} {labels_list[i]}" for i in range(n_det)],fontsize=8,color='#94a3b8')
            ax2.set_xlabel("Confidence Score",color='#64748b');ax2.set_xlim(0,1.05)
            ax2.axvline(score_thr,color='#00d4ff',linestyle='--',linewidth=1.5,label=f'Threshold = {score_thr}')
            ax2.legend(fontsize=8,facecolor='#0a1628',labelcolor='#94a3b8')
            ax2.set_title("Per-Instance Confidence",fontweight='bold',color='#e2e8f0');ax2.tick_params(colors='#64748b')
            for s in ['top','right']:ax2.spines[s].set_visible(False)
            for s in ['bottom','left']:ax2.spines[s].set_color('#1e3a5f')
            plt.tight_layout();st.pyplot(fig2);plt.close()

# ── Header ────────────────────────────────────────────────────
env_icon = "🖥️ Local Mode" if RUNNING_LOCAL else "☁️ Cloud Mode"
st.markdown(f"""
<div class="neural-header">
    <p class="main-title">🎭 Mask R-CNN</p>
    <p class="main-title" style="font-size:1.5rem;margin-top:-6px;">Instance Segmentation</p>
    <p class="subtitle">ResNet-50 FPN &nbsp;·&nbsp; MS-COCO 2017 Pretrained &nbsp;·&nbsp; 80 Object Categories &nbsp;·&nbsp; Image + Video + Webcam</p>
    <div style="margin-top:16px;display:flex;gap:8px;flex-wrap:wrap;">
        <span class="badge">🧠 ResNet-50 FPN</span>
        <span class="badge">📦 MS-COCO 2017</span>
        <span class="badge">🏷️ 80 Categories</span>
        <span class="badge">🎬 Video Support</span>
        <span class="badge">{env_icon}</span>
    </div>
</div>
""", unsafe_allow_html=True)
st.markdown("---")

# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    score_thr = st.slider("🎯 Confidence Threshold", 0.10, 0.95, 0.50, 0.05)
    st.markdown('<div class="slider-desc"><b>How confident should the model be?</b><br>↑ Higher → More accurate, fewer detections<br>↓ Lower → More detections, possible false positives</div>', unsafe_allow_html=True)
    mask_thr = st.slider("🎨 Mask Threshold", 0.10, 0.90, 0.50, 0.05)
    st.markdown('<div class="slider-desc"><b>How tight should the object silhouette be?</b><br>↑ Higher → Sharp, precise edges<br>↓ Lower → Looser mask, more area coverage</div>', unsafe_allow_html=True)
    alpha = st.slider("🌫️ Mask Transparency", 0.10, 0.90, 0.45, 0.05)
    st.markdown('<div class="slider-desc"><b>Color overlay opacity:</b><br>↑ Higher → More opaque mask overlay<br>↓ Lower → Original image more visible</div>', unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("## 👁️ Display Options")
    show_masks  = st.checkbox("Show Masks",  value=True)
    show_boxes  = st.checkbox("Show Bounding Boxes", value=True)
    show_labels = st.checkbox("Show Labels", value=True)
    st.markdown("---")
    if st.session_state['pil_image'] is not None:
        if st.button("🔄 Re-run with New Settings", type="primary", width="stretch"):
            st.session_state['result_img'] = None
            st.session_state['results']    = None
            st.session_state['n_det']      = 0
    st.markdown("---")
    st.markdown("## 🖥️ System Info")
    st.markdown(f"**Device:** `{str(device).upper()}`")
    st.markdown(f"**PyTorch:** `{torch.__version__}`")
    st.markdown(f"**Mode:** {'🖥️ Local' if RUNNING_LOCAL else '☁️ Cloud'}")
    if torch.cuda.is_available():
        st.success(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        st.info("Running on CPU")

# ── Tabs ──────────────────────────────────────────────────────
st.markdown("### 📥 Select Input Source")
tab1, tab2, tab3, tab4 = st.tabs(["📁 Upload File","🔗 Image / Video URL","🖼️ Sample Images","📷 Webcam"])

# ── Tab 1: Upload ─────────────────────────────────────────────
with tab1:
    uploaded = st.file_uploader("Upload an Image or Video file", type=["jpg","jpeg","png","mp4","avi","mov","mkv"])
    if uploaded:
        file_type = uploaded.type
        if file_type.startswith("image"):
            img = Image.open(uploaded).convert("RGB")
            st.session_state['pil_image']  = img
            st.session_state['result_img'] = None
            st.session_state['results']    = None
            st.success(f"✅ {uploaded.name} loaded successfully ({img.size[0]}×{img.size[1]}px)")
            st.image(img, width="stretch")
        elif file_type.startswith("video"):
            st.success(f"✅ Video uploaded: **{uploaded.name}**")
            st.video(uploaded)
            st.markdown("---")
            st.markdown("""<div class="info-box">
⚡ <b>Detection will run on every frame.</b><br>
Large videos may take longer to process on cloud.<br>
Once complete, you can download the fully annotated output video.
</div>""", unsafe_allow_html=True)
            if st.button("🚀 Start Video Detection", type="primary", width="stretch"):
                tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
                tfile.write(uploaded.read()); tfile.flush()
                with st.spinner("🧠 Loading model..."):
                    model = load_model()
                out_path, all_classes, total_det = process_video(tfile.name, model, uploaded.name)
                cap2 = cv2.VideoCapture(tfile.name)
                total_frames = int(cap2.get(cv2.CAP_PROP_FRAME_COUNT)); cap2.release()
                show_video_summary(out_path, all_classes, total_det, total_frames, uploaded.name)
                try: os.unlink(tfile.name)
                except: pass
    elif st.session_state['pil_image'] is not None:
        st.info("ℹ️ Previous image is loaded. Upload a new file or click Run Detection below.")
        st.image(st.session_state['pil_image'], width="stretch")

# ── Tab 2: URL ────────────────────────────────────────────────
with tab2:
    st.markdown("""<div class="info-box">
💡 <b>Only direct file URLs are supported.</b><br>
🖼️ Image: <code>.jpg &nbsp;.jpeg &nbsp;.png &nbsp;.webp</code> &nbsp;|&nbsp; 🎬 Video: <code>.mp4 &nbsp;.avi &nbsp;.mov &nbsp;.mkv</code><br>
⚠️ YouTube, Bing, and Google links are <b>not supported</b> — they do not serve direct files.
</div>""", unsafe_allow_html=True)
    url = st.text_input("Paste a direct Image or Video URL:", placeholder="https://example.com/image.jpg")
    if url:
        url_clean  = url.lower().split("?")[0].split("#")[0]
        video_exts = (".mp4",".avi",".mov",".mkv",".webm",".mpeg")
        image_exts = (".jpg",".jpeg",".png",".gif",".webp",".bmp")
        is_video   = any(url_clean.endswith(e) for e in video_exts)
        is_image   = any(url_clean.endswith(e) for e in image_exts)
        if not is_video and not is_image:
            try:
                with st.spinner("🔍 Detecting URL type..."):
                    head = requests.head(url, timeout=8, allow_redirects=True, headers={"User-Agent":"Mozilla/5.0"})
                    ct = head.headers.get("Content-Type","").lower()
                if any(v in ct for v in ["video","mp4","webm","mpeg"]): is_video=True
                elif any(i in ct for i in ["image","jpeg","png","gif","webp"]): is_image=True
                elif "text/html" in ct:
                    st.error("❌ This URL points to a webpage, not a direct file. Please provide a direct image or video URL.")
                    is_image=False
                else: is_image=True
            except Exception: is_image=True

        if is_image and not is_video:
            try:
                with st.spinner("⬇️ Downloading image..."):
                    r = requests.get(url, timeout=15, headers={"User-Agent":"Mozilla/5.0"})
                    r.raise_for_status()
                    ct = r.headers.get("Content-Type","").lower()
                    if "text/html" in ct:
                        st.error("❌ The server returned a webpage instead of an image. Please use a direct image file URL.")
                        st.stop()
                    img = Image.open(BytesIO(r.content)).convert("RGB")
                st.session_state['pil_image']  = img
                st.session_state['result_img'] = None
                st.session_state['results']    = None
                st.success(f"✅ Image downloaded successfully! ({img.size[0]}×{img.size[1]}px)")
                st.image(img, width="stretch")
            except Exception as e:
                st.error(f"❌ Failed to load image: {e}")
                st.markdown("""<div class="warn-box">
💡 <b>Possible reasons:</b><br>
• The URL does not point to a direct image file<br>
• The server blocked the request (access restricted)<br>
• The URL has expired or is no longer valid<br><br>
✅ <b>Tip:</b> Open the URL in your browser — it should display the image directly, with no webpage around it.
</div>""", unsafe_allow_html=True)

        elif is_video:
            st.markdown(f"""<div class="info-box">🎬 <b>Video URL detected!</b><br>
<code style="font-size:0.8rem;word-break:break-all;">{url[:100]}{'...' if len(url)>100 else ''}</code></div>""", unsafe_allow_html=True)
            if st.button("⬇️ Download & Run Detection", type="primary", width="stretch"):
                progress_dl = st.progress(0, text="⬇️ Downloading video...")
                try:
                    with requests.get(url, stream=True, timeout=120, headers={"User-Agent":"Mozilla/5.0"}) as r:
                        r.raise_for_status()
                        ct = r.headers.get("Content-Type","").lower()
                        if "text/html" in ct:
                            st.error("❌ This is a webpage. Please provide a direct .mp4 file URL.")
                            st.stop()
                        total_size = int(r.headers.get("content-length",0))
                        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
                        downloaded=0
                        for chunk in r.iter_content(chunk_size=1024*1024):
                            if chunk:
                                tfile.write(chunk); downloaded+=len(chunk)
                                if total_size: progress_dl.progress(min(downloaded/total_size,1.0), text=f"⬇️ {downloaded//(1024*1024)} MB / {total_size//(1024*1024)} MB")
                        tfile.flush()
                    progress_dl.progress(1.0, text="✅ Download complete!")
                    st.success(f"✅ Video downloaded ({downloaded//(1024*1024)} MB)")
                except Exception as e:
                    st.error(f"❌ Download failed: {e}"); st.stop()
                with st.spinner("🧠 Loading model..."): model = load_model()
                out_path, all_classes, total_det = process_video(tfile.name, model)
                cap2 = cv2.VideoCapture(tfile.name)
                total_frames = int(cap2.get(cv2.CAP_PROP_FRAME_COUNT)); cap2.release()
                show_video_summary(out_path, all_classes, total_det, total_frames, "url_video.mp4")
                try: os.unlink(tfile.name)
                except: pass

# ── Tab 3: Sample Images ──────────────────────────────────────
with tab3:
    choice = st.selectbox("Select a sample image:", list(SAMPLE_IMAGES.keys()))
    if st.button("📥 Load Sample Image", width="stretch"):
        try:
            with st.spinner("Loading sample image..."):
                r = requests.get(SAMPLE_IMAGES[choice], timeout=10)
                img = Image.open(BytesIO(r.content)).convert("RGB")
            st.session_state['pil_image']  = img
            st.session_state['result_img'] = None
            st.session_state['results']    = None
            st.success(f"✅ Sample loaded! ({img.size[0]}×{img.size[1]}px)")
            st.image(img, width="stretch")
        except Exception as e:
            st.error(f"❌ Failed to load sample: {e}")

# ── Tab 4: Webcam ─────────────────────────────────────────────
with tab4:
    if not RUNNING_LOCAL:
        st.markdown("""<div class="cloud-box">
<h2>📸 Webcam — Photo Mode</h2>
<p style="color:#64748b;font-size:1rem;">Take a photo using your camera.<br>Mask R-CNN detection will run automatically.</p>
</div>""", unsafe_allow_html=True)
        camera_img = st.camera_input("Click the button to capture a photo")
        if camera_img is not None:
            pil_cam = Image.open(camera_img).convert("RGB")
            with st.spinner("🧠 Loading model..."): model = load_model()
            with st.spinner("⚡ Running inference..."): cam_results = run_inference(model, pil_cam, score_thr)
            with st.spinner("🎨 Rendering results..."):
                cam_output, cam_n = draw_results(pil_cam, cam_results, mask_thr=mask_thr, show_masks=show_masks, show_boxes=show_boxes, show_labels=show_labels, alpha=alpha)
            c1,c2 = st.columns(2)
            with c1: st.markdown("**📷 Original Photo**"); st.image(pil_cam, width="stretch")
            with c2: st.markdown(f"**🎭 Detection Result — {cam_n} object(s)**"); st.image(cam_output, width="stretch")
            buf=BytesIO(); Image.fromarray(cam_output).save(buf, format="PNG")
            st.download_button("⬇️ Download Result", buf.getvalue(), "webcam_result.png", "image/png", width="stretch")
            st.markdown("---"); show_result_details(cam_results, cam_n, score_thr)
        st.markdown("---")
        with st.expander("🎥 How to run Live Video Detection? (Local Setup Guide)"):
            st.markdown("""
Live video detection is only available on a local machine.
```bash
pip install streamlit torch torchvision opencv-python pillow requests pandas matplotlib imageio imageio-ffmpeg
streamlit run app.py
```
Open `http://localhost:8501` → Webcam tab → Live Video → ▶️ Start
            """)
    else:
        st.markdown('<div class="info-box">🖥️ <b>Local mode detected!</b> Both webcam modes are available.</div>', unsafe_allow_html=True)
        webcam_mode = st.radio("Select Mode:", ["📸 Single Photo","🎥 Live Video"], horizontal=True)
        if webcam_mode == "📸 Single Photo":
            camera_img = st.camera_input("Click to capture a photo")
            if camera_img is not None:
                pil_cam = Image.open(camera_img).convert("RGB")
                with st.spinner("🧠 Loading model..."): model = load_model()
                with st.spinner("⚡ Running inference..."): cam_results = run_inference(model, pil_cam, score_thr)
                with st.spinner("🎨 Rendering results..."):
                    cam_output, cam_n = draw_results(pil_cam, cam_results, mask_thr=mask_thr, show_masks=show_masks, show_boxes=show_boxes, show_labels=show_labels, alpha=alpha)
                c1,c2=st.columns(2)
                with c1: st.markdown("**📷 Original**"); st.image(pil_cam, width="stretch")
                with c2: st.markdown(f"**🎭 {cam_n} object(s) detected**"); st.image(cam_output, width="stretch")
                buf=BytesIO(); Image.fromarray(cam_output).save(buf, format="PNG")
                st.download_button("⬇️ Download Result", buf.getvalue(), "photo_result.png", "image/png", width="stretch")
                st.markdown("---"); show_result_details(cam_results, cam_n, score_thr)
        else:
            st.markdown('<div class="local-box"><h3>🎥 Live Video Detection</h3><p style="color:#64748b;">Real-time object detection via webcam</p></div>', unsafe_allow_html=True)
            col1,col2=st.columns(2)
            start_btn=col1.button("▶️ Start Detection", type="primary", width="stretch")
            stop_btn=col2.button("⏹️ Stop", width="stretch")
            if start_btn: st.session_state.live_on=True
            if stop_btn:  st.session_state.live_on=False
            if st.session_state.live_on:
                st.markdown('<span class="live-badge">● LIVE</span>', unsafe_allow_html=True)
                frame_ph=st.empty(); stats_ph=st.empty()
                with st.spinner("🧠 Loading model..."): model=load_model()
                cap=cv2.VideoCapture(0)
                if not cap.isOpened():
                    st.error("❌ Could not open webcam!"); st.session_state.live_on=False
                else:
                    frame_count=0
                    while st.session_state.live_on:
                        ret,frame=cap.read()
                        if not ret: break
                        frame_rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
                        pil_f=Image.fromarray(frame_rgb)
                        res=run_inference(model,pil_f,score_thr)
                        out_f,nd=draw_results(pil_f,res,mask_thr=mask_thr,show_masks=show_masks,show_boxes=show_boxes,show_labels=show_labels,alpha=alpha)
                        frame_ph.image(out_f,caption=f"Frame #{frame_count} — {nd} object(s) detected",width="stretch")
                        detected=list(set([COCO_CLASSES.get(int(l),'?') for l in res['labels']]))
                        stats_ph.markdown(f"🎯 **{nd}** object(s) &nbsp;|&nbsp; ⚡ **{res['time']*1000:.0f}ms** &nbsp;|&nbsp; 🏷️ {', '.join(detected[:4]) or 'None'}")
                        frame_count+=1
                        if not st.session_state.live_on: break
                    cap.release(); st.info("⏹️ Live detection stopped.")

# ── Run Detection ─────────────────────────────────────────────
pil_image = st.session_state.get('pil_image', None)
if pil_image is not None:
    st.markdown("---")
    st.markdown("### 🔮 Run Object Detection")
    if st.session_state['result_img'] is not None:
        left,right=st.columns(2)
        with left:
            st.markdown("**🖼️ Original Image**"); st.image(pil_image, width="stretch")
        with right:
            st.markdown(f"**🎭 Detection Result — {st.session_state['n_det']} instance(s)**")
            st.image(st.session_state['result_img'], width="stretch")
        buf=BytesIO(); Image.fromarray(st.session_state['result_img']).save(buf, format="PNG")
        st.download_button("⬇️ Download Result", buf.getvalue(), "mask_rcnn_output.png", "image/png", width="stretch")
        st.markdown("---")
        show_result_details(st.session_state['results'], st.session_state['n_det'], score_thr)
        st.markdown("---")
    if st.button("🚀 Run Mask R-CNN Detection", type="primary", width="stretch"):
        with st.spinner("🧠 Loading model..."): model=load_model()
        with st.spinner("⚡ Running inference..."): results=run_inference(model, pil_image, score_thr)
        with st.spinner("🎨 Rendering results..."):
            output_img,n_det=draw_results(pil_image, results, mask_thr=mask_thr, show_masks=show_masks, show_boxes=show_boxes, show_labels=show_labels, alpha=alpha)
        st.session_state['result_img']=output_img
        st.session_state['results']=results
        st.session_state['n_det']=n_det
        st.rerun()

# ── About ─────────────────────────────────────────────────────
st.markdown("---")
c1,c2=st.columns(2)
with c1:
    with st.expander("ℹ️ What is Mask R-CNN?"):
        st.markdown("""
**Mask R-CNN** is a deep learning model for instance segmentation that:
- Detects objects across **80 categories**
- Draws a **bounding box** around each detected object
- Generates a **pixel-level instance mask** for each object

**Architecture Pipeline:**
ResNet-50 → FPN → RPN → RoI Align → 3 Prediction Heads (class + box + mask)

**Reference:** He et al., *Mask R-CNN*, ICCV 2017
        """)
with c2:
    with st.expander("📦 Model Information"):
        st.markdown(f"""
| Property | Value |
|---|---|
| Architecture | Mask R-CNN |
| Backbone | ResNet-50 + FPN |
| Pre-trained On | MS-COCO 2017 |
| Object Categories | 80 |
| Parameters | ~44 Million |
| Inference Device | `{str(device).upper()}` |
| Video Detection | ✅ Upload + URL |
| Webcam | ✅ Photo + Live Video |
        """)
with st.expander("🏷️ All 80 COCO Object Categories"):
    cols=st.columns(5)
    for i,cls in enumerate(COCO_CLASSES.values()):
        cols[i%5].markdown(f"<span class='tag'>{cls}</span>", unsafe_allow_html=True)

st.markdown("---")
st.markdown("""<div class="footer">
    🎭 Mask R-CNN Instance Segmentation &nbsp;·&nbsp; Built with Streamlit &nbsp;·&nbsp;
    MS-COCO 2017 Pre-trained &nbsp;·&nbsp; 📁 Image &nbsp;+&nbsp; 🎬 Video &nbsp;+&nbsp; 📷 Webcam
</div>""", unsafe_allow_html=True)
