import streamlit as st
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
import cv2

from utils import load_change_model, preprocess, predict, compute_metrics
from streamlit_image_comparison import image_comparison


# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="TerraChange",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =====================================================
# SIDEBAR
# =====================================================
st.sidebar.title("TerraChange")

page = st.sidebar.radio(
    "Navigate",
    ["Upload & Predict", "Qualitative Examples", "Model Architecture"],
    index=0
)

light_mode = st.sidebar.toggle("Light Mode", value=False)

st.sidebar.markdown("---")
st.sidebar.caption("Self-Supervised Satellite Change Detection")


# =====================================================
# GLOBAL THEME (STABLE LIGHT / DARK)
# =====================================================
if light_mode:
    st.markdown("""
    <style>
    :root {
        --bg1:#f8fafc;
        --bg2:#e5edf5;
        --txt:#0f172a;
        --muted:#475569;
        --card:rgba(255,255,255,0.92);
        --border:#cbd5e1;
    }

    html, body, .stApp {
        background: linear-gradient(135deg,var(--bg1),var(--bg2));
        color: var(--txt) !important;
    }

    h1,h2,h3,p,li,span,label {
        color: var(--txt) !important;
        font-family: Inter, Segoe UI, sans-serif;
    }

    section[data-testid="stSidebar"] {
        background:#f1f5f9;
        border-right:1px solid var(--border);
    }

    section[data-testid="stSidebar"] * {
        color: var(--txt) !important;
    }

    [data-testid="metric-container"] {
        background:var(--card);
        border-radius:14px;
        box-shadow:0 8px 24px rgba(0,0,0,0.08);
    }

    section[data-testid="stFileUploader"] {
        background:var(--card);
        border-radius:12px;
        border:1px solid var(--border);
    }

    img { border-radius:14px; }
    </style>
    """, unsafe_allow_html=True)

else:
    st.markdown("""
    <style>
    :root {
        --bg1:#0b1220;
        --bg2:#0f2027;
        --txt:#eaeaea;
        --muted:#94a3b8;
        --card:rgba(255,255,255,0.06);
        --border:rgba(255,255,255,0.15);
    }

    html, body, .stApp {
        background: linear-gradient(135deg,var(--bg1),var(--bg2));
        color: var(--txt) !important;
    }

    h1,h2,h3,p,li,span,label {
        color: var(--txt) !important;
        font-family: Inter, Segoe UI, sans-serif;
    }

    section[data-testid="stSidebar"] {
        background:#020617;
        border-right:1px solid var(--border);
    }

    section[data-testid="stSidebar"] * {
        color: var(--txt) !important;
    }

    [data-testid="metric-container"] {
        background:var(--card);
        border-radius:14px;
        box-shadow:0 12px 40px rgba(0,0,0,0.35);
    }

    section[data-testid="stFileUploader"] {
        background:rgba(255,255,255,0.05);
        border-radius:12px;
        border:1px dashed var(--border);
    }

    img { border-radius:14px; }
    </style>
    """, unsafe_allow_html=True)


# =====================================================
# METRIC CARDS
# =====================================================
def show_metric_cards(iou, precision, recall, f1):
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("IoU", f"{iou:.3f}")
    c2.metric("Precision", f"{precision:.3f}")
    c3.metric("Recall", f"{recall:.3f}")
    c4.metric("F1-score", f"{f1:.3f}")


# =====================================================
# LOAD MODEL
# =====================================================
model = load_change_model()

st.title("TerraChange")
st.caption("Self-Supervised Satellite Image Change Detection")


# =====================================================
# PAGE 1 — UPLOAD & PREDICT
# =====================================================
if page == "Upload & Predict":

    st.header("Upload Satellite Images")

    c1, c2 = st.columns(2)
    with c1:
        img1_file = st.file_uploader("Upload T1 Image", type=["png","jpg","jpeg"])
    with c2:
        img2_file = st.file_uploader("Upload T2 Image", type=["png","jpg","jpeg"])

    if img1_file and img2_file:
        img1 = Image.open(img1_file).convert("RGB")
        img2 = Image.open(img2_file).convert("RGB")

        t1 = preprocess(img1)
        t2 = preprocess(img2)

        prob, pred = predict(model, t1, t2)

        pred_resized = cv2.resize(
            pred.astype(np.uint8),
            img1.size,
            interpolation=cv2.INTER_NEAREST
        )
        pred_vis = Image.fromarray(pred_resized * 255)

        st.subheader("Prediction Result")
        c1, c2, c3 = st.columns(3)
        c1.image(img1, caption="T1 Image", use_container_width=True)
        c2.image(img2, caption="T2 Image", use_container_width=True)
        c3.image(pred_vis, caption="Prediction Mask", use_container_width=True)

        # ---------------- CONFIDENCE ----------------
        st.subheader("Prediction Confidence")

        mean_conf = float(np.mean(prob))
        high_conf = float(np.mean(prob > 0.7))

        c1, c2 = st.columns(2)
        c1.metric("Mean Change Confidence", f"{mean_conf:.3f}")
        c2.metric("High-Confidence Pixels (%)", f"{high_conf*100:.2f}%")

        st.caption(
            "Low mean confidence is expected because most pixels remain unchanged. "
            "High-confidence pixels highlight localized structural changes."
        )

        # ---------------- HEATMAP ----------------
        st.subheader("Change Probability Heatmap")

        fig, ax = plt.subplots(figsize=(3,3))
        ax.imshow(prob, cmap="inferno", vmin=0, vmax=1)
        ax.axis("off")
        st.pyplot(fig, use_container_width=False)

        st.markdown("""
        **Confidence Legend**
        - **0.0 – 0.3** → No change  
        - **0.3 – 0.5** → Uncertain  
        - **0.5 – 0.7** → Moderate change  
        - **0.7 – 1.0** → High-confidence change  
        """)


# =====================================================
# PAGE 2 — QUALITATIVE EXAMPLES
# =====================================================
elif page == "Qualitative Examples":

    st.header("Qualitative Examples")

    examples_dir = "notebooks/examples"
    if not os.path.exists(examples_dir):
        st.info("Examples not generated yet.")
    else:
        for idx, ex in enumerate(sorted(os.listdir(examples_dir)), start=1):
            ex_path = os.path.join(examples_dir, ex)

            img1 = Image.open(os.path.join(ex_path,"t1.png")).convert("RGB")
            img2 = Image.open(os.path.join(ex_path,"t2.png")).convert("RGB")
            gt = Image.open(os.path.join(ex_path,"gt.png")).convert("L")

            t1 = preprocess(img1)
            t2 = preprocess(img2)

            prob, pred = predict(model, t1, t2)

            pred_resized = cv2.resize(pred.astype(np.uint8), gt.size, cv2.INTER_NEAREST)
            pred_vis = Image.fromarray(pred_resized * 255)
            gt_vis = Image.fromarray((np.array(gt)>0).astype(np.uint8)*255)

            gt_np = np.array(gt_vis)/255.0
            iou, p, r, f1 = compute_metrics(pred_resized, gt_np)

            st.subheader(f"Example {idx}")

            c1, c2 = st.columns(2)
            with c1:
                image_comparison(img1,img2,"T1","T2",width=350)
            with c2:
                image_comparison(gt_vis,pred_vis,"Ground Truth","Prediction",width=350)

            show_metric_cards(iou,p,r,f1)


# =====================================================
# PAGE 3 — MODEL ARCHITECTURE
# =====================================================
else:

    st.header("Model Architecture")

    st.markdown("""
    TerraChange uses a Siamese ResNet50 encoder and a U-Net decoder.
    Feature differences between two timestamps are decoded into a
    pixel-level change probability map.
    """)

    st.image(
        "assets/model_flow.png",
        caption="Siamese Change Detection Pipeline",
        width=600
    )

    st.markdown("""
    **Why TerraChange works**
    - Shared encoder ensures consistent comparison  
    - Feature differencing suppresses unchanged regions  
    - U-Net decoder preserves spatial detail  
    - Self-supervised pretraining improves robustness  
    """)
