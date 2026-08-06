import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import json
import os
from PIL import Image
from tensorflow.keras.applications.resnet_v2 import preprocess_input
from tensorflow.keras.models import load_model, Model
import datetime

# ── Page config ───────────────────────────────────────────────────
st.set_page_config(
    page_title="Brain MRI Tumor Classifier",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS — dark medical theme ──────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;600;700&display=swap');

    .stApp {
        background-color: #060c18;
        color: #e0e6f0;
        font-family: 'DM Sans', sans-serif;
    }

    [data-testid="stSidebar"] {
        background-color: #080f1e;
        border-right: 1px solid #1a2d45;
    }

    .metric-card {
        background: linear-gradient(135deg, #0a1628, #111f38);
        border: 1px solid #1a3a5c;
        border-radius: 10px;
        padding: 16px;
        text-align: center;
        margin: 6px 0;
        position: relative;
        overflow: hidden;
    }

    .metric-card::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 2px;
        background: linear-gradient(90deg, #4fc3f7, #0288d1);
    }

    .metric-value {
        font-family: 'Space Mono', monospace;
        font-size: 24px;
        font-weight: 700;
        color: #4fc3f7;
        margin: 0;
    }

    .metric-label {
        font-size: 10px;
        color: #6a88a8;
        margin: 4px 0 0 0;
        text-transform: uppercase;
        letter-spacing: 1.5px;
    }

    [data-testid="stFileUploader"] {
        background-color: #0a1628;
        border: 2px dashed #1a3a5c;
        border-radius: 12px;
        padding: 20px;
    }

    .prediction-box {
        background: linear-gradient(135deg, #0a1e33, #081628);
        border: 2px solid #4fc3f7;
        border-radius: 14px;
        padding: 24px;
        text-align: center;
        margin: 12px 0;
        position: relative;
        overflow: hidden;
    }

    .prediction-box::after {
        content: '';
        position: absolute;
        bottom: 0; left: 0; right: 0;
        height: 1px;
        background: linear-gradient(90deg, transparent, #4fc3f7, transparent);
    }

    .prediction-class {
        font-family: 'Space Mono', monospace;
        font-size: 26px;
        font-weight: 700;
        color: #4fc3f7;
        margin: 0;
        text-transform: uppercase;
        letter-spacing: 2px;
    }

    .prediction-conf {
        font-size: 16px;
        color: #81d4fa;
        margin: 8px 0 0 0;
        font-family: 'Space Mono', monospace;
    }

    /* ── UNCERTAINTY ALERT ── */
    .uncertainty-banner {
        background: linear-gradient(135deg, #2a1a00, #1e1200);
        border: 1px solid #f57c00;
        border-left: 4px solid #ff9800;
        border-radius: 10px;
        padding: 14px 18px;
        margin: 14px 0;
        display: flex;
        align-items: flex-start;
        gap: 12px;
    }

    .uncertainty-icon {
        font-size: 22px;
        flex-shrink: 0;
        margin-top: 2px;
    }

    .uncertainty-title {
        color: #ff9800;
        font-weight: 700;
        font-size: 13px;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 4px;
    }

    .uncertainty-text {
        color: #e0a060;
        font-size: 12px;
        line-height: 1.5;
    }

    .uncertainty-classes {
        display: flex;
        gap: 8px;
        margin-top: 8px;
        flex-wrap: wrap;
    }

    .uncertainty-class-chip {
        padding: 3px 10px;
        border-radius: 20px;
        font-size: 11px;
        font-weight: 600;
        font-family: 'Space Mono', monospace;
    }

    /* ── CONFIDENCE BARS ── */
    .bar-container {
        margin: 10px 0;
    }

    .bar-label {
        display: flex;
        justify-content: space-between;
        margin-bottom: 5px;
        font-size: 12px;
        color: #8899b0;
    }

    .bar-track {
        background: #0d1e33;
        border-radius: 6px;
        height: 8px;
        overflow: hidden;
        border: 1px solid #1a3a5c;
    }

    .bar-fill {
        height: 100%;
        border-radius: 6px;
        transition: width 0.5s ease;
    }

    /* ── HEATMAP LEGEND ── */
    .heatmap-legend {
        margin-top: 10px;
        border-radius: 8px;
        overflow: hidden;
        border: 1px solid #1a3a5c;
    }

    .heatmap-legend-bar {
        height: 12px;
        background: linear-gradient(to right,
            #000080, #0000ff, #00ffff, #00ff00,
            #ffff00, #ff8000, #ff0000
        );
    }

    .heatmap-legend-labels {
        display: flex;
        justify-content: space-between;
        padding: 4px 6px;
        background: #0a1628;
        font-size: 10px;
        color: #556677;
        font-family: 'Space Mono', monospace;
    }

    /* ── REPORT BOX ── */
    .report-box {
        background: #08131f;
        border: 1px solid #1a3a5c;
        border-radius: 10px;
        padding: 16px;
        font-family: 'Space Mono', monospace;
        font-size: 11px;
        color: #7799bb;
        line-height: 1.7;
        white-space: pre-wrap;
        margin-top: 10px;
    }

    /* ── SECTION HEADERS ── */
    .section-header {
        color: #4fc3f7;
        font-size: 11px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 2px;
        border-bottom: 1px solid #1a3a5c;
        padding-bottom: 8px;
        margin-bottom: 14px;
        font-family: 'Space Mono', monospace;
    }

    /* ── IMAGE ZOOM WRAPPER ── */
    .zoom-wrapper {
        overflow: hidden;
        border-radius: 10px;
        border: 1px solid #1a3a5c;
        cursor: zoom-in;
        transition: transform 0.25s ease;
    }

    .zoom-wrapper:hover {
        transform: scale(1.02);
        box-shadow: 0 0 30px rgba(79,195,247,0.2);
    }

    .disclaimer {
        background: #121100;
        border: 1px solid #443300;
        border-radius: 8px;
        padding: 12px 14px;
        color: #aa8833;
        font-size: 11px;
        margin-top: 16px;
        line-height: 1.5;
    }

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ── Paths ─────────────────────────────────────────────────────────
MODEL_DIR   = os.path.join(os.path.dirname(__file__), 'model')
MODEL_PATH  = os.path.join(MODEL_DIR, 'best_model.keras')
NAMES_PATH  = os.path.join(MODEL_DIR, 'class_names.json')
CONFIG_PATH = os.path.join(MODEL_DIR, 'config.json')

# ── Load model and metadata ───────────────────────────────────────
@st.cache_resource
def load_resources():
    model       = load_model(MODEL_PATH)
    class_names = json.load(open(NAMES_PATH))
    config      = json.load(open(CONFIG_PATH))
    return model, class_names, config

model, class_names, config = load_resources()

IMAGE_SIZE = (224, 224)

CLASS_COLORS = {
    'glioma_tumor'     : '#ef5350',
    'meningioma_tumor' : '#ff9800',
    'normal'           : '#66bb6a',
    'pituitary_tumor'  : '#ab47bc',
}

CLASS_LABELS = {
    'glioma_tumor'     : 'Glioma Tumor',
    'meningioma_tumor' : 'Meningioma Tumor',
    'normal'           : 'Normal',
    'pituitary_tumor'  : 'Pituitary Tumor',
}

CLASS_DESCRIPTIONS = {
    'glioma_tumor'     : 'Arises from glial cells. Most common primary brain tumor. Requires urgent specialist review.',
    'meningioma_tumor' : 'Arises from meninges. Often slow-growing; many are benign. Follow-up imaging advised.',
    'normal'           : 'No tumor detected in this scan. Routine follow-up as clinically indicated.',
    'pituitary_tumor'  : 'Located near the pituitary gland. May affect hormones. Endocrinology referral often needed.',
}

UNCERTAINTY_THRESHOLD = 0.15   # flag if gap between top-2 is below 15%


# ── FIXED Grad-CAM ────────────────────────────────────────────────
def make_gradcam(img_array, model, last_conv='conv5_block3_out'):
    """
    Grad-CAM that works whether grad_model returns tensors or a list.
    """
    grad_model = Model(
        inputs=model.inputs,
        outputs=[model.get_layer(last_conv).output, model.output]
    )

    img_tensor = tf.cast(img_array, tf.float32)

    with tf.GradientTape() as tape:
        tape.watch(img_tensor)
        outputs = grad_model(img_tensor)

        conv_out = tf.cast(outputs[0], tf.float32)   # (1, H, W, C)
        preds    = tf.cast(outputs[1], tf.float32)   # (1, num_classes)

        pred_idx      = tf.argmax(preds[0])
        class_channel = tf.gather(preds[0], pred_idx)

    grads = tape.gradient(class_channel, conv_out)

    if grads is None:
        st.warning("Grad-CAM: gradients are None — check last_conv layer name.")
        return np.zeros((7, 7))

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))   # (C,)

    conv_map = conv_out[0]                                  # (H, W, C)
    heatmap  = conv_map @ pooled_grads[..., tf.newaxis]     # (H, W, 1)
    heatmap  = tf.squeeze(heatmap)                          # (H, W)

    heatmap  = tf.nn.relu(heatmap)
    max_val  = tf.reduce_max(heatmap)
    if max_val > 0:
        heatmap = heatmap / max_val

    return heatmap.numpy()


def overlay_gradcam(img_rgb, heatmap, alpha=0.5):
    heatmap_resized = cv2.resize(heatmap.astype(np.float32), IMAGE_SIZE)
    # Histogram equalization for maximum contrast
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_eq    = cv2.equalizeHist(heatmap_uint8)
    # Power boost — amplifies mid-range activations
    heatmap_norm  = (heatmap_eq / 255.0) ** 0.5
    heatmap_final   = np.uint8(255 * heatmap_norm)
    heatmap_colored = cv2.applyColorMap(heatmap_final, cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(np.array(img_rgb), IMAGE_SIZE)
    overlay     = cv2.addWeighted(img_resized, 1 - alpha, heatmap_colored, alpha, 0)
    return overlay, heatmap_norm


def preprocess_image(pil_image):
    img = pil_image.convert('RGB').resize(IMAGE_SIZE)
    arr = np.expand_dims(np.array(img).astype(np.float32), axis=0)
    return preprocess_input(arr), np.array(img)


def find_last_conv_layer(model):
    """Auto-detect the last Conv2D layer name as a fallback."""
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
    return 'conv5_block3_out'


def generate_report(pred_class, confidence, predictions, class_names):
    now   = datetime.datetime.now().strftime("%Y-%m-%d  %H:%M")
    lines = [
        f"BRAIN MRI CLASSIFICATION REPORT",
        f"Generated : {now}",
        f"Model     : ResNet50V2 (Transfer Learning)",
        f"{'─'*44}",
        f"PRIMARY FINDING",
        f"  Class      : {CLASS_LABELS.get(pred_class, pred_class)}",
        f"  Confidence : {confidence:.1f}%",
        f"  Note       : {CLASS_DESCRIPTIONS.get(pred_class, '')}",
        f"{'─'*44}",
        f"DIFFERENTIAL PROBABILITIES",
    ]
    sorted_preds = sorted(
        zip(class_names, predictions[0]),
        key=lambda x: x[1], reverse=True
    )
    for cls, prob in sorted_preds:
        bar_len = int(float(prob) * 20)
        bar     = '█' * bar_len + '░' * (20 - bar_len)
        lines.append(
            f"  {CLASS_LABELS.get(cls, cls):<22} {bar}  {float(prob)*100:5.1f}%"
        )
    lines += [
        f"{'─'*44}",
        f"⚠  FOR RESEARCH USE ONLY.",
        f"   Always consult a qualified radiologist.",
    ]
    return "\n".join(lines)


# ── Sidebar ───────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding:20px 0 24px;'>
        <div style='font-size:52px; filter:drop-shadow(0 0 12px #4fc3f766)'>🧠</div>
        <div style='font-family:Space Mono,monospace; font-size:16px;
                    font-weight:700; color:#4fc3f7; margin-top:10px;
                    line-height:1.3;'>
            Brain MRI<br>Tumor Classifier
        </div>
        <div style='font-size:10px; color:#44607a; margin-top:6px;
                    letter-spacing:1.5px; text-transform:uppercase;'>
            ResNet50V2 — Transfer Learning
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(
        "<div class='section-header'>Model Performance</div>",
        unsafe_allow_html=True
    )
    for value, label in [
        ("90.65%",  "Test Accuracy"),
        ("0.8728",  "Cohen's Kappa"),
        ("0.9796",  "Macro AUC"),
        ("89.9 MB", "Model Size"),
    ]:
        st.markdown(f"""
        <div class='metric-card'>
            <p class='metric-value'>{value}</p>
            <p class='metric-label'>{label}</p>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        "<div class='section-header'>Classes</div>",
        unsafe_allow_html=True
    )
    for cls, label in CLASS_LABELS.items():
        color = CLASS_COLORS[cls]
        st.markdown(f"""
        <div style='display:flex; align-items:center; margin:7px 0;'>
            <div style='width:10px; height:10px; border-radius:50%;
                        background:{color}; margin-right:10px; flex-shrink:0;
                        box-shadow:0 0 6px {color}88;'></div>
            <span style='font-size:12px; color:#8899b0;'>{label}</span>
        </div>""", unsafe_allow_html=True)

    # Grad-CAM alpha slider
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        "<div class='section-header'>Grad-CAM Settings</div>",
        unsafe_allow_html=True
    )
    gradcam_alpha = st.slider(
        "Heatmap Opacity", 0.2, 0.9, 0.5, 0.05,
        help="Controls how strongly the heatmap overlays the scan"
    )

    st.markdown("""
    <div class='disclaimer'>
        ⚠ For research purposes only.<br>
        Not intended for clinical diagnosis.<br>
        Always consult a qualified radiologist.
    </div>""", unsafe_allow_html=True)


# ── Main content ──────────────────────────────────────────────────
st.markdown("""
<div style='padding:10px 0 22px;'>
    <h1 style='font-family:Space Mono,monospace; color:#4fc3f7;
               font-size:26px; font-weight:700; margin:0;
               letter-spacing:1px;'>
        Brain MRI Tumor Classification
    </h1>
    <p style='color:#44607a; font-size:13px; margin:6px 0 0;'>
        Upload an MRI scan to classify tumor type using ResNet50V2
        with Grad-CAM visual explainability
    </p>
</div>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Upload MRI Image",
    type=['jpg', 'jpeg', 'png'],
    help="Upload a brain MRI scan (JPG or PNG)"
)

if uploaded_file is not None:
    pil_image           = Image.open(uploaded_file)
    img_array, img_rgb  = preprocess_image(pil_image)

    with st.spinner("Analysing MRI scan…"):
        predictions = model.predict(img_array, verbose=0)
        pred_idx    = int(np.argmax(predictions[0]))
        pred_class  = class_names[pred_idx]
        confidence  = float(predictions[0][pred_idx]) * 100

        # Auto-detect last conv layer in case name differs
        last_conv   = 'conv5_block3_out'
        try:
            model.get_layer(last_conv)
        except ValueError:
            last_conv = find_last_conv_layer(model)
            st.info(f"Auto-detected last conv layer: {last_conv}")

        heatmap              = make_gradcam(img_array, model, last_conv)
        overlay, heatmap_raw = overlay_gradcam(img_rgb, heatmap, alpha=gradcam_alpha)

    # ── Uncertainty detection ─────────────────────────────────────
    sorted_probs = sorted(predictions[0], reverse=True)
    gap          = sorted_probs[0] - sorted_probs[1]

    if gap < UNCERTAINTY_THRESHOLD:
        top2_indices = np.argsort(predictions[0])[-2:][::-1]
        c1_name      = CLASS_LABELS.get(class_names[top2_indices[0]], class_names[top2_indices[0]])
        c2_name      = CLASS_LABELS.get(class_names[top2_indices[1]], class_names[top2_indices[1]])
        c1_color     = CLASS_COLORS.get(class_names[top2_indices[0]], '#4fc3f7')
        c2_color     = CLASS_COLORS.get(class_names[top2_indices[1]], '#4fc3f7')
        c1_pct       = float(predictions[0][top2_indices[0]]) * 100
        c2_pct       = float(predictions[0][top2_indices[1]]) * 100

        st.markdown(f"""
        <div class='uncertainty-banner'>
            <div class='uncertainty-icon'>⚠️</div>
            <div>
                <div class='uncertainty-title'>Ambiguous Prediction — Low Confidence Gap</div>
                <div class='uncertainty-text'>
                    The model is uncertain between two classes
                    (gap: <strong>{gap*100:.1f}%</strong>).
                    Results should be interpreted with extra caution and reviewed by a specialist.
                </div>
                <div class='uncertainty-classes'>
                    <span class='uncertainty-class-chip'
                          style='background:{c1_color}22; color:{c1_color};
                                 border:1px solid {c1_color}66;'>
                        {c1_name}: {c1_pct:.1f}%
                    </span>
                    <span class='uncertainty-class-chip'
                          style='background:{c2_color}22; color:{c2_color};
                                 border:1px solid {c2_color}66;'>
                        {c2_name}: {c2_pct:.1f}%
                    </span>
                </div>
            </div>
        </div>""", unsafe_allow_html=True)

    # ── Three-column layout ───────────────────────────────────────
    col1, col2, col3 = st.columns([1, 1, 1])

    with col1:
        st.markdown(
            "<div class='section-header'>Original Scan</div>",
            unsafe_allow_html=True
        )
        st.markdown("<div class='zoom-wrapper'>", unsafe_allow_html=True)
        st.image(img_rgb, caption="Uploaded MRI", width='stretch')
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown(
            "<div class='section-header'>Grad-CAM Heatmap</div>",
            unsafe_allow_html=True
        )
        st.markdown("<div class='zoom-wrapper'>", unsafe_allow_html=True)
        st.image(overlay, caption="Model attention regions", width='stretch')
        st.markdown("</div>", unsafe_allow_html=True)

        # Heatmap intensity legend
        st.markdown("""
        <div class='heatmap-legend'>
            <div class='heatmap-legend-bar'></div>
            <div class='heatmap-legend-labels'>
                <span>Low</span>
                <span>Medium</span>
                <span>High</span>
            </div>
        </div>""", unsafe_allow_html=True)

        # Pure heatmap toggle
        with st.expander("🔬 View raw heatmap"):
            st.image(
                (heatmap_raw * 255).astype(np.uint8),
                caption="Raw Grad-CAM (grayscale)",
                width='stretch',
                clamp=True
            )

    with col3:
        st.markdown(
            "<div class='section-header'>Classification Result</div>",
            unsafe_allow_html=True
        )

        color = CLASS_COLORS.get(pred_class, '#4fc3f7')
        label = CLASS_LABELS.get(pred_class, pred_class)
        desc  = CLASS_DESCRIPTIONS.get(pred_class, '')

        st.markdown(f"""
        <div class='prediction-box' style='border-color:{color};'>
            <p class='prediction-class' style='color:{color};'>{label}</p>
            <p class='prediction-conf'>Confidence: {confidence:.1f}%</p>
            <p style='font-size:11px; color:#667788; margin-top:10px;
                      line-height:1.5;'>{desc}</p>
        </div>""", unsafe_allow_html=True)

        st.markdown(
            "<div class='section-header' style='margin-top:18px;'>"
            "Class Probabilities</div>",
            unsafe_allow_html=True
        )

        sorted_preds = sorted(
            zip(class_names, predictions[0]),
            key=lambda x: x[1], reverse=True
        )

        for cls, prob in sorted_preds:
            pct   = float(prob) * 100
            c     = CLASS_COLORS.get(cls, '#4fc3f7')
            lbl   = CLASS_LABELS.get(cls, cls)
            bold  = "font-weight:700;" if cls == pred_class else ""
            st.markdown(f"""
            <div class='bar-container'>
                <div class='bar-label'>
                    <span style='{bold} color:{"#e0e6f0" if cls == pred_class else "#8899b0"}'>{lbl}</span>
                    <span style='color:{c}; font-weight:600;
                                 font-family:Space Mono,monospace;'>{pct:.1f}%</span>
                </div>
                <div class='bar-track'>
                    <div class='bar-fill' style='width:{pct:.1f}%; background:{c};
                                                  box-shadow:0 0 8px {c}66;'></div>
                </div>
            </div>""", unsafe_allow_html=True)

    # ── Report section ────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    with st.expander("📋 Generate Clinical Summary Report"):
        report_text = generate_report(pred_class, confidence, predictions, class_names)
        st.markdown(
            f"<div class='report-box'>{report_text}</div>",
            unsafe_allow_html=True
        )
        st.download_button(
            label="⬇️ Download Report (.txt)",
            data=report_text,
            file_name=f"mri_report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain"
        )

else:
    st.markdown("""
    <div style='text-align:center; padding:70px 20px;
                background:#080f1e; border:2px dashed #1a3a5c;
                border-radius:16px; margin-top:24px;'>
        <div style='font-size:64px; margin-bottom:16px;
                    filter:drop-shadow(0 0 20px #4fc3f766);'>🧠</div>
        <div style='font-family:Space Mono,monospace; font-size:18px;
                    color:#4fc3f7; font-weight:700;'>
            Upload an MRI scan to begin
        </div>
        <div style='font-size:13px; color:#44607a; margin-top:10px;'>
            Supported formats: JPG, JPEG, PNG
        </div>
        <div style='font-size:11px; color:#2a4a66; margin-top:24px;
                    max-width:400px; margin-left:auto; margin-right:auto;
                    line-height:1.6;'>
            The model will classify the scan into one of four categories:<br>
            Glioma · Meningioma · Pituitary Tumor · Normal
        </div>
    </div>
    """, unsafe_allow_html=True)