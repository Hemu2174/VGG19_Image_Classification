"""
Image Classification Using VGG-19 Model
========================================
A Streamlit web application that classifies images using the pretrained
VGG-19 model from PyTorch / Torchvision (ImageNet weights).
"""

import io
import json
import urllib.request

import pandas as pd
import streamlit as st
import torch
import torchvision.models as models
from PIL import Image, UnidentifiedImageError
from torchvision.models import VGG19_Weights

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="VGG-19 Image Classifier",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS
# Design token palette
#   --bg-page      : #f0f2f6  (light grey page)
#   --bg-card      : #ffffff  (white cards)
#   --bg-card-alt  : #eef1f7  (tinted alt cards)
#   --text-primary : #1a1a2e  (near-black body text)
#   --text-secondary: #4a5568 (muted labels)
#   --accent       : #667eea  (indigo)
#   --accent-dark  : #764ba2  (purple)
#   --sidebar-bg   : #1a1a2e → #0f3460 gradient
#   --sidebar-text : #e0e0e0
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
        /* ═══════════════════════════════════════════════════════════════════
           1. PAGE & BODY
        ═══════════════════════════════════════════════════════════════════ */
        html, body,
        [data-testid="stAppViewContainer"],
        [data-testid="stMain"],
        .main {
            background-color: #f0f2f6 !important;
            color: #1a1a2e !important;
        }

        /* Ensure every plain paragraph / list / span on the main canvas
           is dark so nothing can go white-on-light.
           The file-chip text (section 6) uses a higher-specificity selector
           to override this rule for the filename/size elements. */
        .main p,
        .main li,
        .main span,
        .main label,
        .main strong,
        .main em,
        .main small,
        .main td,
        .main th {
            color: #1a1a2e !important;
        }

        /* ═══════════════════════════════════════════════════════════════════
           2. HEADINGS  (h1–h4)
        ═══════════════════════════════════════════════════════════════════ */
        h1 {
            color: #1a1a2e !important;
            font-weight: 800 !important;
            letter-spacing: -0.5px;
        }
        h2 { color: #16213e !important; font-weight: 700 !important; }
        h3 { color: #16213e !important; font-weight: 600 !important; }
        h4 { color: #2d3748 !important; font-weight: 600 !important; }

        /* ═══════════════════════════════════════════════════════════════════
           3. SIDEBAR
           Use targeted selectors rather than a wildcard (*) so that
           inline HTML inside the sidebar can still override color safely.
        ═══════════════════════════════════════════════════════════════════ */
        section[data-testid="stSidebar"] > div:first-child {
            background: linear-gradient(160deg, #1a1a2e 0%, #16213e 55%, #0f3460 100%);
        }

        /* Direct text nodes rendered by Streamlit's markdown inside sidebar */
        section[data-testid="stSidebar"] p,
        section[data-testid="stSidebar"] li,
        section[data-testid="stSidebar"] span:not([data-testid]),
        section[data-testid="stSidebar"] h1,
        section[data-testid="stSidebar"] h2,
        section[data-testid="stSidebar"] h3,
        section[data-testid="stSidebar"] h4,
        section[data-testid="stSidebar"] label,
        section[data-testid="stSidebar"] strong,
        section[data-testid="stSidebar"] em,
        section[data-testid="stSidebar"] small,
        section[data-testid="stSidebar"] .stCaption,
        section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {
            color: #dde3f0 !important;
        }

        /* Sidebar table cells */
        section[data-testid="stSidebar"] td,
        section[data-testid="stSidebar"] th {
            color: #dde3f0 !important;
            border-color: rgba(255,255,255,0.15) !important;
        }

        /* Sidebar code / inline code */
        section[data-testid="stSidebar"] code {
            color: #a5b4fc !important;
            background: rgba(255,255,255,0.1) !important;
        }

        /* Sidebar horizontal rule */
        section[data-testid="stSidebar"] hr {
            border-color: rgba(255,255,255,0.2) !important;
        }

        /* Sidebar links */
        section[data-testid="stSidebar"] a {
            color: #93c5fd !important;
        }
        section[data-testid="stSidebar"] a:hover {
            color: #bfdbfe !important;
            text-decoration: underline;
        }

        /* ═══════════════════════════════════════════════════════════════════
           4. IMAGE METADATA CARD
           Explicit dark-on-light card so Streamlit theme cannot bleed in.
        ═══════════════════════════════════════════════════════════════════ */
        .info-card {
            background: #ffffff;
            border: 1px solid #d1d9e6;
            border-left: 4px solid #667eea;
            border-radius: 12px;
            padding: 1.1rem 1.4rem;
            box-shadow: 0 2px 10px rgba(102,126,234,0.08);
            margin-top: 0.75rem;
            margin-bottom: 1rem;
            /* Force colour context so nothing inherits a dark theme */
            color-scheme: light;
        }

        /* Every element INSIDE .info-card must be dark */
        .info-card,
        .info-card p,
        .info-card span,
        .info-card strong,
        .info-card em,
        .info-card br {
            color: #1a1a2e !important;
        }

        .info-card .card-row {
            display: flex;
            align-items: baseline;
            gap: 0.5rem;
            padding: 0.3rem 0;
            border-bottom: 1px solid #eef1f7;
        }
        .info-card .card-row:last-child { border-bottom: none; }

        .info-card .card-label {
            font-size: 0.78rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            color: #667eea !important;
            min-width: 105px;
            flex-shrink: 0;
        }

        .info-card .card-value {
            font-size: 0.92rem;
            color: #1a1a2e !important;
            font-weight: 500;
            word-break: break-all;
        }

        /* ═══════════════════════════════════════════════════════════════════
           5. TOP-PREDICTION BADGE
        ═══════════════════════════════════════════════════════════════════ */
        .prediction-badge {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border-radius: 12px;
            padding: 1.1rem 1.5rem;
            text-align: center;
            margin-bottom: 1.1rem;
            box-shadow: 0 4px 15px rgba(102,126,234,0.35);
        }
        .prediction-badge .badge-label {
            font-size: 1.3rem;
            font-weight: 800;
            color: #ffffff !important;
            display: block;
            margin-bottom: 0.3rem;
        }
        .prediction-badge .badge-conf {
            font-size: 0.95rem;
            font-weight: 400;
            color: #e0e7ff !important;
            display: block;
        }

        /* ═══════════════════════════════════════════════════════════════════
           6. FILE UPLOADER
        ═══════════════════════════════════════════════════════════════════ */

        /* Outer wrapper – white card with indigo dashed border */
        [data-testid="stFileUploader"] {
            background: #ffffff !important;
            border: 2px dashed #667eea !important;
            border-radius: 14px !important;
            padding: 0.4rem 0.6rem !important;
            box-shadow: 0 2px 10px rgba(102,126,234,0.08);
        }

        /* Uploader label ("Choose an image file") */
        [data-testid="stFileUploader"] > label,
        [data-testid="stFileUploader"] > label p,
        [data-testid="stFileUploader"] > label span {
            color: #1a1a2e !important;
            font-weight: 600 !important;
            font-size: 0.9rem !important;
        }

        /* Dropzone inner box – force light background so text is readable */
        [data-testid="stFileUploaderDropzone"] {
            background: #f5f7ff !important;
            border: none !important;
            border-radius: 10px !important;
        }

        /* "Drag and drop file here" heading text */
        [data-testid="stFileUploaderDropzoneInstructions"] p,
        [data-testid="stFileUploaderDropzoneInstructions"] span {
            color: #1a1a2e !important;
            font-weight: 600 !important;
        }

        /* "Limit 200MB per file • JPG, JPEG, PNG" small line */
        [data-testid="stFileUploaderDropzoneInstructions"] small,
        [data-testid="stFileUploaderDropzoneInstructions"] small span {
            color: #667eea !important;
            font-weight: 500 !important;
        }

        /* Cloud / upload icon */
        [data-testid="stFileUploaderDropzone"] svg {
            color: #667eea !important;
            fill: #667eea !important;
        }

        /* "Browse files" button – scoped ONLY inside the dropzone so the
           [data-testid="stBaseButton-secondary"] rule never bleeds out to
           the toolbar Deploy button or the ⋮ overflow menu.             */
        [data-testid="stFileUploaderDropzone"] button {
            background: #667eea !important;
            color: #ffffff !important;
            border: none !important;
            border-radius: 8px !important;
            font-weight: 600 !important;
            padding: 0.35rem 1rem !important;
            cursor: pointer !important;
        }
        [data-testid="stFileUploaderDropzone"] button:hover {
            background: #5a6fd6 !important;
        }

        /* ── Uploaded file chip: container ── */
        [data-testid="stFileUploader"] [data-testid="stFileUploaderFile"],
        [data-testid="stFileUploaderFile"] {
            background: #F3F4F6 !important;
            border: 1px solid #d1d9e6 !important;
            border-radius: 8px !important;
            margin-top: 0.4rem !important;
            padding: 0.35rem 0.7rem !important;
            /* Isolate colour-scheme so Streamlit dark theme cannot bleed in */
            color-scheme: light !important;
        }

        /* ── Filename text ──────────────────────────────────────────────────
           Specificity ladder: html body [attr] [attr] = 4 tokens,
           which cleanly beats .main span (2 tokens) from section 1.
           This is the correct fix – the previous :not() approach was
           invalid CSS3 (descendant selectors inside :not() are Level 4
           only) and silently fell back to matching everything.           */
        html body [data-testid="stFileUploaderFile"] span,
        html body [data-testid="stFileUploaderFile"] p,
        html body [data-testid="stFileUploaderFile"] small,
        html body [data-testid="stFileUploader"] [data-testid="stFileUploaderFile"] span,
        html body [data-testid="stFileUploader"] [data-testid="stFileUploaderFile"] p,
        html body [data-testid="stFileUploader"] [data-testid="stFileUploaderFile"] small {
            color: #1F2937 !important;
            font-weight: 500 !important;
        }

        /* ── File-type icon next to the name ── */
        [data-testid="stFileUploaderFile"] svg,
        [data-testid="stFileUploader"] [data-testid="stFileUploaderFile"] svg {
            color: #667eea !important;
            fill: #667eea !important;
        }

        /* ── × remove button ── */
        [data-testid="stFileUploaderDeleteBtn"] button,
        [data-testid="stFileUploader"] [data-testid="stFileUploaderDeleteBtn"] button {
            background: transparent !important;
            color: #4a5568 !important;
            border: none !important;
        }
        [data-testid="stFileUploaderDeleteBtn"] button:hover,
        [data-testid="stFileUploader"] [data-testid="stFileUploaderDeleteBtn"] button:hover {
            color: #764ba2 !important;
        }

        /* ═══════════════════════════════════════════════════════════════════
           7. EXPANDERS
        ═══════════════════════════════════════════════════════════════════ */
        [data-testid="stExpander"] {
            background: #ffffff;
            border: 1px solid #d1d9e6;
            border-radius: 10px;
            overflow: hidden;
        }
        [data-testid="stExpander"] summary p,
        [data-testid="stExpander"] summary span {
            color: #1a1a2e !important;
            font-weight: 600;
        }
        [data-testid="stExpander"] [data-testid="stMarkdownContainer"] p,
        [data-testid="stExpander"] [data-testid="stMarkdownContainer"] li,
        [data-testid="stExpander"] [data-testid="stMarkdownContainer"] td,
        [data-testid="stExpander"] [data-testid="stMarkdownContainer"] th {
            color: #2d3748 !important;
        }
        [data-testid="stExpander"] code {
            background: #eef1f7 !important;
            color: #667eea !important;
        }

        /* ═══════════════════════════════════════════════════════════════════
           8. ALERT BOXES  (info / warning / error / success)
        ═══════════════════════════════════════════════════════════════════ */
        [data-testid="stAlert"] p,
        [data-testid="stAlert"] span {
            color: #1a1a2e !important;
        }

        /* ═══════════════════════════════════════════════════════════════════
           9. DATAFRAME / TABLE
        ═══════════════════════════════════════════════════════════════════ */
        [data-testid="stDataFrame"] {
            border-radius: 10px;
            overflow: hidden;
            border: 1px solid #d1d9e6;
        }

        /* ═══════════════════════════════════════════════════════════════════
           10. SPINNER TEXT
        ═══════════════════════════════════════════════════════════════════ */
        [data-testid="stSpinner"] p,
        [data-testid="stSpinner"] span {
            color: #667eea !important;
            font-weight: 500;
        }

        /* ═══════════════════════════════════════════════════════════════════
           11. CAPTION / SMALL TEXT
        ═══════════════════════════════════════════════════════════════════ */
        .stCaption,
        [data-testid="stCaptionContainer"] p {
            color: #718096 !important;
        }

        /* ═══════════════════════════════════════════════════════════════════
           12. HORIZONTAL RULE on main canvas
        ═══════════════════════════════════════════════════════════════════ */
        .main hr {
            border-color: #d1d9e6 !important;
        }

        /* ═══════════════════════════════════════════════════════════════════
           13. SECTION SUBHEADING (subtitle paragraph under h1)
        ═══════════════════════════════════════════════════════════════════ */
        [data-testid="stMarkdownContainer"] > p:first-of-type {
            color: #4a5568 !important;
            font-size: 1rem;
        }

        /* ═══════════════════════════════════════════════════════════════════
           14. TOP TOOLBAR  (Deploy button + ⋮ overflow menu)
           Streamlit's header bar sits in [data-testid="stHeader"].
           Both the Deploy label and the overflow icon are rendered as
           stBaseButton-secondary inside that header.  We give the whole
           header a clean white/light background and force dark text so
           every control is readable regardless of the page theme.
        ═══════════════════════════════════════════════════════════════════ */
        [data-testid="stHeader"] {
            background-color: #ffffff !important;
            border-bottom: 1px solid #e2e8f0 !important;
        }

        /* All text/icons inside the header toolbar */
        [data-testid="stHeader"] span,
        [data-testid="stHeader"] p,
        [data-testid="stHeader"] svg {
            color: #1a1a2e !important;
            fill: #1a1a2e !important;
        }

        /* Deploy button and ⋮ overflow button inside the header only */
        [data-testid="stHeader"] button,
        [data-testid="stHeader"] [data-testid="stBaseButton-secondary"] {
            background: transparent !important;
            color: #1a1a2e !important;
            border: 1px solid #d1d9e6 !important;
            border-radius: 6px !important;
        }
        [data-testid="stHeader"] button:hover,
        [data-testid="stHeader"] [data-testid="stBaseButton-secondary"]:hover {
            background: #f0f2f6 !important;
            border-color: #667eea !important;
            color: #667eea !important;
        }
        [data-testid="stHeader"] button:hover svg,
        [data-testid="stHeader"] [data-testid="stBaseButton-secondary"]:hover svg {
            color: #667eea !important;
            fill: #667eea !important;
        }

        /* "Deploy" text label specifically */
        [data-testid="stHeader"] [data-testid="stBaseButton-secondary"] span {
            color: #1a1a2e !important;
            font-weight: 600 !important;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# ImageNet class labels  (1 000 classes)
# Fetched once and cached so the app works fully offline on subsequent runs.
# ---------------------------------------------------------------------------
IMAGENET_LABELS_URL = (
    "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/"
    "master/imagenet-simple-labels.json"
)

@st.cache_resource(show_spinner=False)
def load_imagenet_labels() -> list[str]:
    """Download and cache the 1 000 ImageNet class-name labels."""
    try:
        with urllib.request.urlopen(IMAGENET_LABELS_URL, timeout=10) as resp:
            labels = json.loads(resp.read().decode())
        return labels
    except Exception:
        # Fallback: return numeric class indices if network is unavailable
        return [f"class_{i}" for i in range(1000)]


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_model() -> tuple[torch.nn.Module, object]:
    """Load pretrained VGG-19 with ImageNet weights and return (model, weights)."""
    weights = VGG19_Weights.DEFAULT
    model = models.vgg19(weights=weights)
    model.eval()
    return model, weights


# ---------------------------------------------------------------------------
# Inference helper
# ---------------------------------------------------------------------------
def classify_image(
    image: Image.Image,
    model: torch.nn.Module,
    preprocess,
    labels: list[str],
    top_k: int = 5,
) -> list[dict]:
    """
    Run a forward pass and return the top-k predictions.

    Returns
    -------
    list of dict  →  [{"rank": 1, "label": "...", "confidence": 0.95}, ...]
    """
    # Ensure RGB (handles RGBA / grayscale PNGs gracefully)
    image_rgb = image.convert("RGB")

    input_tensor = preprocess(image_rgb).unsqueeze(0)  # (1, 3, 224, 224)

    with torch.no_grad():
        logits = model(input_tensor)          # (1, 1000)
        probs = torch.softmax(logits, dim=1)  # convert to probabilities

    top_probs, top_indices = torch.topk(probs, top_k)

    results = []
    for rank, (prob, idx) in enumerate(
        zip(top_probs[0].tolist(), top_indices[0].tolist()), start=1
    ):
        label = labels[idx] if idx < len(labels) else f"class_{idx}"
        # Capitalise and replace underscores for readability
        label = label.replace("_", " ").title()
        results.append({"rank": rank, "label": label, "confidence": prob})

    return results


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
def render_sidebar() -> None:
    with st.sidebar:
        st.markdown("## 🔬 VGG-19 Classifier")
        st.markdown("---")

        st.markdown("### 📌 About This Project")
        st.markdown(
            """
            This academic application demonstrates **deep learning-based image
            classification** using the **VGG-19** convolutional neural network
            pretrained on the [ImageNet](https://www.image-net.org/) dataset
            (1 000 object categories).
            """
        )

        st.markdown("---")
        st.markdown("### 🧠 Model Details")
        st.markdown(
            """
            | Property | Value |
            |----------|-------|
            | Architecture | VGG-19 |
            | Input size | 224 × 224 px |
            | Parameters | ~143 M |
            | Dataset | ImageNet-1K |
            | Framework | PyTorch |
            | Weights source | `torchvision` |
            """
        )

        st.markdown("---")
        st.markdown("### 🔄 Application Workflow")
        st.markdown(
            """
            1. **Upload** a JPG / PNG image  
            2. **Preprocess** – resize, centre-crop, normalise  
            3. **Forward pass** through VGG-19  
            4. **Softmax** → probability over 1 000 classes  
            5. **Display** top-5 results + chart  
            """
        )

        st.markdown("---")
        st.markdown("### ⚠️ Limitations")
        st.markdown(
            """
            - Recognises only ImageNet-1K categories  
            - Performance degrades on very low-resolution images  
            - Computationally heavy (CPU inference may be slow)  
            - Not suitable for medical / safety-critical use  
            """
        )

        st.markdown("---")
        st.markdown("### 🚀 Future Enhancements")
        st.markdown(
            """
            - Fine-tune on custom datasets  
            - Add ResNet / EfficientNet comparison  
            - Enable GPU / CUDA acceleration  
            - Implement Grad-CAM visualisations  
            - Deploy via Streamlit Cloud or HuggingFace Spaces  
            """
        )

        st.markdown("---")
        st.caption("Built with ❤️ using PyTorch & Streamlit")


# ---------------------------------------------------------------------------
# Main page
# ---------------------------------------------------------------------------
def main() -> None:
    render_sidebar()

    # ── Header ──────────────────────────────────────────────────────────────
    st.markdown("# 🔬 Image Classification Using VGG-19")
    st.markdown(
        "Upload any image and the pretrained **VGG-19** model will predict "
        "the top-5 object categories from ImageNet."
    )
    st.markdown("---")

    # ── Load assets (cached after first run) ────────────────────────────────
    with st.spinner("Loading VGG-19 model … (first run may take ~30 s)"):
        model, weights = load_model()
        preprocess = weights.transforms()

    with st.spinner("Loading ImageNet labels …"):
        labels = load_imagenet_labels()

    # ── File uploader ────────────────────────────────────────────────────────
    uploaded_file = st.file_uploader(
        "📂 Choose an image file",
        type=["jpg", "jpeg", "png"],
        help="Supported formats: JPG, JPEG, PNG",
    )

    if uploaded_file is None:
        st.info("👆 Upload an image above to get started.")

        # Demo tip
        with st.expander("💡 Tips for best results"):
            st.markdown(
                """
                - Use clear, well-lit photos  
                - Centre the main object in the frame  
                - Avoid heavily cropped or abstract images  
                - All 1 000 ImageNet categories work (animals, vehicles, food, etc.)  
                """
            )
        return

    # ── Load & validate image ────────────────────────────────────────────────
    try:
        image_bytes = uploaded_file.read()
        image = Image.open(io.BytesIO(image_bytes))
        image.verify()                        # detect truncated / corrupt files
        image = Image.open(io.BytesIO(image_bytes))  # reopen after verify()
    except UnidentifiedImageError:
        st.error("❌ The uploaded file is not a valid image. Please try another file.")
        return
    except Exception as exc:
        st.error(f"❌ Could not open the image: {exc}")
        return

    # ── Layout: image left, results right ───────────────────────────────────
    col_img, col_res = st.columns([1, 1], gap="large")

    with col_img:
        st.markdown("### 🖼️ Uploaded Image")
        st.image(image, use_container_width=True, caption=uploaded_file.name)

        w, h = image.size
        mode = image.mode
        st.markdown(
            f"""
            <div class="info-card">
                <div class="card-row">
                    <span class="card-label">📐 Dimensions</span>
                    <span class="card-value">{w} &times; {h} px</span>
                </div>
                <div class="card-row">
                    <span class="card-label">🎨 Colour mode</span>
                    <span class="card-value">{mode}</span>
                </div>
                <div class="card-row">
                    <span class="card-label">📄 File name</span>
                    <span class="card-value">{uploaded_file.name}</span>
                </div>
                <div class="card-row">
                    <span class="card-label">📦 File size</span>
                    <span class="card-value">{len(image_bytes) / 1024:.1f} KB</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ── Run inference ────────────────────────────────────────────────────────
    with col_res:
        st.markdown("### 🤖 Prediction Results")

        with st.spinner("Classifying … please wait"):
            try:
                predictions = classify_image(image, model, preprocess, labels, top_k=5)
            except Exception as exc:
                st.error(f"❌ Inference failed: {exc}")
                return

        if not predictions:
            st.warning("No predictions returned. Please try a different image.")
            return

        # Top-1 badge
        top = predictions[0]
        st.markdown(
            f"""
            <div class="prediction-badge">
                <span class="badge-label">🏆 {top['label']}</span>
                <span class="badge-conf">Confidence: {top['confidence'] * 100:.2f}%</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # ── Top-5 table ──────────────────────────────────────────────────────
        st.markdown("#### 📊 Top-5 Predictions")

        df = pd.DataFrame(predictions)
        df["confidence_pct"] = df["confidence"].apply(lambda x: f"{x * 100:.2f}%")
        df_display = df[["rank", "label", "confidence_pct"]].rename(
            columns={"rank": "Rank", "label": "Category", "confidence_pct": "Confidence"}
        )
        st.dataframe(df_display, use_container_width=True, hide_index=True)

        # ── Bar chart ────────────────────────────────────────────────────────
        st.markdown("#### 📈 Confidence Chart")

        chart_df = pd.DataFrame(
            {
                "Category": [p["label"] for p in predictions],
                "Confidence (%)": [round(p["confidence"] * 100, 2) for p in predictions],
            }
        )
        st.bar_chart(chart_df.set_index("Category"), use_container_width=True)

    # ── Model info footer ────────────────────────────────────────────────────
    st.markdown("---")
    with st.expander("ℹ️ How VGG-19 Works"):
        st.markdown(
            """
            **VGG-19** (Visual Geometry Group, Oxford, 2014) is a deep CNN with:

            | Block | Layers | Filters |
            |-------|--------|---------|
            | Block 1 | 2 Conv + MaxPool | 64 |
            | Block 2 | 2 Conv + MaxPool | 128 |
            | Block 3 | 4 Conv + MaxPool | 256 |
            | Block 4 | 4 Conv + MaxPool | 512 |
            | Block 5 | 4 Conv + MaxPool | 512 |
            | FC | 3 Fully-Connected | 4096 → 4096 → 1000 |

            **Preprocessing pipeline** applied to every input:
            1. Resize shorter edge to **256 px**  
            2. Centre-crop to **224 × 224 px**  
            3. Normalise with ImageNet mean `[0.485, 0.456, 0.406]`
               and std `[0.229, 0.224, 0.225]`

            The final layer outputs 1 000 logits → **Softmax** converts them to
            probabilities.  The highest probability class is the predicted label.
            """
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    main()
