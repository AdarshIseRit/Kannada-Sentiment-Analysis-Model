import os
import sys
from io import BytesIO
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import streamlit as st

# Ensure project root is on sys.path so we can import agents.
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from agents.prediction_agent import PredictionAgent  # noqa: E402

st.set_page_config(
    page_title="BhavaSpandana AI — Glass Iris Dashboard",
    layout="wide",
    page_icon="🌸",
)

# Emotion metadata with Kannada translations, icons, and theme gradients
EMOTION_META = {
    "happy": {
        "kannada": "ಸಂತೋಷ",
        "emoji": "😊",
        "color": "#16A34A",
        "gradient": "linear-gradient(135deg, #15803D 0%, #166534 100%)",
        "desc": "ಸಂತೋಷ ಹಾಗೂ ಉತ್ಸಾಹಭರಿತ ಭಾವನೆ (Joy & Contentment)",
    },
    "sad": {
        "kannada": "ದುಃಖ",
        "emoji": "😢",
        "color": "#2563EB",
        "gradient": "linear-gradient(135deg, #1D4ED8 0%, #1E40AF 100%)",
        "desc": "ದುಃಖ ಅಥವಾ ನೋವಿನ ಭಾವನೆ (Sorrow & Melancholy)",
    },
    "anger": {
        "kannada": "ಕೋಪ",
        "emoji": "😠",
        "color": "#DC2626",
        "gradient": "linear-gradient(135deg, #B91C1C 0%, #991B1B 100%)",
        "desc": "ಕೋಪ ಹಾಗೂ ಅಸಮಾಧಾನ (Anger & Frustration)",
    },
    "fear": {
        "kannada": "ಭಯ",
        "emoji": "😨",
        "color": "#EA580C",
        "gradient": "linear-gradient(135deg, #C2410C 0%, #9A3412 100%)",
        "desc": "ಭಯ ಹಾಗೂ ಆತಂಕ (Fear & Anxiety)",
    },
    "disgust": {
        "kannada": "ಅಸಹ್ಯ",
        "emoji": "🤢",
        "color": "#9333EA",
        "gradient": "linear-gradient(135deg, #7E22CE 0%, #6B21A8 100%)",
        "desc": "ಅಸಹ್ಯ ಅಥವಾ ಅಸಮ್ಮತಿ (Disgust & Aversion)",
    },
    "neutral": {
        "kannada": "ತಟಸ್ಥ",
        "emoji": "😐",
        "color": "#4B5563",
        "gradient": "linear-gradient(135deg, #4B5563 0%, #374151 100%)",
        "desc": "ಸಾಮಾನ್ಯ ಅಥವಾ ಸಮಚಿತ್ತ ಭಾವನೆ (Neutral & Balanced)",
    },
}

# Glass Iris CSS System
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter+Tight:wght@400;500;600;700&display=swap');

/* Base page theme */
html, body, [data-testid="stAppViewContainer"] {
    background-color: #0E0E0E !important;
    font-family: 'Inter Tight', system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif !important;
    color: #FFFFFF !important;
}

[data-testid="stHeader"] {
    background: transparent !important;
}

.main .block-container {
    padding-top: 1rem !important;
    padding-bottom: 3rem !important;
    max-width: 1040px !important;
}

/* Electric blue top ambient glow */
.top-ambient-glow {
    position: fixed;
    top: 0;
    left: 0;
    right: 0;
    height: 380px;
    background: radial-gradient(circle at 50% -10%, rgba(64,110,255,0.48) 0%, rgba(124,72,240,0.22) 40%, rgba(14,14,14,0) 75%);
    pointer-events: none;
    z-index: 0;
}

/* Glass Shell Top Bar */
.glass-top-bar {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0.75rem 1.25rem;
    background: rgba(21, 21, 21, 0.75);
    backdrop-filter: blur(16px);
    border: 1px solid rgba(255, 255, 255, 0.07);
    border-radius: 14px;
    margin-bottom: 1.25rem;
}

.breadcrumb-nav {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    font-size: 0.88rem;
    color: #8C8C8C;
}

.breadcrumb-active {
    color: #FFFFFF;
    font-weight: 600;
}

.status-pill {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.35rem 0.85rem;
    background: rgba(47, 107, 255, 0.12);
    border: 1px solid rgba(47, 107, 255, 0.35);
    border-radius: 9999px;
    font-size: 0.8rem;
    font-weight: 500;
    color: #E0E7FF;
}

.status-dot {
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: #2F6BFF;
    box-shadow: 0 0 8px #2F6BFF;
}

/* Glass Iris Orb */
.orb-container {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    margin: 1.2rem auto 0.8rem auto;
    position: relative;
}

.living-orb {
    width: 104px;
    height: 104px;
    border-radius: 50%;
    position: relative;
    background: radial-gradient(circle at 36% 34%, rgba(255, 255, 255, 0.6) 0%, rgba(47, 107, 255, 0.85) 30%, rgba(90, 76, 245, 0.95) 58%, rgba(14, 14, 14, 0.98) 90%);
    box-shadow: 
        0 0 45px rgba(47, 107, 255, 0.55),
        0 0 95px rgba(124, 72, 240, 0.35),
        inset 0 0 25px rgba(255, 255, 255, 0.45),
        inset -8px -8px 22px rgba(0, 0, 0, 0.85);
    animation: orbPulse 4s ease-in-out infinite alternate;
}

.living-orb::before {
    content: '';
    position: absolute;
    top: 14%;
    left: 20%;
    width: 32%;
    height: 22%;
    border-radius: 50%;
    background: radial-gradient(ellipse at center, rgba(255, 255, 255, 0.9) 0%, rgba(255, 255, 255, 0) 75%);
    transform: rotate(-25deg);
}

.orb-ring {
    position: absolute;
    width: 136px;
    height: 136px;
    border-radius: 50%;
    border: 1px dashed rgba(64, 110, 255, 0.28);
    animation: ringSpin 20s linear infinite;
}

@keyframes orbPulse {
    0% {
        transform: scale(0.96) translateY(0px);
        box-shadow: 0 0 35px rgba(47, 107, 255, 0.45), 0 0 75px rgba(124, 72, 240, 0.25);
    }
    100% {
        transform: scale(1.02) translateY(-6px);
        box-shadow: 0 0 60px rgba(47, 107, 255, 0.75), 0 0 115px rgba(124, 72, 240, 0.45);
    }
}

@keyframes ringSpin {
    from { transform: rotate(0deg); }
    to { transform: rotate(360deg); }
}

/* Hero Typography */
.hero-greeting {
    text-align: center;
    font-size: 0.95rem;
    color: #8C8C8C;
    margin-top: 0.6rem;
    letter-spacing: 0.04em;
    text-transform: uppercase;
}

.hero-headline {
    text-align: center;
    font-size: 1.85rem;
    font-weight: 600;
    letter-spacing: -0.01em;
    color: #FFFFFF;
    margin-top: 0.2rem;
    margin-bottom: 0.35rem;
}

.hero-subline {
    text-align: center;
    font-size: 0.96rem;
    color: #8C8C8C;
    margin-bottom: 1.5rem;
}

/* Glass Iris Composer */
.composer-wrapper {
    background: #151515;
    border-radius: 16px;
    padding: 1.25rem 1.5rem 1rem 1.5rem;
    border: 1.2px solid transparent;
    background-image: linear-gradient(#151515, #151515), linear-gradient(135deg, #2F6BFF, #6366F1, #5A4CF5, #2F6BFF);
    background-origin: border-box;
    background-clip: padding-box, border-box;
    box-shadow: 0 14px 40px rgba(0, 0, 0, 0.45);
    margin-bottom: 1.5rem;
}

/* Streamlit input styling inside composer */
[data-testid="stTextArea"] textarea {
    background: transparent !important;
    color: #FFFFFF !important;
    border: none !important;
    font-size: 1rem !important;
    font-family: inherit !important;
    box-shadow: none !important;
    padding: 0 !important;
}

[data-testid="stTextArea"] textarea:focus {
    outline: none !important;
}

/* Quick Action Cards */
.quick-cards-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 1rem;
    margin-top: 1rem;
    margin-bottom: 1.5rem;
}

.quick-card {
    background: #151515;
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 14px;
    padding: 1.15rem 1.25rem;
    transition: all 0.2s ease;
}

.quick-card:hover {
    border-color: rgba(47, 107, 255, 0.4);
    background: #1A1A1A;
    transform: translateY(-2px);
}

.quick-card-icon {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 34px;
    height: 34px;
    border-radius: 9px;
    margin-bottom: 0.75rem;
    font-size: 1rem;
}

.icon-blue { background: rgba(47, 107, 255, 0.16); color: #2F6BFF; }
.icon-violet { background: rgba(90, 76, 245, 0.16); color: #5A4CF5; }
.icon-indigo { background: rgba(99, 102, 241, 0.16); color: #818CF8; }

.quick-card-title {
    font-size: 0.95rem;
    font-weight: 600;
    color: #FFFFFF;
    margin-bottom: 0.2rem;
}

.quick-card-desc {
    font-size: 0.82rem;
    color: #8C8C8C;
    line-height: 1.35;
}

/* Glass Sentiment Output Card */
.glass-sentiment-card {
    border-radius: 16px;
    padding: 1.4rem 1.6rem;
    margin-top: 1rem;
    margin-bottom: 1.25rem;
    color: #FFFFFF;
    border: 1px solid rgba(255, 255, 255, 0.15);
    box-shadow: 0 16px 45px rgba(0, 0, 0, 0.5);
}

.badge-tag {
    display: inline-block;
    padding: 0.2rem 0.65rem;
    border-radius: 9999px;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    background: rgba(255, 255, 255, 0.16);
    backdrop-filter: blur(6px);
    margin-bottom: 0.5rem;
}

/* Custom button accents */
button[kind="primary"] {
    background: linear-gradient(135deg, #2F6BFF 0%, #5A4CF5 100%) !important;
    border: none !important;
    color: #FFFFFF !important;
    font-weight: 600 !important;
    border-radius: 10px !important;
}

button[kind="secondary"] {
    background: #1D1D1D !important;
    color: #E0E7FF !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    border-radius: 10px !important;
}
</style>
<div class="top-ambient-glow"></div>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def get_prediction_agent() -> Optional[PredictionAgent]:
    """
    Load and cache the trained PredictionAgent.
    """
    model_dir = os.path.join(ROOT_DIR, "models")
    if not os.path.exists(model_dir):
        return None
    try:
        return PredictionAgent(model_dir=model_dir)
    except Exception as exc:
        print(f"[Dashboard] Error loading model: {exc}")
        return None


def get_sentiment_color(sentiment: str) -> str:
    """
    Map sentiment labels to brand colors.
    """
    mapping: Dict[str, str] = {
        "happy": "#16A34A",
        "sad": "#2563EB",
        "angry": "#DC2626",
        "anger": "#DC2626",
        "fear": "#EA580C",
        "disgust": "#7C3AED",
        "neutral": "#6B7280",
    }
    return mapping.get(str(sentiment).lower(), "#4B5563")


def render_shell_header(agent: Optional[PredictionAgent]) -> None:
    """
    Render top glass shell bar with breadcrumbs and live AI status pill.
    """
    status_label = "AI Model Running" if agent is not None else "Model Initializing"
    device_label = f"Device: {agent.device}" if agent is not None else "Device: CPU"

    st.markdown(
        f"""
        <div class="glass-top-bar">
            <div class="breadcrumb-nav">
                <span>🌸 Home</span>
                <span>›</span>
                <span class="breadcrumb-active">BhavaSpandana AI (ಭಾವಸ್ಪಂದನ)</span>
            </div>
            <div class="status-pill">
                <div class="status-dot"></div>
                <span>{status_label}</span>
                <span style="opacity:0.5;">|</span>
                <span>{device_label}</span>
                <span style="opacity:0.5;">|</span>
                <span>96.52% Acc</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_hero_orb() -> None:
    """
    Render living glass iris orb with specular refraction and glow.
    """
    st.markdown(
        """
        <div class="orb-container">
            <div class="orb-ring"></div>
            <div class="living-orb"></div>
        </div>
        <div class="hero-greeting">ನಮಸ್ಕಾರ · KANNADA EMOTION INTELLIGENCE</div>
        <div class="hero-headline">WHAT'S ON YOUR MIND?</div>
        <div class="hero-subline">Decode contextual emotions and sentiments in Kannada text using BhavaSpandana AI.</div>
        """,
        unsafe_allow_html=True,
    )


def render_quick_action_cards() -> None:
    """
    Render three Glass Iris quick-action feature cards.
    """
    st.markdown(
        """
        <div class="quick-cards-grid">
            <div class="quick-card">
                <div class="quick-card-icon icon-blue">⚡</div>
                <div class="quick-card-title">Real-Time Inference</div>
                <div class="quick-card-desc">Type or paste Kannada text for instant soft-voting emotion classification.</div>
            </div>
            <div class="quick-card">
                <div class="quick-card-icon icon-violet">📂</div>
                <div class="quick-card-title">Batch Excel Scoring</div>
                <div class="quick-card-desc">Upload spreadsheets with kannada_text to bulk score hundreds of sentences.</div>
            </div>
            <div class="quick-card">
                <div class="quick-card-icon icon-indigo">📊</div>
                <div class="quick-card-title">Model Diagnostics</div>
                <div class="quick-card-desc">Inspect held-out confusion matrix heatmap and multi-class F1 benchmarks.</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_composer_section(agent: Optional[PredictionAgent]) -> None:
    """
    Render the central Glass Iris composer and real-time prediction output.
    """
    # Sample shortcut chips
    samples = [
        ("😊 ಸಂತೋಷ", "ನಾನು ಇಂದು ತುಂಬಾ ಸಂತೋಷವಾಗಿದ್ದೇನೆ"),
        ("😢 ದುಃಖ", "ನನಗೆ ತುಂಬಾ ದುಃಖವಾಗಿದೆ ಮತ್ತು ಬೇಸರವಾಗಿದೆ"),
        ("😠 ಕೋಪ", "ಇದು ನನಗೆ ತುಂಬಾ ಕೋಪ ತರಿಸುತ್ತದೆ"),
        ("😨 ಭಯ", "ಅದು ತುಂಬಾ ಭಯಂಕರವಾಗಿದೆ ನನಗೆ ಭಯವಾಗ್ತಿದೆ"),
        ("🤢 ಅಸಹ್ಯ", "ಇದು ತುಂಬಾ ಅಸಹ್ಯವಾಗಿದೆ"),
        ("😐 ತಟಸ್ಥ", "ಇಂದು ಹವಾಮಾನ ಸಾಮಾನ್ಯವಾಗಿದೆ ಮತ್ತು ಶಾಂತವಾಗಿದೆ"),
    ]

    col_chips = st.columns(6)
    for idx, (label, sample_text) in enumerate(samples):
        with col_chips[idx]:
            if st.button(label, key=f"chip_{idx}", use_container_width=True):
                st.session_state["composer_input"] = sample_text

    current_text = st.session_state.get("composer_input", "ನಾನು ಇಂದು ತುಂಬಾ ಸಂತೋಷವಾಗಿದ್ದೇನೆ")

    # The 114u Hairline Gradient Composer Card
    st.markdown("<div class='composer-wrapper'>", unsafe_allow_html=True)
    input_text = st.text_area(
        "Ask Anything in Kannada...",
        value=current_text,
        height=95,
        placeholder="Type or paste a Kannada sentence to analyze emotion (e.g. ನಾನು ತುಂಬಾ ಸಂತೋಷವಾಗಿದ್ದೇನೆ)...",
        label_visibility="collapsed",
    )

    col_btn_left, col_btn_right = st.columns([3, 1])
    with col_btn_left:
        st.caption("⚡ Model: **BhavaSpandana AI** (MuRIL BERT 65% + Char TF-IDF 35% Soft Voting)")
    with col_btn_right:
        predict_clicked = st.button("Predict Emotion →", type="primary", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # Perform inference and display result
    if predict_clicked or ("composer_input" in st.session_state and input_text):
        if not input_text or not input_text.strip():
            st.warning("Please enter some Kannada text before predicting.")
        elif agent is None:
            st.error("Model artifacts not found in `models/`. Please run `python train.py` first.")
        else:
            try:
                results = agent.predict_texts([input_text])
                if results:
                    res = results[0]
                    sentiment = str(res["sentiment"]).lower()
                    confidence = float(res["confidence"])
                    cleaned = str(res["cleaned"])
                    probs = res.get("probabilities", {})

                    meta = EMOTION_META.get(
                        sentiment,
                        {
                            "kannada": sentiment.title(),
                            "emoji": "🌸",
                            "color": "#2F6BFF",
                            "gradient": "linear-gradient(135deg, #2F6BFF 0%, #151515 100%)",
                            "desc": "Emotion",
                        },
                    )

                    st.markdown(
                        f"""
                        <div class="glass-sentiment-card" style="background: {meta['gradient']};">
                            <span class="badge-tag">Predicted Sentiment</span>
                            <div style="font-size: 2rem; font-weight: 700; margin-bottom: 0.2rem;">
                                {meta['emoji']} {meta['kannada']} <span style="font-size: 1.2rem; opacity: 0.85;">({sentiment.upper()})</span>
                            </div>
                            <div style="font-size: 0.95rem; opacity: 0.9; margin-bottom: 0.5rem;">{meta['desc']}</div>
                            <div style="font-size: 0.85rem; opacity: 0.8;">
                                <strong>Cleaned Text:</strong> {cleaned if cleaned else "—"}
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                    st.write(f"**Confidence**: `{confidence:.2f}%`")
                    st.progress(min(max(confidence / 100.0, 0.0), 1.0))

                    # Full emotion distribution breakdown
                    if probs:
                        with st.expander("📊 View All 6 Emotion Probabilities", expanded=False):
                            prob_rows = []
                            for k, v in probs.items():
                                m = EMOTION_META.get(k.lower(), {"kannada": k, "emoji": ""})
                                prob_rows.append({"Emotion": f"{m['emoji']} {m['kannada']} ({k.title()})", "Confidence (%)": round(v, 2)})
                            p_df = pd.DataFrame(prob_rows).sort_values(by="Confidence (%)", ascending=False)
                            st.dataframe(
                                p_df.style.bar(subset=["Confidence (%)"], color=meta["color"], vmin=0, vmax=100),
                                use_container_width=True,
                                hide_index=True,
                            )
            except Exception as exc:
                st.error(f"Prediction failed: {exc}")


def render_batch_section(agent: Optional[PredictionAgent]) -> None:
    """
    Render batch spreadsheet scoring for Excel files.
    """
    st.markdown("### 📂 Batch File Scoring")
    st.caption("Upload an Excel (`.xlsx`) or CSV file containing a `kannada_text` column.")

    uploaded_file = st.file_uploader(
        "Upload dataset (.xlsx, .csv)",
        type=["xlsx", "csv"],
        accept_multiple_files=False,
    )

    if uploaded_file is not None:
        if agent is None:
            st.error("Model agent not ready. Please verify models exist.")
        else:
            try:
                if uploaded_file.name.endswith(".csv"):
                    df = pd.read_csv(uploaded_file)
                else:
                    df = pd.read_excel(uploaded_file)

                if "kannada_text" not in df.columns:
                    st.error("Uploaded file must contain a `kannada_text` column.")
                else:
                    texts = df["kannada_text"].astype(str).tolist()
                    with st.spinner(f"Scoring {len(texts)} Kannada sentences..."):
                        results = agent.predict_texts(texts)

                    if results:
                        results_df = pd.DataFrame(results)

                        def style_sentiment(val: str) -> str:
                            color = get_sentiment_color(str(val))
                            return f"background-color: {color}; color: #ffffff; font-weight: 600;"

                        styler = results_df.style
                        if hasattr(styler, "map"):
                            styled = styler.map(style_sentiment, subset=["sentiment"])
                        else:
                            styled = styler.applymap(style_sentiment, subset=["sentiment"])

                        st.write("Preview of Predictions:")
                        st.dataframe(styled, use_container_width=True)

                        buffer = BytesIO()
                        results_df.to_excel(buffer, index=False)
                        buffer.seek(0)

                        st.download_button(
                            "📥 Download predictions.xlsx",
                            data=buffer,
                            file_name="bhavaspandana_predictions.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            type="primary",
                            use_container_width=True,
                        )
            except Exception as exc:
                st.error(f"Batch prediction error: {exc}")


def render_diagnostics_section() -> None:
    """
    Render confusion matrix heatmap and classification report.
    """
    st.markdown("### 📊 Model Performance & Diagnostics")
    st.caption("Evaluation results on the 488-sample held-out test split.")

    cm_path = os.path.join(ROOT_DIR, "artifacts", "confusion_matrix.png")
    cr_path = os.path.join(ROOT_DIR, "artifacts", "classification_report.csv")

    col_cm, col_cr = st.columns([1.3, 1])

    with col_cm:
        if os.path.exists(cm_path):
            st.image(
                cm_path,
                caption="BhavaSpandana AI — Confusion Matrix Heatmap",
                use_container_width=True,
            )
        else:
            st.info("Confusion matrix image not found. Run `python train.py` to generate it.")

    with col_cr:
        if os.path.exists(cr_path):
            try:
                cr_df = pd.read_csv(cr_path)
                st.markdown("**Per-Class Classification Report:**")
                numeric_cols = ["precision", "recall", "f1-score", "support"]
                style = cr_df.style.background_gradient(
                    cmap="Blues",
                    subset=[c for c in numeric_cols if c in cr_df.columns],
                )
                st.dataframe(style, use_container_width=True)
            except Exception as exc:
                st.error(f"Error loading classification report: {exc}")
        else:
            st.info("Classification report not found. Run `python train.py` to generate it.")


def sidebar(agent: Optional[PredictionAgent]) -> None:
    """
    Render the left glass rail sidebar.
    """
    with st.sidebar:
        st.markdown("### 🌸 BhavaSpandana AI")
        st.markdown("*(ಭಾವಸ್ಪಂದನ AI)*")
        st.markdown(
            """
**Model Type**  
Hybrid **MuRIL BERT** (65%) + **TF-IDF** (35%)  
Logistic Regression Ensemble

**Classes Supported**  
Happy · Sad · Anger · Fear · Disgust · Neutral

**Unicode Support**  
U+0C80–U+0CFF (Kannada Script)
"""
        )
        st.markdown("---")
        st.markdown("#### ⚡ Status")
        if agent is not None:
            st.success("🟢 **Model Active**")
            st.caption(f"Inference Device: `{agent.device}`")
        else:
            st.error("🔴 **Models Not Loaded**")

        if st.button("🔄 Reload Cache", use_container_width=True):
            st.cache_resource.clear()
            st.rerun()

        st.markdown("---")
        st.caption("Crafted for Kannada NLP & Emotion Intelligence.")


def main() -> None:
    """
    Main Glass Iris Chat Dashboard entry point.
    """
    agent = get_prediction_agent()

    sidebar(agent)
    render_shell_header(agent)

    # Segmented Mode Switcher
    selected_mode = st.radio(
        "Navigation Mode",
        options=["💬 Chat & Realtime", "📂 Batch Scoring", "📊 Diagnostics & Benchmarks"],
        horizontal=True,
        label_visibility="collapsed",
    )

    if selected_mode == "💬 Chat & Realtime":
        render_hero_orb()
        render_composer_section(agent)
        render_quick_action_cards()

    elif selected_mode == "📂 Batch Scoring":
        render_batch_section(agent)

    elif selected_mode == "📊 Diagnostics & Benchmarks":
        render_diagnostics_section()


if __name__ == "__main__":
    main()
