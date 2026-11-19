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
    page_title="ಭಾವಸ್ಪಂದನ (BhavaSpandana AI) - Kannada Emotion AI",
    layout="wide",
    page_icon="🌸",
)

# Kannada emotion metadata with Kannada terms, emojis, colors, and descriptions
EMOTION_META = {
    "happy": {
        "kannada": "ಸಂತೋಷ",
        "emoji": "😊",
        "color": "#16A34A",
        "gradient": "linear-gradient(135deg, #15803D 0%, #166534 100%)",
        "desc": "ಸಂತೋಷ ಹಾಗೂ ಆನಂದದ ಭಾವನೆ (Joy & Contentment)",
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

CUSTOM_CSS = """
<style>
body {
    background: linear-gradient(135deg, #0b091a 0%, #1a1638 50%, #120e29 100%);
    font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    color: #ffffff;
}

.main .block-container {
    padding-top: 1.5rem;
    padding-bottom: 2.5rem;
    max-width: 1200px;
}

.brand-hero {
    text-align: center;
    padding: 1.5rem 1rem 1rem 1rem;
    margin-bottom: 1.5rem;
    background: radial-gradient(circle at 50% 20%, rgba(124, 58, 237, 0.18) 0%, rgba(15, 12, 41, 0) 70%);
    border-radius: 16px;
    border: 1px solid rgba(139, 92, 246, 0.2);
}

.brand-title {
    font-size: 2.4rem;
    font-weight: 800;
    letter-spacing: -0.02em;
    background: linear-gradient(135deg, #F9FAFB 0%, #C4B5FD 50%, #A78BFA 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.3rem;
}

.brand-subtitle {
    font-size: 1.05rem;
    color: #D1D5DB;
    margin-bottom: 0.6rem;
}

.brand-tags {
    display: flex;
    justify-content: center;
    gap: 0.5rem;
    flex-wrap: wrap;
    margin-top: 0.5rem;
}

.brand-tag {
    background: rgba(139, 92, 246, 0.15);
    border: 1px solid rgba(167, 139, 250, 0.35);
    color: #E9D5FF;
    border-radius: 9999px;
    padding: 0.25rem 0.8rem;
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 0.04em;
}

.card {
    background: rgba(255, 255, 255, 0.04);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 14px;
    padding: 1.4rem 1.6rem;
    backdrop-filter: blur(12px);
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.35);
    margin-bottom: 1rem;
}

.metric-card {
    text-align: center;
    padding: 1rem;
    background: rgba(255, 255, 255, 0.03);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 12px;
}

.metric-label {
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #9CA3AF;
    margin-bottom: 0.2rem;
}

.metric-value {
    font-size: 1.6rem;
    font-weight: 700;
    color: #F9FAFB;
}

.metric-subtext {
    font-size: 0.8rem;
    color: #A78BFA;
}

.sentiment-card {
    border-radius: 14px;
    padding: 1.5rem;
    margin-top: 0.5rem;
    color: #ffffff;
    box-shadow: 0 16px 40px rgba(0, 0, 0, 0.45);
    border: 1px solid rgba(255, 255, 255, 0.18);
}

.sentiment-kannada {
    font-size: 2.2rem;
    font-weight: 800;
    line-height: 1.2;
    margin-bottom: 0.2rem;
}

.sentiment-english {
    font-size: 1.2rem;
    font-weight: 600;
    opacity: 0.95;
    margin-bottom: 0.5rem;
}

.sentiment-desc {
    font-size: 0.88rem;
    opacity: 0.85;
    margin-bottom: 0.8rem;
}

.section-title {
    font-size: 1.15rem;
    font-weight: 700;
    letter-spacing: 0.05em;
    color: #E0E7FF;
    margin-bottom: 0.2rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

.section-subtitle {
    font-size: 0.9rem;
    color: #9CA3AF;
    margin-bottom: 1rem;
}

.sample-chip {
    display: inline-block;
    background: rgba(139, 92, 246, 0.12);
    border: 1px solid rgba(139, 92, 246, 0.3);
    border-radius: 8px;
    padding: 0.4rem 0.8rem;
    margin: 0.2rem;
    font-size: 0.85rem;
    color: #DDD6FE;
    cursor: pointer;
}
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


@st.cache_resource(show_spinner="ಭಾವಸ್ಪಂದನ ಮಾದರಿ ಲೋಡ್ ಆಗುತ್ತಿದೆ / Loading BhavaSpandana AI models...")
def get_prediction_agent() -> Optional[PredictionAgent]:
    """
    Load and cache the PredictionAgent instance across user sessions.
    """
    model_dir = os.path.join(ROOT_DIR, "models")
    if not os.path.exists(model_dir):
        return None
    try:
        return PredictionAgent(model_dir=model_dir)
    except Exception as exc:
        print(f"[Dashboard] Error initializing PredictionAgent: {exc}")
        return None


def get_sentiment_color(sentiment: str) -> str:
    """
    Map sentiment label to display color.
    """
    meta = EMOTION_META.get(sentiment.lower())
    return meta["color"] if meta else "#4B5563"


def sidebar(agent: Optional[PredictionAgent]) -> None:
    """
    Render informative sidebar.
    """
    with st.sidebar:
        st.markdown("### 🌸 ಭಾವಸ್ಪಂದನ AI")
        st.markdown(
            "**BhavaSpandana AI** is a state-of-the-art hybrid emotion classification system "
            "engineered specifically for the **Kannada** language."
        )

        st.markdown("---")
        st.markdown("#### 🧠 Model Architecture")
        st.markdown(
            """
- **Transformer**: Google MuRIL (Multilingual BERT for Indian Languages)
- **Feature Layer**: MuRIL [CLS] 768-d embeddings (65% weight)
- **N-Gram Layer**: Sublinear char_wb TF-IDF (35% weight)
- **Ensemble**: Soft-voting Logistic Regression
- **Unicode Support**: U+0C80–U+0CFF (Kannada script)
"""
        )

        st.markdown("---")
        st.markdown("#### ⚡ System Status")
        if agent is not None:
            st.success("🟢 **Model Active & Ready**")
            st.caption(f"Inference Device: `{agent.device}`")
        else:
            st.error("🔴 **Models Not Found**")
            st.caption("Please run `python train.py` to train and save models.")

        if st.button("🔄 Reload Model Cache", use_container_width=True):
            st.cache_resource.clear()
            st.rerun()

        st.markdown("---")
        st.caption("Developed with ❤️ for Kannada NLP & Emotion Intelligence.")


def header_section() -> None:
    """
    Render main header and project metrics.
    """
    st.markdown(
        """
        <div class="brand-hero">
            <div class="brand-title">ಭಾವಸ್ಪಂದನ (BhavaSpandana AI)</div>
            <div class="brand-subtitle">ಕನ್ನಡ ಭಾವನೆ ವಿಶ್ಲೇಷಣೆ — Kannada Emotion & Sentiment Intelligence Engine</div>
            <div class="brand-tags">
                <span class="brand-tag">🤖 Google MuRIL BERT</span>
                <span class="brand-tag">⚡ Char TF-IDF Ensemble</span>
                <span class="brand-tag">🎯 96.52% Test Accuracy</span>
                <span class="brand-tag">✨ 6 Kannada Emotions</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(
            """
            <div class="metric-card">
              <div class="metric-label">Model Accuracy</div>
              <div class="metric-value">96.52%</div>
              <div class="metric-subtext">On held-out test split</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            """
            <div class="metric-card">
              <div class="metric-label">Macro F1 Score</div>
              <div class="metric-value">0.9649</div>
              <div class="metric-subtext">Balanced multi-class</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            """
            <div class="metric-card">
              <div class="metric-label">Hybrid Ensemble</div>
              <div class="metric-value">MuRIL + TF-IDF</div>
              <div class="metric-subtext">65% Transformer + 35% TF-IDF</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col4:
        st.markdown(
            """
            <div class="metric-card">
              <div class="metric-label">Dataset Samples</div>
              <div class="metric-value">2,436</div>
              <div class="metric-subtext">6 Balanced Emotion Classes</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def single_prediction_section(agent: Optional[PredictionAgent]) -> None:
    """
    Real-time Kannada sentence prediction interface with sample shortcuts and probability breakdown.
    """
    st.markdown(
        "<div class='section-title'>💬 ನೈಜ ಸಮಯದ ವಿಶ್ಲೇಷಣೆ / Real-time Sentence Inference</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<div class='section-subtitle'>Type or paste a Kannada sentence below, or click any sample sentence to test the model.</div>",
        unsafe_allow_html=True,
    )

    # Pre-defined sample sentences for quick user testing
    samples = [
        ("😊 ಸಂತೋಷ", "ನಾನು ಇಂದು ತುಂಬಾ ಸಂತೋಷವಾಗಿದ್ದೇನೆ"),
        ("😢 ದುಃಖ", "ನನಗೆ ತುಂಬಾ ದುಃಖವಾಗಿದೆ ಮತ್ತು ಬೇಸರವಾಗಿದೆ"),
        ("😠 ಕೋಪ", "ಇದು ನನಗೆ ತುಂಬಾ ಕೋಪ ತರಿಸುತ್ತದೆ"),
        ("😨 ಭಯ", "ಅದು ತುಂಬಾ ಭಯಂಕರವಾಗಿದೆ ನನಗೆ ಭಯವಾಗ್ತಿದೆ"),
        ("🤢 ಅಸಹ್ಯ", "ಇದು ತುಂಬಾ ಅಸಹ್ಯವಾಗಿದೆ"),
        ("😐 ತಟಸ್ಥ", "ಇಂದು ಹವಾಮಾನ ಸಾಮಾನ್ಯವಾಗಿದೆ ಮತ್ತು ಶಾಂತವಾಗಿದೆ"),
    ]

    st.write("**Quick Test Samples (ಕ್ಲಿಕ್ ಮಾಡಿ):**")
    cols = st.columns(6)
    for idx, (label, sample_text) in enumerate(samples):
        with cols[idx]:
            if st.button(label, key=f"sample_{idx}", use_container_width=True):
                st.session_state["input_text"] = sample_text

    default_text = st.session_state.get("input_text", "ನಾನು ಇಂದು ತುಂಬಾ ಸಂತೋಷವಾಗಿದ್ದೇನೆ")

    col_input, col_result = st.columns([1.3, 1])

    with col_input:
        input_text = st.text_area(
            "ಕನ್ನಡ ವಾಕ್ಯವನ್ನು ನಮೂದಿಸಿ (Enter Kannada text):",
            value=default_text,
            height=140,
            placeholder="ಉದಾಹರಣೆ: ಇಂದು ಎಲ್ಲವೂ ಬಹಳ ಚೆನ್ನಾಗಿ ನಡೆದಿದೆ...",
        )
        predict_clicked = st.button("🔍 ಭಾವನೆ ವಿಶ್ಲೇಷಿಸಿ / Predict Sentiment", type="primary", use_container_width=True)

    with col_result:
        if predict_clicked or ("input_text" in st.session_state and input_text):
            if not input_text or not input_text.strip():
                st.warning("ದಯವಿಟ್ಟು ಕನ್ನಡ ವಾಕ್ಯವನ್ನು ನಮೂದಿಸಿ (Please enter some Kannada text).")
            elif agent is None:
                st.error("ಭಾವಸ್ಪಂದನ ಮಾದರಿ ಲಭ್ಯವಿಲ್ಲ. ದಯವಿಟ್ಟು `python train.py` ಚಲಾಯಿಸಿ (Model not found).")
            else:
                try:
                    with st.spinner("ವಿಶ್ಲೇಷಿಸಲಾಗುತ್ತಿದೆ / Analyzing..."):
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
                                "emoji": "🎯",
                                "color": "#4B5563",
                                "gradient": "linear-gradient(135deg, #4B5563 0%, #1F2937 100%)",
                                "desc": "ಭಾವನೆ",
                            },
                        )

                        st.markdown(
                            f"""
                            <div class="sentiment-card" style="background: {meta['gradient']};">
                              <div style="font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.08em; opacity: 0.85;">
                                ನಿರೀಕ್ಷಿತ ಭಾವನೆ / Predicted Emotion
                              </div>
                              <div class="sentiment-kannada">{meta['emoji']} {meta['kannada']}</div>
                              <div class="sentiment-english">{sentiment.upper()} ({confidence:.1f}% Confidence)</div>
                              <div class="sentiment-desc">{meta['desc']}</div>
                              <div style="font-size: 0.78rem; opacity: 0.8; margin-top: 0.4rem;">
                                <strong>ಶುದ್ಧೀಕರಿಸಿದ ಪಠ್ಯ (Cleaned):</strong> {cleaned if cleaned else "—"}
                              </div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                        # Display probabilities across all 6 classes
                        if probs:
                            st.write("**ಭಾವನೆಗಳ ಸಂಭವನೀಯತೆ ಹಂಚಿಕೆ (Emotion Probabilities):**")
                            prob_data = []
                            for k, v in probs.items():
                                m = EMOTION_META.get(k.lower(), {"kannada": k, "emoji": ""})
                                prob_data.append({"Emotion": f"{m['emoji']} {m['kannada']} ({k.title()})", "Probability (%)": round(v, 2)})
                            prob_df = pd.DataFrame(prob_data).sort_values(by="Probability (%)", ascending=False)
                            st.dataframe(
                                prob_df.style.bar(subset=["Probability (%)"], color=meta["color"], vmin=0, vmax=100),
                                use_container_width=True,
                                hide_index=True,
                            )
                except Exception as exc:
                    st.error(f"Prediction failed: {exc}")


def batch_prediction_section(agent: Optional[PredictionAgent]) -> None:
    """
    Batch scoring section for Excel / CSV uploads.
    """
    st.markdown(
        "<div class='section-title'>📂 ಬ್ಯಾಚ್ ವಿಶ್ಲೇಷಣೆ / Batch Prediction</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<div class='section-subtitle'>Upload an Excel (`.xlsx`) or CSV (`.csv`) file containing a `kannada_text` column to score in bulk.</div>",
        unsafe_allow_html=True,
    )

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    col_up, col_tpl = st.columns([3, 1])

    with col_up:
        uploaded_file = st.file_uploader(
            "Upload file with a 'kannada_text' column",
            type=["xlsx", "csv"],
            accept_multiple_files=False,
        )

    with col_tpl:
        st.write("**ಮಾದರಿ ಫೈಲ್ / Sample Template:**")
        sample_df = pd.DataFrame(
            {
                "kannada_text": [
                    "ನಾನು ಇಂದು ತುಂಬಾ ಸಂತೋಷವಾಗಿದ್ದೇನೆ",
                    "ನನಗೆ ತುಂಬಾ ದುಃಖವಾಗಿದೆ",
                    "ಇದು ನನಗೆ ತುಂಬಾ ಕೋಪ ತರಿಸುತ್ತದೆ",
                    "ಅದು ತುಂಬಾ ಭಯಂಕರವಾಗಿದೆ",
                    "ಇದು ತುಂಬಾ ಅಸಹ್ಯವಾಗಿದೆ",
                    "ಇಂದು ಹವಾಮಾನ ಸಾಮಾನ್ಯವಾಗಿದೆ",
                ]
            }
        )
        tpl_buffer = BytesIO()
        sample_df.to_excel(tpl_buffer, index=False)
        tpl_buffer.seek(0)
        st.download_button(
            "📥 Download Sample Excel",
            data=tpl_buffer,
            file_name="kannada_sample_template.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )

    if uploaded_file is not None:
        if agent is None:
            st.error("ಭಾವಸ್ಪಂದನ ಮಾದರಿ ಲಭ್ಯವಿಲ್ಲ. ದಯವಿಟ್ಟು ಮಾದರಿಗಳನ್ನು ತರಬೇತುಗೊಳಿಸಿ (Model not available).")
        else:
            try:
                if uploaded_file.name.endswith(".csv"):
                    df = pd.read_csv(uploaded_file)
                else:
                    df = pd.read_excel(uploaded_file)

                if "kannada_text" not in df.columns:
                    st.error("The uploaded file must contain a `kannada_text` column.")
                else:
                    raw_texts = df["kannada_text"].astype(str).tolist()
                    with st.spinner(f"Scoring {len(raw_texts)} Kannada sentences..."):
                        results = agent.predict_texts(raw_texts)

                    if results:
                        results_df = pd.DataFrame(results)

                        # Add Kannada Emotion translation column
                        results_df["kannada_emotion"] = results_df["sentiment"].apply(
                            lambda s: EMOTION_META.get(str(s).lower(), {}).get("kannada", str(s))
                        )

                        # Reorder columns for clean display
                        cols_order = ["text", "kannada_emotion", "sentiment", "confidence", "cleaned"]
                        cols_present = [c for c in cols_order if c in results_df.columns]
                        display_df = results_df[cols_present]

                        st.success(f"✅ Successfully scored {len(results_df)} sentences!")

                        # Metrics breakdown
                        sentiment_counts = results_df["sentiment"].value_counts()
                        st.write("**ಭಾವನೆಗಳ ಹಂಚಿಕೆ (Emotion Distribution):**")
                        cols_dist = st.columns(len(sentiment_counts))
                        for i, (sent, count) in enumerate(sentiment_counts.items()):
                            m = EMOTION_META.get(sent.lower(), {"kannada": sent, "emoji": ""})
                            with cols_dist[i]:
                                st.metric(f"{m['emoji']} {m['kannada']}", f"{count} ({count/len(results_df):.1%})")

                        st.write("Preview of predictions:")
                        st.dataframe(display_df, use_container_width=True)

                        out_buffer = BytesIO()
                        results_df.to_excel(out_buffer, index=False)
                        out_buffer.seek(0)

                        st.download_button(
                            "📥 Download Predictions (Excel)",
                            data=out_buffer,
                            file_name="bhavaspandana_predictions.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            type="primary",
                            use_container_width=True,
                        )
            except Exception as exc:
                st.error(f"Batch prediction failed: {exc}")

    st.markdown("</div>", unsafe_allow_html=True)


def diagnostics_section() -> None:
    """
    Model evaluation artifacts, confusion matrix heatmap, and classification report.
    """
    st.markdown(
        "<div class='section-title'>📊 ಮಾದರಿ ಕಾರ್ಯಕ್ಷಮತೆ / Model Evaluation & Diagnostics</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<div class='section-subtitle'>Quantitative performance metrics, per-class F1-scores, and confusion matrix on held-out test data.</div>",
        unsafe_allow_html=True,
    )

    cm_path = os.path.join(ROOT_DIR, "artifacts", "confusion_matrix.png")
    cr_path = os.path.join(ROOT_DIR, "artifacts", "classification_report.csv")

    col_cm, col_cr = st.columns([1.2, 1])

    with col_cm:
        if os.path.exists(cm_path):
            st.image(
                cm_path,
                caption="ಭಾವಸ್ಪಂದನ (BhavaSpandana AI) - Confusion Matrix Heatmap",
                use_container_width=True,
            )
        else:
            st.info("Confusion matrix image not found. Run `python train.py` to generate it.")

    with col_cr:
        if os.path.exists(cr_path):
            try:
                cr_df = pd.read_csv(cr_path)
                st.markdown("**Per-Class Precision, Recall & F1-Score:**")
                numeric_cols = ["precision", "recall", "f1-score", "support"]
                styled = cr_df.style.background_gradient(
                    cmap="Blues",
                    subset=[c for c in numeric_cols if c in cr_df.columns],
                )
                st.dataframe(styled, use_container_width=True)
            except Exception as exc:
                st.error(f"Could not load classification report: {exc}")
        else:
            st.info("Classification report not found. Run `python train.py` to generate it.")


def main() -> None:
    """
    Main application entry point.
    """
    agent = get_prediction_agent()

    sidebar(agent)
    header_section()
    st.markdown("---")

    single_prediction_section(agent)
    st.markdown("---")

    batch_prediction_section(agent)
    st.markdown("---")

    diagnostics_section()


if __name__ == "__main__":
    main()
