import html
import os
import sys
from io import BytesIO
from typing import Dict

import pandas as pd
import streamlit as st


# Ensure project root is on sys.path so we can import agents.
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from agents.prediction_agent import PredictionAgent  # noqa: E402


LOGO_PNG = os.path.join(ROOT_DIR, "assets", "logo.png")

st.set_page_config(
    page_title="BhavaSpandana · Find the feeling in Kannada text",
    layout="wide",
    page_icon=LOGO_PNG if os.path.exists(LOGO_PNG) else "💬",
)


# Page colours come from .streamlit/config.toml (light). This adds the type, the top bar,
# the hero, the prompt card and the result card.
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Anek+Kannada:wght@400;500;600;700&family=DM+Sans:opsz,wght@9..40,400;9..40,500;9..40,600&display=swap');

:root {
    --ink: #1C1917;
    --muted: #5F5B57;
    --line: rgba(0, 0, 0, 0.08);
    --gold: #F5B700;   /* Karnataka flag yellow, used sparingly as the brand accent */
    --ease: cubic-bezier(0.16, 1, 0.3, 1);
    /* Anek Kannada carries matching Kannada + Latin letterforms; DM Sans for reading text */
    --font: "DM Sans", "Anek Kannada", "Noto Sans Kannada", "Nirmala UI", sans-serif;
    --font-display: "Anek Kannada", "DM Sans", "Nirmala UI", sans-serif;
}

/* body font everywhere except Streamlit's icon font */
.stApp, .stApp p, .stApp li, .stApp label, .stApp textarea, .stApp input,
.stApp button, .stApp h1, .stApp h2, .stApp h3,
.stApp span:not([data-testid="stIconMaterial"]) { font-family: var(--font); }
.stApp .brand, .stApp .headline, .stApp .section h2, .stApp .kn,
.stApp .result-title, .stApp textarea { font-family: var(--font-display); }
::selection { background: rgba(245, 183, 0, 0.35); }

[data-testid="stHeader"] { background: transparent; }
/* hero background: soft tint + faint Kannada letters (HERO_LETTERS_SVG), painted on the page
   container so it sits behind everything and scrolls with the content */
[data-testid="stMain"] {
    background:
        url("data:image/svg+xml;utf8,__LETTERS__") top center / 100% auto no-repeat local,
        linear-gradient(180deg, #E9EFE6 0px, #F3F1E9 520px, #FAFAF7 900px) top / 100% 900px no-repeat local,
        #FAFAF7;
}
.block-container { max-width: 1120px; padding-top: 1.5rem; padding-bottom: 2rem; }

/* ---------- top bar ---------- */
.brand {
    display: inline-flex; align-items: center; gap: 10px;
    font-size: 25px; font-weight: 600; color: var(--ink); line-height: 1;
}
.brand svg { flex: none; }
.brand small { font-family: var(--font); font-size: 13px; font-weight: 400; color: var(--muted); padding-top: 4px; }
.nav-links { display: flex; justify-content: center; gap: 36px; margin: 0; }
.nav-links a { font-size: 15px; color: #1A1A1A; text-decoration: none; opacity: 0.9; transition: opacity 0.18s ease; }
.nav-links a:hover { opacity: 0.55; }
.status { display: inline-flex; align-items: center; gap: 8px; font-size: 14px; color: var(--ink);
          background: #fff; border: 1px solid var(--line); border-radius: 10px; padding: 9px 16px; }
.dot { width: 8px; height: 8px; border-radius: 50%; background: #22C55E; box-shadow: 0 0 0 3px rgba(34, 197, 94, 0.18); }
.st-key-load_model, .st-key-check { width: 100% !important; }
.st-key-load_model .stButton, .st-key-check .stButton { display: flex; justify-content: flex-end; }
.st-key-load_model button { border-radius: 10px; padding: 0.55rem 1.3rem; }

/* ---------- hero ---------- */
.hero { text-align: center; padding: clamp(44px, 8vh, 88px) 0 36px; }
.badge {
    display: inline-flex; align-items: center; gap: 12px;
    background: #fff; border: 1px solid var(--line); border-radius: 999px;
    padding: 6px 16px 6px 6px; font-size: 13.5px; color: #4A4A4A;
}
.badge b { background: var(--ink); color: #fff; border-radius: 999px; padding: 3px 12px; font-weight: 500; }
.headline {
    margin: 22px auto 0; font-size: clamp(36px, 5.6vw, 70px); line-height: 1.02;
    font-weight: 600; letter-spacing: -0.02em; color: var(--ink);
}
/* marker-pen highlight on the key word */
.headline em { font-style: normal; text-decoration: underline; text-decoration-color: rgba(245, 183, 0, 0.6);
    text-decoration-thickness: 0.14em; text-underline-offset: 0.1em; text-decoration-skip-ink: none; }
.sub { margin: 18px auto 0; font-size: clamp(15px, 1.3vw, 17px); line-height: 1.65; color: var(--muted); max-width: 60ch; text-wrap: balance; }

/* ---------- prompt card (st.container key="prompt") ---------- */
.st-key-prompt {
    background: #fff;
    border: 1px solid var(--line); border-radius: 20px; padding: 18px 18px 14px;
    box-shadow: 0 18px 44px -26px rgba(0, 0, 0, 0.28);
    transition: border-color 0.18s ease, box-shadow 0.18s ease;
}
.st-key-prompt:focus-within { border-color: rgba(0, 0, 0, 0.3); box-shadow: 0 0 0 4px rgba(245, 183, 0, 0.18), 0 18px 44px -26px rgba(0, 0, 0, 0.28); }
.st-key-prompt [data-baseweb="textarea"],
.st-key-prompt [data-baseweb="base-input"] { background: transparent; border: 0; }
/* Streamlit gives children a fixed pixel width that ignores the card padding */
.st-key-prompt [data-testid="stElementContainer"] { width: 100% !important; }
.st-key-prompt textarea { resize: none; background: transparent; font-size: 19px; line-height: 1.55; padding: 2px 4px; }
.st-key-check button { border-radius: 12px; padding: 0.6rem 1.2rem; white-space: nowrap; }

/* ---------- result panel (right of the input) ---------- */
.panel {
    border-radius: 20px; padding: 20px 22px; min-height: 100%;
    border: 1px solid var(--line); background: rgba(255, 255, 255, 0.75);
}
.panel-title { font-size: 0.85rem; color: var(--muted); margin: 0 0 12px; }
.chips { display: flex; flex-wrap: wrap; gap: 8px; }
.chip {
    display: inline-flex; align-items: baseline; gap: 6px; padding: 6px 12px; border-radius: 999px;
    font-size: 14px; font-weight: 500; color: var(--c);
    background: color-mix(in srgb, var(--c) 10%, white); border: 1px solid color-mix(in srgb, var(--c) 22%, white);
}
.chip .kn { font-size: 13px; opacity: 0.8; font-weight: 500; }
.chip b { color: var(--ink); font-weight: 600; }

.result-card {
    border-radius: 20px; padding: 20px 22px;
    background: color-mix(in srgb, var(--c) 9%, white);
    border: 1px solid color-mix(in srgb, var(--c) 28%, white);
    box-shadow: 0 18px 44px -28px color-mix(in srgb, var(--c) 70%, black);
}
.result-top { display: flex; justify-content: space-between; align-items: baseline; font-size: 0.85rem; color: var(--muted); }
.result-top b { color: var(--c); font-size: 0.95rem; }
.result-title { display: flex; align-items: baseline; gap: 12px; flex-wrap: wrap; margin: 2px 0 10px; }
.result-title .en { font-size: 2.6rem; font-weight: 700; color: var(--c); letter-spacing: -0.02em; line-height: 1.1; }
.result-title .kn { font-size: 1.5rem; font-weight: 500; color: color-mix(in srgb, var(--c) 75%, black); }
.meter { height: 8px; border-radius: 99px; background: color-mix(in srgb, var(--c) 15%, white); overflow: hidden; }
.meter span { display: block; height: 100%; border-radius: 99px; background: var(--c); }
.breakdown { margin: 16px 0 12px; display: grid; gap: 7px; }
.row { display: grid; grid-template-columns: 70px 1fr 42px; align-items: center; gap: 10px; font-size: 13px; color: var(--muted); }
.row .bar { height: 6px; border-radius: 99px; background: rgba(0, 0, 0, 0.06); overflow: hidden; }
.row .bar span { display: block; height: 100%; border-radius: 99px; background: var(--c); }
.row .pct { text-align: right; font-variant-numeric: tabular-nums; }
.result-read { font-size: 0.82rem; color: var(--muted); word-break: break-word; border-top: 1px solid color-mix(in srgb, var(--c) 20%, white); padding-top: 10px; }
.result-read .kn { color: var(--ink); }

/* ---------- lower sections ---------- */
.section { scroll-margin-top: 80px; padding-top: 80px; }
.section .eyebrow { font-size: 13px; font-weight: 600; color: #A07800; letter-spacing: 0.04em; text-transform: uppercase; margin: 0; }
.section h2 { font-size: clamp(28px, 3vw, 38px); font-weight: 600; letter-spacing: -0.02em; margin: 6px 0 8px; padding: 0; line-height: 1.1; }
.section p.lead { color: var(--muted); margin: 0 0 20px; max-width: 64ch; line-height: 1.65; }
.section p.lead code { color: var(--ink); }

.footer {
    margin-top: 80px; padding-top: 20px; border-top: 1px solid var(--line);
    display: flex; justify-content: space-between; flex-wrap: wrap; gap: 8px; font-size: 13px; color: var(--muted);
}

/* gentle one-time entrance for the hero text only */
@keyframes rise { from { opacity: 0; transform: translate3d(0, 14px, 0); } to { opacity: 1; transform: none; } }
.rise { animation: rise 0.9s var(--ease) both; }
.rise-1 { animation-delay: 80ms; }
.rise-2 { animation-delay: 160ms; }
@media (prefers-reduced-motion: reduce) { .rise { animation: none; } }

@media (max-width: 900px) {
    .nav-links { gap: 20px; }
    .brand small { display: none; }
}
@media (max-width: 640px) {
    .nav-links { display: none; }
    .hero { padding-top: 28px; }
    .st-key-prompt textarea { font-size: 17px; } /* stay >= 16px or iOS zooms on focus */
    /* letters get too small and crowded on phones; keep only the tint */
    [data-testid="stMain"] {
        background: linear-gradient(180deg, #E9EFE6 0px, #FAFAF7 700px) top / 100% 700px no-repeat local, #FAFAF7;
    }
}
</style>
"""

# Speech bubble with a pulse line: ಭಾವ (feeling) + ಸ್ಪಂದನ (pulse). Same mark as assets/logo.png.
LOGO_SVG = (
    '<svg width="32" height="32" viewBox="0 0 32 32" aria-hidden="true">'
    '<rect x="2" y="3" width="28" height="22" rx="7" fill="#1C1917"/>'
    '<path d="M7 23v7l7-6z" fill="#1C1917"/>'
    '<path d="M7 14h4l2-5 3 10 2.5-7 1.5 2h5" fill="none" stroke="#F5B700" stroke-width="2.4" '
    'stroke-linecap="round" stroke-linejoin="round"/></svg>'
)

# Large faint letters kept to the left/right edges so they never sit behind the headline.
HERO_LETTERS = [
    ("ಅ", 90, 300, 200), ("ಭಾ", 1180, 260, 170), ("ಸ್ಪಂ", 110, 760, 150), ("ದ", 1200, 720, 210),
    ("ಕ", 20, 540, 130),
]
HERO_LETTERS_SVG = (
    "<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 1440 900'>"
    "<g fill='%231C1917' fill-opacity='0.05' font-family='Noto Sans Kannada, Nirmala UI, Tunga, sans-serif' font-weight='600'>"
    + "".join(f"<text x='{x}' y='{y}' font-size='{size}'>{ch}</text>" for ch, x, y, size in HERO_LETTERS)
    + "</g></svg>"
)

if "prediction_agent" not in st.session_state:
    st.session_state["prediction_agent"] = None


def get_sentiment_color(sentiment: str) -> str:
    """
    Map sentiment labels to brand colors.
    """
    mapping: Dict[str, str] = {
        "happy": "#16A34A",   # green
        "sad": "#2563EB",     # blue
        "angry": "#DC2626",   # red
        "anger": "#DC2626",   # red
        "fear": "#EA580C",    # orange
        "disgust": "#7C3AED", # purple
        "neutral": "#6B7280", # gray
    }
    return mapping.get(sentiment.lower(), "#4B5563")


# Kannada name for each feeling, shown next to the English label.
KANNADA_NAMES: Dict[str, str] = {
    "happy": "ಸಂತೋಷ",
    "sad": "ದುಃಖ",
    "anger": "ಕೋಪ",
    "angry": "ಕೋಪ",
    "fear": "ಭಯ",
    "disgust": "ಅಸಹ್ಯ",
    "neutral": "ತಟಸ್ಥ",
}
FEELINGS = ["happy", "sad", "anger", "fear", "disgust", "neutral"]


def feeling_class(label: str) -> str:
    """
    CSS class that sets --c for a feeling. Streamlit strips custom properties from inline
    style attributes, so colours are passed through these classes instead.
    """
    key = str(label).lower()
    return f"f-{key}" if key in KANNADA_NAMES else "f-other"


FEELING_CSS = "<style>" + "".join(
    f".f-{key} {{ --c: {get_sentiment_color(key)}; }}" for key in KANNADA_NAMES
) + f".f-other {{ --c: {get_sentiment_color('')}; }}</style>"

st.markdown(CUSTOM_CSS.replace("__LETTERS__", HERO_LETTERS_SVG) + FEELING_CSS, unsafe_allow_html=True)


def feeling_chip(label: str, extra: str = "") -> str:
    """
    Small coloured pill: English name, Kannada name and optional extra text (e.g. a count).
    """
    kn = KANNADA_NAMES.get(label.lower(), "")
    return (
        f"<span class='chip {feeling_class(label)}'>"
        f"{html.escape(label.title())}<span class='kn'>{kn}</span>{extra}</span>"
    )


def nav_section() -> None:
    """
    Render the top bar: brand, section links and the model loader.
    """
    col_brand, col_links, col_cta = st.columns([1.2, 1.6, 1.2], vertical_alignment="center")
    col_brand.markdown(
        f"<div class='brand'>{LOGO_SVG}ಭಾವಸ್ಪಂದನ <small>BhavaSpandana</small></div>",
        unsafe_allow_html=True,
    )
    col_links.markdown(
        "<nav class='nav-links'><a href='#try'>Try it</a><a href='#excel'>Excel file</a>"
        "<a href='#how'>How it works</a></nav>",
        unsafe_allow_html=True,
    )

    load_error = None
    with col_cta:
        slot = st.empty()  # holds the button, swapped for a status pill once loaded
        if st.session_state["prediction_agent"] is None:
            if slot.button("Load model", key="load_model", type="primary"):
                try:
                    with st.spinner("Loading, this can take a minute the first time..."):
                        st.session_state["prediction_agent"] = PredictionAgent(model_dir="models")
                except Exception as exc:  # pragma: no cover - UI feedback
                    load_error = exc
        if st.session_state["prediction_agent"] is not None:
            slot.markdown(
                "<div style='text-align:right'><span class='status'><span class='dot'></span>Model ready</span></div>",
                unsafe_allow_html=True,
            )

    if load_error is not None:
        st.error(f"Couldn't load the model: {load_error}")


def hero_section() -> None:
    """
    Render the hero: badge, headline and short description.
    """
    st.markdown(
        "<div class='hero' id='try'>"
        "<div class='badge rise'><b>96.5%</b>accurate on sentences it had never seen</div>"
        "<h1 class='headline rise rise-1'>Find the <em>feeling</em> behind<br>every Kannada sentence.</h1>"
        "<p class='sub rise rise-2'>Write something in Kannada and BhavaSpandana tells you whether "
        "it sounds happy, sad, angry, afraid, disgusted or neutral. One sentence here, or a whole "
        "spreadsheet further down.</p>"
        "</div>",
        unsafe_allow_html=True,
    )


def model_required() -> bool:
    """
    Show a hint and return False if models are not loaded yet.
    """
    if st.session_state["prediction_agent"] is None:
        st.info("Load the model first, using the button at the top right.")
        return False
    return True


def result_card_html(result: Dict[str, object]) -> str:
    """
    Coloured result card: feeling in English + Kannada, confidence meter, score per feeling.
    """
    sentiment = str(result["sentiment"])
    confidence = min(max(float(result["confidence"]), 0.0), 100.0)
    cleaned = str(result["cleaned"])
    preview = cleaned[:120] + ("..." if len(cleaned) > 120 else "")
    probabilities = result.get("probabilities") or {}

    rows = "".join(
        f"<div class='row {feeling_class(label)}'>"
        f"<span>{html.escape(str(label).title())}</span>"
        f"<span class='bar'><span style='width:{min(max(float(p), 0.0), 100.0):.1f}%'></span></span>"
        f"<span class='pct'>{float(p):.0f}%</span></div>"
        for label, p in sorted(probabilities.items(), key=lambda kv: -float(kv[1]))
    )
    breakdown = f"<div class='breakdown'>{rows}</div>" if rows else "<div style='height:14px'></div>"

    return (
        f"<div class='result-card {feeling_class(sentiment)}'>"
        f"<div class='result-top'><span>This sounds</span><b>{confidence:.0f}% sure</b></div>"
        f"<div class='result-title'><span class='en'>{html.escape(sentiment.title())}</span>"
        f"<span class='kn'>{KANNADA_NAMES.get(sentiment.lower(), '')}</span></div>"
        f"<div class='meter'><span style='width:{confidence:.1f}%'></span></div>"
        f"{breakdown}"
        f"<div class='result-read'>What the model read: <span class='kn'>{html.escape(preview)}</span></div>"
        "</div>"
    )


def single_prediction_section() -> None:
    """
    Render the single text prediction interface: prompt card on the left, result on the right.
    """
    col_input, col_result = st.columns([1.35, 1], gap="large")

    with col_input, st.container(key="prompt"):
        input_text = st.text_area(
            "Your sentence",
            label_visibility="collapsed",
            height=170,
            placeholder="ಇವತ್ತು ತುಂಬಾ ಖುಷಿಯಾಗಿದೆ...",
        )
        col_hint, col_btn = st.columns([2, 1], vertical_alignment="center")
        col_hint.caption("Kannada script only. English letters are skipped.")
        with col_btn:
            predict_clicked = st.button(
                "Check", key="check", type="primary", icon=":material/arrow_forward:"
            )

    with col_result:
        if not predict_clicked:
            chips = "".join(feeling_chip(f) for f in FEELINGS)
            st.markdown(
                "<div class='panel'><p class='panel-title'>Your result will show up here. "
                f"It can spot six feelings:</p><div class='chips'>{chips}</div></div>",
                unsafe_allow_html=True,
            )
            return

        if not input_text or not input_text.strip():
            st.warning("Type a sentence first.")
            return
        if not model_required():
            return

        try:
            agent: PredictionAgent = st.session_state["prediction_agent"]
            with st.spinner("Reading..."):
                results = agent.predict_texts([input_text])
            if not results:
                st.warning("The model didn't return anything for this sentence.")
                return
            st.markdown(result_card_html(results[0]), unsafe_allow_html=True)
        except Exception as exc:  # pragma: no cover - UI feedback
            st.error(f"Something went wrong: {exc}")


def section_heading(anchor: str, eyebrow: str, title: str, lead: str) -> None:
    """
    Render a lower-section heading that the top bar links can jump to.
    """
    st.markdown(
        f"<div class='section' id='{anchor}'><p class='eyebrow'>{eyebrow}</p>"
        f"<h2>{title}</h2><p class='lead'>{lead}</p></div>",
        unsafe_allow_html=True,
    )


def batch_prediction_section() -> None:
    """
    Render the batch prediction interface for uploaded Excel files.
    """
    section_heading(
        "excel",
        "Excel file",
        "Check a whole spreadsheet at once",
        "Put your sentences in a column named <code>kannada_text</code>. You'll get the same "
        "rows back with a feeling next to each one.",
    )

    uploaded_file = st.file_uploader("Excel file (.xlsx)", type=["xlsx"], accept_multiple_files=False)

    if uploaded_file is None or not model_required():
        return

    try:
        df = pd.read_excel(uploaded_file)
        if "kannada_text" not in df.columns:
            st.error("This file has no `kannada_text` column. Rename the column with your sentences and try again.")
            return

        texts = df["kannada_text"].astype(str).tolist()
        agent: PredictionAgent = st.session_state["prediction_agent"]
        with st.spinner(f"Reading {len(texts)} sentences..."):
            results = agent.predict_texts(texts)
        if not results:
            st.warning("The model didn't return anything for this file.")
            return

        results_df = pd.DataFrame(results)

        def style_sentiment(val: str) -> str:
            color = get_sentiment_color(str(val))
            return f"color: {color}; font-weight: 600;"

        styler = results_df.style
        if hasattr(styler, "map"):
            styled = styler.map(style_sentiment, subset=["sentiment"])
        else:
            styled = styler.applymap(style_sentiment, subset=["sentiment"])

        top_col, btn_col = st.columns([3, 1], vertical_alignment="center")
        counts = results_df["sentiment"].value_counts()
        count_chips = "".join(feeling_chip(str(label), f"<b>{n}</b>") for label, n in counts.items())
        top_col.markdown(
            f"<p style='margin:0 0 8px'><b>{len(results_df)} sentences done.</b></p>"
            f"<div class='chips'>{count_chips}</div>",
            unsafe_allow_html=True,
        )

        buffer = BytesIO()
        results_df.to_excel(buffer, index=False)
        buffer.seek(0)
        btn_col.download_button(
            "Download results",
            data=buffer,
            file_name="predictions.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            type="primary",
            use_container_width=True,
        )

        st.dataframe(
            styled,
            use_container_width=True,
            column_config={
                "confidence": st.column_config.NumberColumn("confidence", format="%.1f%%"),
            },
        )
    except Exception as exc:  # pragma: no cover - UI feedback
        st.error(f"Couldn't read this file: {exc}")


def metrics_section() -> None:
    """
    Render how the model works, its confusion matrix and classification report.
    """
    section_heading(
        "how",
        "How it works",
        "Two models, one answer",
        "<b>MuRIL</b>, a BERT model trained on Indian languages, and a <b>TF-IDF</b> word model "
        "each read the sentence and guess through logistic regression. Their guesses are combined, "
        "with MuRIL counting for 65% and TF-IDF for 35%. It was trained on Kannada social media "
        "text labelled into six feelings. Below is how it did on sentences it hadn't seen during "
        "training.",
    )

    cm_path = os.path.join(ROOT_DIR, "artifacts", "confusion_matrix.png")
    cr_path = os.path.join(ROOT_DIR, "artifacts", "classification_report.csv")

    cols = st.columns([1.3, 1], gap="large")

    with cols[0]:
        st.markdown("**Where it gets confused**")
        if os.path.exists(cm_path):
            st.image(cm_path, use_container_width=True)
        else:
            st.info("No confusion matrix yet. Train the model first with `python train.py`.")

    with cols[1]:
        st.markdown("**Scores for each feeling**")
        if os.path.exists(cr_path):
            try:
                cr_df = pd.read_csv(cr_path)
                numeric_cols = ["precision", "recall", "f1-score", "support"]
                present = [c for c in numeric_cols if c in cr_df.columns]
                # support is a row count, except on the "accuracy" row where it holds the score
                style = cr_df.style.background_gradient(cmap="Blues", subset=present).format(
                    lambda v: f"{v:.3f}" if v < 1.5 else f"{v:.0f}", subset=present
                )
                st.dataframe(style, use_container_width=True, hide_index=True)
            except Exception as exc:  # pragma: no cover - UI feedback
                st.error(f"Couldn't load the scores: {exc}")
        else:
            st.info("No scores yet. They're created when you train the model.")


def footer_section() -> None:
    """
    Render a small closing line.
    """
    st.markdown(
        "<div class='footer'><span>ಭಾವಸ್ಪಂದನ · BhavaSpandana</span>"
        "<span>Kannada sentiment analysis with MuRIL + TF-IDF</span></div>",
        unsafe_allow_html=True,
    )


def main() -> None:
    """
    Render the full Streamlit page for Kannada Sentiment Analysis.
    """
    nav_section()
    hero_section()
    single_prediction_section()
    batch_prediction_section()
    metrics_section()
    footer_section()


if __name__ == "__main__":
    main()
