from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from spam_detector.prediction import SpamDetectorService
from spam_detector.training import build_default_dataset, load_dataset, train_and_evaluate

st.set_page_config(
    page_title="Inbox Shield",
    page_icon="✉️",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
:root {
    --bg: #0b1020;
    --panel: rgba(13, 20, 40, 0.82);
    --panel-strong: rgba(18, 27, 54, 0.96);
    --text: #eef2ff;
    --muted: #b2b9d6;
    --accent: #7dd3fc;
    --accent-2: #f59e0b;
    --border: rgba(255, 255, 255, 0.10);
}

html, body, [class*="css"] {
    font-family: "Trebuchet MS", "Segoe UI", sans-serif;
}

.stApp {
    background:
        radial-gradient(circle at top left, rgba(125, 211, 252, 0.18), transparent 30%),
        radial-gradient(circle at top right, rgba(245, 158, 11, 0.15), transparent 28%),
        linear-gradient(180deg, #08101d 0%, #0b1327 55%, #09101c 100%);
    color: var(--text);
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}

.hero {
    background: linear-gradient(135deg, rgba(125, 211, 252, 0.12), rgba(245, 158, 11, 0.10));
    border: 1px solid var(--border);
    border-radius: 28px;
    padding: 2rem;
    box-shadow: 0 24px 80px rgba(0, 0, 0, 0.30);
    backdrop-filter: blur(12px);
}

.hero h1 {
    margin-bottom: 0.25rem;
    font-size: 3rem;
    letter-spacing: -0.04em;
}

.hero p {
    color: var(--muted);
    font-size: 1.04rem;
    max-width: 62ch;
}

.metric-card, .panel, .stTextArea, .stSelectbox, .stFileUploader {
    border-radius: 20px !important;
}

.panel {
    background: var(--panel);
    border: 1px solid var(--border);
    padding: 1.25rem;
    box-shadow: 0 16px 40px rgba(0, 0, 0, 0.18);
}

.small-label {
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 0.14em;
    font-size: 0.72rem;
    margin-bottom: 0.35rem;
}

.prediction-flag {
    display: inline-block;
    padding: 0.45rem 0.9rem;
    border-radius: 999px;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    border: 1px solid var(--border);
}

.prediction-spam {
    background: rgba(245, 158, 11, 0.16);
    color: #ffd180;
}

.prediction-ham {
    background: rgba(125, 211, 252, 0.14);
    color: #8be7ff;
}
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

st.markdown(
    """
    <div class="hero">
        <h1>Inbox Shield</h1>
        <p>
            A polished email spam detector that trains on a curated sample dataset, scores new messages,
            and lets you retrain on your own CSV whenever you need a fresher model.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

service = SpamDetectorService()

with st.sidebar:
    st.header("Control Center")
    mode = st.radio("Model source", ["Use saved model", "Retrain from sample data", "Retrain from uploaded CSV"], index=0)
    artifact_path = st.text_input("Model artifact path", value=str(service.artifact_path))
    st.caption("The app saves the trained pipeline here so predictions stay fast on later runs.")

    uploaded_file = None
    if mode == "Retrain from uploaded CSV":
        uploaded_file = st.file_uploader("Upload CSV with label,text columns", type=["csv"])

    if st.button("Train / Refresh Model", use_container_width=True):
        try:
            if mode == "Retrain from uploaded CSV":
                if uploaded_file is None:
                    st.error("Upload a CSV before retraining.")
                else:
                    uploaded_df = pd.read_csv(uploaded_file)
                    training_result = train_and_evaluate(uploaded_df, artifact_path=artifact_path)
                    st.session_state["model_metrics"] = training_result.metrics
                    st.session_state["evaluation_rows"] = training_result.evaluation_rows
                    st.session_state["service"] = SpamDetectorService(artifact_path)
                    st.success("Model retrained on your uploaded data.")
            elif mode == "Retrain from sample data":
                training_result = train_and_evaluate(build_default_dataset(), artifact_path=artifact_path)
                st.session_state["model_metrics"] = training_result.metrics
                st.session_state["evaluation_rows"] = training_result.evaluation_rows
                st.session_state["service"] = SpamDetectorService(artifact_path)
                st.success("Model retrained on the sample dataset.")
            else:
                st.session_state["service"] = SpamDetectorService(artifact_path)
                st.info("Using the saved model.")
        except Exception as error:
            st.error(f"Training failed: {error}")

current_artifact_path = Path(artifact_path)
active_service = st.session_state.get("service")
if active_service is None or active_service.artifact_path != current_artifact_path:
    active_service = SpamDetectorService(current_artifact_path)
    st.session_state["service"] = active_service

left, right = st.columns([1.15, 0.85], gap="large")

with left:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="small-label">Try a message</div>', unsafe_allow_html=True)
    sample_messages = {
        "Friendly check-in": "Hey, are we still on for lunch today?",
        "Suspicious offer": "Congratulations, you have won a free prize. Click now!",
        "Security alert": "Urgent! Your account is locked. Verify your password immediately.",
    }
    example_choice = st.selectbox("Quick examples", list(sample_messages.keys()))
    email_text = st.text_area(
        "Paste an email or SMS",
        value=sample_messages[example_choice],
        height=220,
        placeholder="Type a message here...",
    )

    if st.button("Analyze Message", type="primary", use_container_width=True):
        result = active_service.predict_text(email_text)
        badge_class = "prediction-spam" if result.label == "spam" else "prediction-ham"
        st.markdown(
            f'<div class="prediction-flag {badge_class}">{result.label.upper()}</div>',
            unsafe_allow_html=True,
        )
        c1, c2, c3 = st.columns(3)
        c1.metric("Spam probability", f"{result.spam_probability:.1%}")
        c2.metric("Ham probability", f"{result.ham_probability:.1%}")
        c3.metric("Decision", "Blocked" if result.label == "spam" else "Allowed")
    st.markdown('</div>', unsafe_allow_html=True)

with right:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="small-label">Model snapshot</div>', unsafe_allow_html=True)
    metrics = st.session_state.get("model_metrics")
    if metrics is None:
        training_result = train_and_evaluate(build_default_dataset(), artifact_path=artifact_path)
        metrics = training_result.metrics
        st.session_state["model_metrics"] = metrics
        st.session_state["evaluation_rows"] = training_result.evaluation_rows
        st.session_state["service"] = SpamDetectorService(artifact_path)

    metric_cols = st.columns(4)
    metric_cols[0].metric("Accuracy", f"{metrics['accuracy']:.1%}")
    metric_cols[1].metric("Precision", f"{metrics['precision']:.1%}")
    metric_cols[2].metric("Recall", f"{metrics['recall']:.1%}")
    metric_cols[3].metric("F1", f"{metrics['f1']:.1%}")

    st.caption(f"Training rows: {metrics['train_size']} | Test rows: {metrics['test_size']}")
    st.text(metrics["classification_report"])

    confusion = pd.DataFrame(metrics["confusion_matrix"], index=["actual_ham", "actual_spam"], columns=["pred_ham", "pred_spam"])
    st.dataframe(confusion, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("### Recent evaluation samples")
rows = st.session_state.get("evaluation_rows", [])
if rows:
    st.dataframe(pd.DataFrame(rows).head(10), use_container_width=True)
else:
    st.info("Train the model to populate evaluation samples.")
