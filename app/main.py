# streamlit_app.py
import streamlit as st
import joblib
import os
import re
import numpy as np

# --- Config ---
MODEL_PATH = "model/fake_news_clf.joblib"
VECT_PATH = "model/tfidf_vectorizer.joblib"

# --- Helper functions ---
def clean_text(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = s.lower()
    s = re.sub(r"http\S+", " ", s)
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

@st.cache_resource
def load_artifacts():
    if not (os.path.exists(MODEL_PATH) and os.path.exists(VECT_PATH)):
        raise FileNotFoundError(f"Model or vectorizer missing. Expected: {MODEL_PATH}, {VECT_PATH}")
    model = joblib.load(MODEL_PATH)
    vect = joblib.load(VECT_PATH)
    return model, vect

# --- Page layout ---
st.set_page_config(page_title="Fake News Detector", layout="centered")
st.title("📰 Fake News Detector")
st.write("Paste an article or headline below and press **Predict**. `prediction=1` -> Fake, `0` -> Real")

# load model
with st.spinner("Loading model..."):
    model, vectorizer = load_artifacts()

# input options
st.sidebar.header("Input options")
input_mode = st.sidebar.radio("Input type", ("Paste text", "Upload file", "Use sample"))

text_input = ""
if input_mode == "Paste text":
    text_input = st.text_area("Article text", height=250, placeholder="Paste article text or headline here...")
elif input_mode == "Upload file":
    uploaded = st.file_uploader("Upload a .txt file", type=["txt"])
    if uploaded is not None:
        raw = uploaded.read().decode("utf-8", errors="ignore")
        text_input = st.text_area("File content", value=raw, height=250)
else:
    # sample
    sample = st.sidebar.selectbox("Sample text", [
        "Breaking: scientists discover incredible cure that changes medicine forever",
        "President addresses the nation on the economic recovery plan",
        "Celebrity endorses miracle weight loss pill in shocking announcement"
    ])
    text_input = st.text_area("Sample text", value=sample, height=250)

# predict button
if st.button("Predict"):
    if not text_input.strip():
        st.warning("Please provide some text before predicting.")
    else:
        with st.spinner("Cleaning text and running model..."):
            cleaned = clean_text(text_input)
            X = vectorizer.transform([cleaned])
            pred = int(model.predict(X)[0])
            proba = float(model.predict_proba(X).max())

        label = "Fake" if pred == 1 else "Real"
        st.subheader(f"Prediction: {label} ( {pred} )")
        st.write(f"Confidence: **{proba:.3f}**")

        # show probability bar
        st.progress(int(proba * 100))

        # show raw probabilities for both classes if model supports predict_proba
        try:
            probs = model.predict_proba(X)[0]
            st.write("Probabilities:", {"fake": float(probs[1]), "real": float(probs[0])})
        except Exception:
            st.info("Model does not support `predict_proba` display.")

# footer
st.markdown("---")
st.markdown("Built with ❤️ using Streamlit — run locally with `streamlit run streamlit_app.py`")