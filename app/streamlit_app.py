import streamlit as st
import joblib
import os
import re

MODEL_PATH = "model/fake_news_clf.joblib"
VECT_PATH = "model/tfidf_vectorizer.joblib"

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
        raise FileNotFoundError("Model files not found in /model folder")
    model = joblib.load(MODEL_PATH)
    vect = joblib.load(VECT_PATH)
    return model, vect

st.set_page_config(page_title="Fake News Detector")

st.title("📰 Fake News Detector (TF-IDF + Logistic)")
st.write("Enter text and click **Predict** to classify it as Fake or Real.")

model, vectorizer = load_artifacts()

text_input = st.text_area("Enter article text below:", height=200)

if st.button("Predict"):
    if not text_input.strip():
        st.warning("Please enter some text first.")
    else:
        cleaned = clean_text(text_input)
        vec = vectorizer.transform([cleaned])
        pred = int(model.predict(vec)[0])
        prob = float(model.predict_proba(vec).max())

        label = "Fake" if pred == 1 else "Real"

        st.subheader(f"Prediction: **{label}**")
        st.write(f"Confidence: **{prob:.3f}**")
        st.progress(int(prob * 100))