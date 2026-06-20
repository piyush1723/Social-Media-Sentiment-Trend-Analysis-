import streamlit as st
import pandas as pd
import numpy as np
import joblib
import re
from functools import lru_cache
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Sentiment Intelligence Platform",
    page_icon="📊",
    layout="wide"
)

# ---------------- LOAD MODEL (SAFE) ----------------
@st.cache_resource
def load_model():
    try:
        model = joblib.load("models/sentiment_model.pkl")
        vectorizer = joblib.load("models/tfidf_vectorizer.pkl")
        return model, vectorizer
    except Exception as e:
        st.error(f"Model loading failed: {e}")
        return None, None

model, tfidf = load_model()

# ---------------- NLP INIT (CACHE) ----------------
@st.cache_resource
def load_nlp():
    import nltk
    nltk.download("stopwords", quiet=True)
    stop_words = set(stopwords.words("english"))
    stemmer = PorterStemmer()
    return stop_words, stemmer

stop_words, stemmer = load_nlp()

# ---------------- PREPROCESSING (OPTIMIZED) ----------------
@lru_cache(maxsize=5000)
def preprocess(text: str) -> str:
    text = text.lower()
    text = re.sub(r'@\w+|#', '', text)
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-zA-Z ]', ' ', text)
    words = text.split()

    words = [w for w in words if w not in stop_words]
    words = [stemmer.stem(w) for w in words]

    return " ".join(words).strip()

# ---------------- SESSION STATE ----------------
if "history" not in st.session_state:
    st.session_state.history = []

# ---------------- SIDEBAR ----------------
st.sidebar.title("📌 Dashboard")

page = st.sidebar.radio(
    "Navigation",
    ["🏠 Analyzer", "📈 Analytics", "ℹ️ About"]
)

st.sidebar.divider()

st.sidebar.markdown("### Model Info")
st.sidebar.success("Model Loaded" if model else "Model Not Loaded")

st.sidebar.write("Algorithm: LinearSVC")
st.sidebar.write("Features: TF-IDF")
st.sidebar.write("Classes: 4")

# ---------------- MAIN PAGE ----------------
if page == "🏠 Analyzer":

    st.markdown("""
    <div style="
        padding:25px;
        border-radius:18px;
        background:linear-gradient(135deg,#0f172a,#1e293b);
        color:white;
        text-align:center;">
        <h1>📊 Sentiment Intelligence Platform</h1>
        <p>AI Powered NLP Classification System</p>
    </div>
    """, unsafe_allow_html=True)

    st.write("")

    examples = {
        "Positive": "I absolutely love this product!",
        "Negative": "This is the worst experience ever",
        "Neutral": "The update was released today",
        "Irrelevant": "My dog is sleeping on the sofa"
    }

    choice = st.selectbox(
        "Choose Example or Write Custom",
        ["Custom"] + list(examples.keys())
    )

    text_input = st.text_area(
        "Enter Text",
        value=examples.get(choice, ""),
        height=140
    )

    if st.button("🚀 Predict Sentiment", use_container_width=True):

        if not text_input.strip():
            st.warning("Please enter valid text")
            st.stop()

        if model is None or tfidf is None:
            st.error("Model not loaded properly")
            st.stop()

        with st.spinner("Analyzing sentiment..."):

            processed = preprocess(text_input)
            vector = tfidf.transform([processed])

            pred = model.predict(vector)[0]

            # safer confidence
            try:
                scores = model.decision_function(vector)
                confidence = float(np.max(scores))
                confidence = 1 / (1 + np.exp(-confidence))  # sigmoid scaling
                confidence = round(confidence * 100, 2)
            except:
                confidence = 0

        # history
        st.session_state.history.append({
            "Text": text_input[:60],
            "Prediction": pred,
            "Confidence": confidence
        })

        # output UI
        st.subheader("Result")

        color_map = {
            "Positive": "success",
            "Negative": "error",
            "Neutral": "info",
            "Irrelevant": "warning"
        }

        getattr(st, color_map.get(pred, "info"))(f"{pred} Sentiment")

        st.metric("Confidence", f"{confidence}%")
        st.progress(confidence / 100)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### Original")
            st.write(text_input)

        with col2:
            st.markdown("### Processed")
            st.write(processed)

    # history export
    if st.session_state.history:
        st.download_button(
            "📥 Download History",
            pd.DataFrame(st.session_state.history).to_csv(index=False),
            "history.csv",
            "text/csv"
        )

# ---------------- ANALYTICS ----------------
elif page == "📈 Analytics":

    st.title("📈 Analytics Dashboard")

    df = pd.DataFrame(st.session_state.history)

    if not df.empty:

        st.bar_chart(df["Prediction"].value_counts())

        st.dataframe(df)

    else:
        st.info("No data available yet")

# ---------------- ABOUT ----------------
else:
    st.title("ℹ️ About")

    st.markdown("""
    ### Sentiment Analysis System

    - NLP preprocessing
    - TF-IDF vectorization
    - LinearSVC classifier
    - Streamlit deployment

    ### Improvements in this version
    - Faster preprocessing (cached)
    - Safer model loading
    - Better confidence scoring
    - Export feature
    - Production-ready structure
    """)

# ---------------- FOOTER ----------------
st.markdown("---")
st.markdown("<center>Built with ❤️ using Streamlit + NLP + ML</center>", unsafe_allow_html=True)