import re
import joblib
import numpy as np
import pandas as pd
import streamlit as st


MODEL_PATH = r"C:\Users\HP\OneDrive\Desktop\Social-Media-Sentiment-Trend-Analysis\models\sentiment_pipeline.pkl"


st.set_page_config(
    page_title="Sentiment Intelligence Platform",
    page_icon="📊",
    layout="wide",
)


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"@\w+", " ", text)
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"#", "", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


@st.cache_resource
def load_model():
    try:
        return joblib.load(MODEL_PATH)
    except Exception as e:
        st.error(f"Model loading failed: {e}")
        return None


pipeline = load_model()

if "history" not in st.session_state:
    st.session_state.history = []

st.sidebar.title("📌 Dashboard")

page = st.sidebar.radio(
    "Navigation",
    ["🏠 Analyzer", "📈 Analytics", "ℹ️ About"],
)

st.sidebar.divider()
st.sidebar.markdown("### Model Info")
st.sidebar.success("Model Loaded" if pipeline else "Model Not Loaded")
st.sidebar.write("Algorithm: LinearSVC")
st.sidebar.write("Features: TF-IDF")
st.sidebar.write("Input: Tweet text only")
st.sidebar.write("Classes: Positive, Negative, Neutral")

if pipeline is not None:
    st.sidebar.write("Loaded Classes:")
    st.sidebar.write(list(pipeline.classes_))

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
        "Positive": "I absolutely love this game, it is amazing!",
        "Negative": "This is the worst update ever, I hate it.",
        "Neutral": "The update was released today.",
    }

    choice = st.selectbox(
        "Choose Example or Write Custom",
        ["Custom"] + list(examples.keys()),
    )

    text_input = st.text_area(
        "Enter Text",
        value=examples.get(choice, ""),
        height=140,
    )

    if st.button("🚀 Predict Sentiment", use_container_width=True):
        if not text_input.strip():
            st.warning("Please enter valid text.")
            st.stop()

        if pipeline is None:
            st.error("Model not loaded properly.")
            st.stop()

        with st.spinner("Analyzing sentiment..."):
            cleaned_text = clean_text(text_input)

            if not cleaned_text:
                st.warning("Text became empty after cleaning. Please enter more meaningful text.")
                st.stop()

            pred = pipeline.predict([cleaned_text])[0]

            try:
                scores = pipeline.decision_function([cleaned_text])
                confidence = float(np.max(scores))
                confidence = 1 / (1 + np.exp(-confidence))
                confidence = round(confidence * 100, 2)
            except Exception:
                confidence = 0

        st.session_state.history.append({
            "Text": text_input[:100],
            "Cleaned_Text": cleaned_text,
            "Prediction": pred,
            "Confidence": confidence,
        })

        st.subheader("Result")

        color_map = {
            "Positive": "success",
            "Negative": "error",
            "Neutral": "info",
        }

        getattr(st, color_map.get(pred, "info"))(f"{pred} Sentiment")
        st.metric("Confidence", f"{confidence}%")
        st.progress(confidence / 100)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### Original Input")
            st.write(text_input)

        with col2:
            st.markdown("### Cleaned Text")
            st.write(cleaned_text)

    if st.session_state.history:
        st.download_button(
            "📥 Download History",
            pd.DataFrame(st.session_state.history).to_csv(index=False),
            "history.csv",
            "text/csv",
        )

elif page == "📈 Analytics":
    st.title("📈 Analytics Dashboard")

    df = pd.DataFrame(st.session_state.history)

    if not df.empty:
        st.bar_chart(df["Prediction"].value_counts())
        st.dataframe(df)
    else:
        st.info("No data available yet.")

else:
    st.title("ℹ️ About")

    st.markdown("""
    ### Sentiment Analysis System

    - 3-class sentiment classification
    - Tweet text cleaning
    - TF-IDF vectorization
    - LinearSVC classifier
    - Streamlit deployment

    ### Flow

    User Text → Cleaning → Saved Model Pipeline → Sentiment
    """)

st.markdown("---")
st.markdown(
    "<center>Built with ❤️ using Streamlit + NLP + ML</center>",
    unsafe_allow_html=True,
)