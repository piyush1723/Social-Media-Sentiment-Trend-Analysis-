import streamlit as st
import joblib
import time

# =========================
# LOAD MODEL
# =========================
model = joblib.load("models/sentiment_model.pkl")
tfidf = joblib.load("models/tfidf_vectorizer.pkl")

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="AI Hate Speech Detector",
    page_icon="🧠",
    layout="centered"
)

# =========================
# CUSTOM UI
# =========================
st.markdown("""
<style>

/* Background */
.stApp {
    background: radial-gradient(circle at top, #0f172a, #020617);
    color: white;
    font-family: 'Segoe UI';
}

/* Container */
.block-container {
    padding-top: 2rem;
    max-width: 800px;
}

/* Title */
h1 {
    text-align: center;
    font-size: 42px !important;
    background: linear-gradient(90deg, #60a5fa, #a78bfa, #f472b6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 800;
}

/* Subtitle */
.subtitle {
    text-align: center;
    color: #94a3b8;
    margin-bottom: 20px;
}

/* Text area */
textarea {
    background: rgba(255,255,255,0.04) !important;
    color: white !important;
    border-radius: 14px !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    font-size: 16px !important;
    padding: 12px;
}

/* Button */
.stButton > button {
    width: 100%;
    padding: 14px;
    border-radius: 14px;
    background: linear-gradient(90deg, #2563eb, #7c3aed);
    color: white;
    font-size: 16px;
    border: none;
    transition: 0.3s;
    font-weight: 600;
}

.stButton > button:hover {
    transform: scale(1.03);
    box-shadow: 0 0 25px rgba(99,102,241,0.4);
}

/* Cards */
.card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    padding: 20px;
    border-radius: 16px;
    margin-top: 20px;
    box-shadow: 0 10px 30px rgba(0,0,0,0.4);
}

/* Metric boxes */
.metric {
    background: rgba(255,255,255,0.05);
    padding: 18px;
    border-radius: 14px;
    text-align: center;
}

/* Footer */
.footer {
    text-align: center;
    color: #64748b;
    font-size: 13px;
    margin-top: 30px;
}

.badge {
    display: inline-block;
    padding: 5px 10px;
    background: rgba(99,102,241,0.2);
    border-radius: 10px;
    font-size: 12px;
    margin: 3px;
}
</style>
""", unsafe_allow_html=True)

# =========================
# HEADER
# =========================
st.markdown("<h1>🧠 AI Hate Speech Detector</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Detect toxic & hate speech using Machine Learning (NLP + TF-IDF)</p>", unsafe_allow_html=True)

st.markdown("""
<div style='text-align:center;'>
<span class='badge'>NLP</span>
<span class='badge'>Machine Learning</span>
<span class='badge'>Streamlit</span>
<span class='badge'>TF-IDF</span>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# =========================
# EXAMPLE INPUTS (UX BOOST)
# =========================
st.markdown("### ✨ Try Examples:")
col1, col2, col3 = st.columns(3)

examples = [
    "I love this product",
    "I don't like you, go away",
    "You are amazing!"
]

if col1.button("Example 1"):
    st.session_state.example = examples[0]

if col2.button("Example 2"):
    st.session_state.example = examples[1]

if col3.button("Example 3"):
    st.session_state.example = examples[2]

# =========================
# INPUT
# =========================
tweet = st.text_area(
    "Enter your text below 👇",
    height=130,
    value=st.session_state.get("example", "")
)

# =========================
# PREDICTION
# =========================
if st.button("🚀 Analyze Now"):

    if tweet.strip() == "":
        st.warning("Please enter text.")
    else:

        with st.spinner("Analyzing text using AI model..."):
            time.sleep(1.2)

            tweet_tfidf = tfidf.transform([tweet])
            prediction = model.predict(tweet_tfidf)[0]
            probability = model.predict_proba(tweet_tfidf)[0]
            confidence = probability[prediction] * 100

        # =========================
        # RESULT CARD
        # =========================
        st.markdown("<div class='card'>", unsafe_allow_html=True)

        if prediction == 1:
            st.markdown("## 🚨 Hate Speech Detected")
            st.error("This text may contain toxic or harmful language.")
        else:
            st.markdown("## ✅ Normal Speech")
            st.success("This text looks safe and neutral.")

        st.write(f"### Confidence: {confidence:.2f}%")

        st.markdown("</div>", unsafe_allow_html=True)

        # =========================
        # METRICS
        # =========================
        st.markdown("<div class='card'>", unsafe_allow_html=True)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown(f"""
            <div class='metric'>
            <h4>Normal</h4>
            <h2>{probability[0]*100:.2f}%</h2>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown(f"""
            <div class='metric'>
            <h4>Hate Speech</h4>
            <h2>{probability[1]*100:.2f}%</h2>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

# =========================
# FOOTER
# =========================
st.markdown("""
<div class='footer'>
Built with ❤️ using Streamlit • NLP • Machine Learning <br>
Portfolio Project • AI Hate Speech Detection System
</div>
""", unsafe_allow_html=True)