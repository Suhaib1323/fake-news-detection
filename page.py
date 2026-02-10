import streamlit as st
import joblib
import numpy as np
import plotly.express as px
import re


# PAGE CONFIG

st.set_page_config(
    page_title="Fake News Detection AI",
    page_icon="🧠",
    layout="wide"
)


# DARK THEME CSS

st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
    color: white;
}

.center-container {
    max-width: 850px;
    margin: auto;
}

.stat-card {
    background: rgba(255,255,255,0.05);
    padding: 25px;
    border-radius: 15px;
    backdrop-filter: blur(10px);
    border: 1px solid rgba(255,255,255,0.1);
    text-align: center;
}

.big-number {
    font-size: 26px;
    font-weight: bold;
    color: #00f2ff;
}

.sub-text {
    color: #aaaaaa;
    font-size: 14px;
}

.verdict-fake {
    background: rgba(255,75,75,0.15);
    padding: 18px;
    border-radius: 12px;
    border: 1px solid #ff4b4b;
    font-weight: bold;
}

.verdict-real {
    background: rgba(0,255,150,0.15);
    padding: 18px;
    border-radius: 12px;
    border: 1px solid #00ff99;
    font-weight: bold;
}

textarea {
    background-color: rgba(255,255,255,0.05) !important;
    color: white !important;
}
</style>
""", unsafe_allow_html=True)


model = joblib.load("fake_news_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")


st.markdown('<div class="center-container">', unsafe_allow_html=True)

st.title("🧠 Fake News Detection AI")
st.caption("Advanced NLP System using TF-IDF + Logistic Regression")
st.markdown("---")


 #MODEL INFO CARDS

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    <div class="stat-card">
        <div class="sub-text">Model Accuracy</div>
        <div class="big-number">98.28%</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="stat-card">
        <div class="sub-text">Algorithm Used</div>
        <div class="big-number">TF-IDF + Logistic Regression</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)


# INPUT BOX

st.subheader("✍ Paste News Article Below")
user_input = st.text_area("", height=180)


# PREDICTION

if st.button("🚀 Analyze with AI"):

    if user_input.strip() != "":

        # Vectorize input
        vectorized_input = vectorizer.transform([user_input])

        # Prediction
        prediction = model.predict(vectorized_input)[0]
        probabilities = model.predict_proba(vectorized_input)[0]

        fake_prob = probabilities[0]
        real_prob = probabilities[1]

        st.markdown("<br>", unsafe_allow_html=True)
        st.subheader("🔎 AI Verdict")

        # Verdict Card
        if prediction == 0:
            st.markdown(
                f"<div class='verdict-fake'>🚨 This News is FAKE<br>Confidence: {fake_prob*100:.2f}%</div>",
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"<div class='verdict-real'>✅ This News is REAL<br>Confidence: {real_prob*100:.2f}%</div>",
                unsafe_allow_html=True
            )

        st.markdown("<br>", unsafe_allow_html=True)

    
        # PROBABILITY CHART
        prob_df = {
            "Category": ["Fake", "Real"],
            "Probability": [fake_prob, real_prob]
        }

        fig = px.bar(
            prob_df,
            x="Category",
            y="Probability",
            color="Category",
            color_discrete_map={"Fake": "#ff4b4b", "Real": "#00ff99"},
            template="plotly_dark"
        )

        fig.update_layout(height=350)

        st.plotly_chart(fig, use_container_width=True, key="prob_chart")




# SUSPICIOUS WORD HIGHLIGHT

st.subheader("🚨 Suspicious Keywords")

feature_names = vectorizer.get_feature_names_out()
coef = model.coef_[0]

# Convert  to lowercase
input_text = user_input.lower()

# Tokenize 
input_words = re.findall(r"\b\w+\b", input_text)

# Keep only words that exist in model vocabulary
valid_words = [word for word in input_words if word in feature_names]

# Remove duplicates
valid_words = list(set(valid_words))

# Rank by importance
word_scores = {word: coef[list(feature_names).index(word)] for word in valid_words}

# Get top suspicious words actually in input
top_words = sorted(word_scores, key=word_scores.get, reverse=True)[:10]

highlighted_text = user_input

for word in top_words:
    pattern = re.compile(rf"\b{re.escape(word)}\b", re.IGNORECASE)
    highlighted_text = pattern.sub(
        lambda match: f"<span style='color:#ff4b4b; font-weight:bold;'>{match.group(0)}</span>",
        highlighted_text
    )

st.markdown(
    f"<div style='line-height:1.7; font-size:15px;'>{highlighted_text}</div>",
    unsafe_allow_html=True
)

# FOOTER
st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:gray;'>Developed by Suhaib Shaikh | Advanced NLP ML Project 🚀</p>",
    unsafe_allow_html=True
)
