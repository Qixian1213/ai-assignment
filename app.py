import streamlit as st
import joblib
import re
import string

# ================================
# PAGE CONFIGURATION
# ================================
st.set_page_config(
    page_title="AI Sentiment Analyzer",
    page_icon="🎭",
    layout="centered",
    initial_sidebar_state="expanded"
)

# ================================
# CUSTOM CSS
# ================================
st.markdown("""
<style>
.main { padding-top: 2rem; }
.stButton>button {
    width: 100%;
    background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    color: white;
    font-weight: 600;
    padding: 0.75rem;
    border-radius: 10px;
    border: none;
    font-size: 1.1rem;
    transition: transform 0.2s;
}
.stButton>button:hover {
    transform: translateY(-2px);
    box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
}
.sentiment-card {
    padding: 1.5rem;
    border-radius: 15px;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    text-align: center;
    margin: 1.5rem 0;
    box-shadow: 0 8px 20px rgba(0,0,0,0.1);
}
h1 {
    background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 2.5rem;
    font-weight: 800;
    text-align: center;
    margin-bottom: 0.5rem;
}
.subtitle {
    text-align: center;
    color: #666;
    font-size: 1.1rem;
    margin-bottom: 2rem;
}
</style>
""", unsafe_allow_html=True)

# ================================
# LOAD MODELS & VECTORIZER
# ================================
@st.cache_resource
def load_models():
    models = {
        "Logistic Regression": joblib.load("logistic_model.joblib"),
        "Naive Bayes": joblib.load("nb_model.joblib"),
        "SVM": joblib.load("svm_model.joblib"),
        "Stacking": joblib.load("stacking_model.joblib")
    }
    vectorizer = joblib.load("countvectorizer.joblib")
    return models, vectorizer

models, cv = load_models()

# ================================
# TEXT CLEANING FUNCTION
# ================================
def clean_text(text):
    # Remove words with numbers
    text = re.sub(r"\w*\d\w*", " ", text)
    # Remove punctuation
    text = re.sub(f"[{re.escape(string.punctuation)}]", " ", text.lower())  
    return text.strip()

# ================================
# SENTIMENT EMOJI MAPPER
# ================================
def get_sentiment_emoji(sentiment):
    return {
        "positive": "😊",
        "negative": "😞",
        "neutral": "😐"
    }.get(sentiment.lower(), "🎭")

# ================================
# SIDEBAR CONFIG
# ================================
def sidebar_config():
    # Model Selection
    st.sidebar.header("⚙️ Configuration")
    model_choice = st.sidebar.selectbox(
        "Select AI Model:",
        tuple(models.keys()),
        help="Choose the machine learning model for prediction"
    )
    st.sidebar.divider()
    st.sidebar.subheader("📊 Model Info")
    model_info = {
        "Logistic Regression": "Fast and interpretable, great for binary classification",
        "Naive Bayes": "Probabilistic classifier based on Bayes' theorem",
        "SVM": "Powerful for high-dimensional spaces",
        "Stacking": "Ensemble method combining multiple models"
    }
    st.sidebar.info(model_info[model_choice])
    return model_choice

# ================================
# MAIN APP
# ================================
def main():
    st.markdown("# 🎭 AI Sentiment Analyzer")
    st.markdown('<p class="subtitle">Discover the emotion behind your text with advanced machine learning</p>', unsafe_allow_html=True)

    model_choice = sidebar_config()
    
    st.subheader("✍️ Enter Your Text")
    user_input = st.text_area(
        "Type or paste your text below:",
        height=150,
        placeholder="e.g., This movie was absolutely fantastic! I loved every moment of it.",
        label_visibility="collapsed"
    )
    
    st.caption(f"📊 Character count: {len(user_input)}")

    if st.button("🔮 Analyze Sentiment"):
        if not user_input.strip():
            st.warning("⚠️ Please enter some text to analyze!")
            return

        with st.spinner("🤔 Analyzing sentiment..."):
            # Clean the input
            clean_input = clean_text(user_input)
            # Vectorize
            text_vectorized = cv.transform([clean_input])
            model = models[model_choice]

            # Predict
            prediction = model.predict(text_vectorized)[0]
            probabilities = model.predict_proba(text_vectorized)[0]

            # Display sentiment
            emoji = get_sentiment_emoji(prediction)
            st.markdown(f"""
                <div class="sentiment-card">
                    <h2 style="margin:0; color: white;">{emoji} {prediction.upper()}</h2>
                    <p style="margin:0.5rem 0 0 0; opacity: 0.9;">Detected Sentiment</p>
                </div>
            """, unsafe_allow_html=True)

            # Display confidence
            st.subheader("📈 Confidence Breakdown")
            for label, score in zip(model.classes_, probabilities):
                col1, col2 = st.columns([1, 4])
                with col1:
                    st.write(f"**{label.capitalize()}**")
                with col2:
                    st.progress(score)
                    st.caption(f"{score:.1%}")
    
    # Footer
    st.divider()
    st.markdown("""
        <div style="text-align: center; color: #666; padding: 1rem;">
            <p>💡 <strong>Tip:</strong> Try different models to compare their predictions!</p>
        </div>
    """, unsafe_allow_html=True)

# ================================
# RUN APP
# ================================
if __name__ == "__main__":
    main()
