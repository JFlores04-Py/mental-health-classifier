import streamlit as st
import pickle
import pandas as pd
import numpy as np
import re

# --- Page Config ---
st.set_page_config(
    page_title="Mental Health Classifier",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Load Model with Caching ---
@st.cache_resource
def load_models():
    model = pickle.load(open("mental_health_model.pkl", "rb"))
    vectorizer = pickle.load(open("tfidf_vectorizer.pkl", "rb"))
    return model, vectorizer

model, vectorizer = load_models()

# --- Custom CSS for nicer styling ---
st.markdown("""
<style>
    .big-font {
        font-size: 20px !important;
        font-weight: bold;
    }
    .pred-box {
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        margin: 10px 0;
    }
    .depression-box {
        background-color: #f0e6f6;
        border-left: 8px solid #6c3483;
    }
    .anxiety-box {
        background-color: #fef9e7;
        border-left: 8px solid #d4ac0d;
    }
    .uncertain-box {
        background-color: #f2f3f4;
        border-left: 8px solid #85929e;
    }
</style>
""", unsafe_allow_html=True)

# --- Title Section ---
st.title("🧠 Mental Health Language Classifier")
st.markdown("**Instantly detects if a Reddit-style post suggests Depression or Anxiety**")
st.caption("Built with Logistic Regression on 40k+ Reddit posts | 90% Test Accuracy")

# --- Sidebar: Info ---
with st.sidebar:
    st.header("ℹ️ About This Tool")
    st.markdown("""
    This classifier analyzes the language patterns in text to distinguish between:
    - **Depression**: Words indicating hopelessness, sadness, emptiness, or suicidal ideation.
    - **Anxiety**: Words indicating worry, panic, fear, or stress.
    
    *Data sourced from r/depression and r/anxiety.*
    """)
    st.divider()
    st.caption("Built with Streamlit • Scikit-learn • Pandas")

# --- Main Layout: Two Columns ---
col1, col2 = st.columns([1, 1])

# --- COLUMN 1: Input ---
with col1:
    st.subheader("📝 Enter Text")
    
    # Example buttons
    st.caption("Try these examples:")
    example_col1, example_col2 = st.columns(2)
    
    # Session state to hold input
    if "input_text" not in st.session_state:
        st.session_state.input_text = ""
    
    def set_example(text):
        st.session_state.input_text = text
    
    if example_col1.button("😔 Depressive Post"):
        set_example("i feel so empty and broken today. i cant find the energy to get out of bed and everything seems pointless. i dont want to be here anymore.")
    if example_col2.button("😰 Anxious Post"):
        set_example("my heart wont stop racing and i keep thinking about all the things that could go wrong. i feel like i cant breathe and im so stressed about work.")
    
    # Text Area
    user_input = st.text_area(
        "Paste your text here:",
        value=st.session_state.input_text,
        height=200,
        placeholder="e.g., I've been feeling really down lately and I don't see the point..."
    )
    
    classify_btn = st.button("🔍 Classify", type="primary", use_container_width=True)

# --- COLUMN 2: Results ---
with col2:
    st.subheader("📊 Prediction Results")
    
    if classify_btn and user_input.strip():
        # Clean and predict
        input_vec = vectorizer.transform([user_input])
        prediction = model.predict(input_vec)[0]
        proba = model.predict_proba(input_vec)[0]
        
        # Map labels
        class_map = {0: "Depression", 1: "Anxiety"}
        result = class_map.get(prediction, "Unknown")
        confidence = max(proba) * 100
        
        # ---- Display Prediction with Styled Box ----
        if result == "Depression":
            box_class = "depression-box"
            emoji = "💔"
        elif result == "Anxiety":
            box_class = "anxiety-box"
            emoji = "😰"
        else:
            box_class = "uncertain-box"
            emoji = "🤔"
        
        st.markdown(f"""
        <div class="pred-box {box_class}">
            <span style="font-size: 48px;">{emoji}</span>
            <h1 style="margin: 0;">{result}</h1>
            <p style="font-size: 18px; margin: 0;">Confidence: <b>{confidence:.1f}%</b></p>
        </div>
        """, unsafe_allow_html=True)
        
        # ---- Confidence Gauge ----
        st.caption("Confidence Level")
        if confidence > 80:
            color = "green"
        elif confidence > 50:
            color = "orange"
        else:
            color = "red"
        st.progress(int(confidence), text=f"**{confidence:.1f}%**")
        
        # ---- Probability Breakdown ----
        st.caption("Probability Distribution")
        prob_df = pd.DataFrame({
            "Class": ["Depression", "Anxiety"],
            "Probability": proba
        })
        st.bar_chart(prob_df.set_index("Class"))
        
        # ---- Top Influential Words (Bonus) ----
        # Get feature names and coefficients
        feature_names = vectorizer.get_feature_names_out()
        coefs = model.coef_[0]  # Positive = Depression, Negative = Anxiety
        
        input_features = input_vec.toarray()[0]
        # Get top 3 positive (Depression) and top 3 negative (Anxiety) words from this specific input
        positive_indices = np.argsort(coefs)[-5:][::-1]
        negative_indices = np.argsort(coefs)[:5]
        
        # Filter to words actually present in this text
        present_pos = [(feature_names[i], coefs[i]) for i in positive_indices if input_features[i] > 0]
        present_neg = [(feature_names[i], -coefs[i]) for i in negative_indices if input_features[i] > 0]
        
        if present_pos or present_neg:
            st.caption("Top words influencing this prediction:")
            col_plus, col_minus = st.columns(2)
            if present_pos:
                col_plus.markdown("**🔵 Depression signals:**")
                for word, val in present_pos[:3]:
                    col_plus.write(f"• `{word}`")
            if present_neg:
                col_minus.markdown("**🟡 Anxiety signals:**")
                for word, val in present_neg[:3]:
                    col_minus.write(f"• `{word}`")
    else:
        st.info("👆 Enter text and click **Classify** to see results.")

# --- Footer ---
st.divider()
st.caption("⚠️ Disclaimer: This tool is for educational and research purposes only. It is not a medical diagnostic device. If you are in crisis, please contact a mental health professional.")
