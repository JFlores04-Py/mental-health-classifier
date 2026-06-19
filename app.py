import streamlit as st
import pickle
import pandas as pd
import numpy as np

# --- Page Config ---
st.set_page_config(
    page_title="Mental Health Classifier",
    page_icon="🧠",
    layout="wide",
)

# --- Load Model ---
@st.cache_resource
def load_models():
    model = pickle.load(open("mental_health_model.pkl", "rb"))
    vectorizer = pickle.load(open("tfidf_vectorizer.pkl", "rb"))
    return model, vectorizer

model, vectorizer = load_models()

# --- Custom CSS ---
st.markdown("""
<style>
    .pred-box { padding: 20px; border-radius: 10px; text-align: center; margin: 10px 0; }
    .depression-box { background-color: #f0e6f6; border-left: 8px solid #6c3483; }
    .anxiety-box { background-color: #fef9e7; border-left: 8px solid #d4ac0d; }
    .uncertain-box { background-color: #f2f3f4; border-left: 8px solid #85929e; }
</style>
""", unsafe_allow_html=True)

st.title("🧠 Mental Health Language Classifier")
st.markdown("**Detects if a Reddit-style post suggests Depression or Anxiety**")
st.caption("Built with Logistic Regression on 40k+ Reddit posts | 90% Test Accuracy")

# --- Sidebar ---
with st.sidebar:
    st.header("ℹ️ About")
    st.markdown("""
    This classifier distinguishes between:
    - **Depression**: hopelessness, sadness, emptiness.
    - **Anxiety**: worry, panic, fear, stress.
    """)
    st.caption("Data from r/depression and r/anxiety.")

# --- Main Layout ---
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📝 Enter Text")
    st.caption("Try examples:")
    ex1, ex2 = st.columns(2)
    if "input_text" not in st.session_state:
        st.session_state.input_text = ""
    def set_example(text):
        st.session_state.input_text = text
    if ex1.button("😔 Depressive"):
        set_example("i feel so empty and broken today. i cant find the energy to get out of bed and everything seems pointless. i dont want to be here anymore.")
    if ex2.button("😰 Anxious"):
        set_example("my heart wont stop racing and i keep thinking about all the things that could go wrong. i feel like i cant breathe and im so stressed about work.")
    
    user_input = st.text_area("Paste your text:", value=st.session_state.input_text, height=200)
    classify_btn = st.button("🔍 Classify", type="primary", use_container_width=True)

with col2:
    st.subheader("📊 Results")
    if classify_btn and user_input.strip():
        input_vec = vectorizer.transform([user_input])
        prediction = model.predict(input_vec)[0]   # This is a string: 'depression' or 'anxiety'
        proba = model.predict_proba(input_vec)[0]
        
        # Directly use the prediction string
        result = prediction.capitalize()  # 'Depression' or 'Anxiety'
        confidence = max(proba) * 100
        
        # Choose box style
        if result == "Depression":
            box_class = "depression-box"; emoji = "💔"
        elif result == "Anxiety":
            box_class = "anxiety-box"; emoji = "😰"
        else:
            box_class = "uncertain-box"; emoji = "🤔"
        
        st.markdown(f"""
        <div class="pred-box {box_class}">
            <span style="font-size: 48px;">{emoji}</span>
            <h1 style="margin: 0;">{result}</h1>
            <p style="font-size: 18px; margin: 0;">Confidence: <b>{confidence:.1f}%</b></p>
        </div>
        """, unsafe_allow_html=True)
        
        st.caption("Confidence")
        st.progress(int(confidence), text=f"{confidence:.1f}%")
        st.caption("Probability Distribution")
        class_names = model.classes_
        prob_df = pd.DataFrame([proba], columns=[name.capitalize() for name in class_names])
        st.bar_chart(prob_df.T)
    else:
        st.info("👆 Enter text and click Classify.")

st.divider()
st.caption("⚠️ Disclaimer: Not a medical diagnostic device.")
