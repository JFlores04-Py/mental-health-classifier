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
    st.caption("Try these examples:")
    
    # More diverse examples
    ex_col1, ex_col2, ex_col3, ex_col4 = st.columns(4)
    if "input_text" not in st.session_state:
        st.session_state.input_text = ""
    def set_example(text):
        st.session_state.input_text = text
    
    if ex_col1.button("😔 Depressed"):
        set_example("i feel so empty and hopeless. i have no motivation to do anything. life feels meaningless.")
    if ex_col2.button("😰 Panic"):
        set_example("my heart is pounding and i feel like im having a heart attack. i cant stop shaking and i feel dizzy.")
    if ex_col3.button("😢 Sad"):
        set_example("i just feel sad all the time. nothing makes me happy anymore. i wish i could escape this feeling.")
    if ex_col4.button("😬 Worry"):
        set_example("i keep worrying about everything. i cant sleep because my mind is racing. what if something bad happens?")
    
    user_input = st.text_area("Paste your text:", value=st.session_state.input_text, height=200)
    classify_btn = st.button("🔍 Classify", type="primary", use_container_width=True)

with col2:
    st.subheader("📊 Results")
    if classify_btn and user_input.strip():
        input_vec = vectorizer.transform([user_input])
        prediction = model.predict(input_vec)[0]   # string
        proba = model.predict_proba(input_vec)[0]
        
        # Show debug info (remove later)
        with st.expander("🔧 Debug Info"):
            st.write("**Model Classes:**", model.classes_)
            st.write("**Prediction (string):**", prediction)
            st.write("**Probabilities:**", proba)
            st.write("**Feature names (first 10):**", vectorizer.get_feature_names_out()[:10])
        
        # Result
        result = prediction.capitalize()
        confidence = max(proba) * 100
        
        # Styling
        if result == "Depression":
            box_class = "depression-box"; emoji = "💔"
        elif result == "Anxiety":
            box_class = "anxiety-box"; emoji = "😰"
        else:
            box_class = "uncertain-box"; emoji = "🤔"
        
        st.markdown(f"""
        <div style="padding:20px; border-radius:10px; text-align:center; margin:10px 0; 
                    background-color: {'#f0e6f6' if result=='Depression' else '#fef9e7' if result=='Anxiety' else '#f2f3f4'};
                    border-left: 8px solid {'#6c3483' if result=='Depression' else '#d4ac0d' if result=='Anxiety' else '#85929e'};">
            <span style="font-size:48px;">{emoji}</span>
            <h1 style="margin:0;">{result}</h1>
            <p style="font-size:18px;">Confidence: <b>{confidence:.1f}%</b></p>
        </div>
        """, unsafe_allow_html=True)
        
        st.caption("Confidence Level")
        st.progress(int(confidence), text=f"{confidence:.1f}%")
        st.caption("Probability Distribution")
        class_names = model.classes_
        prob_df = pd.DataFrame([proba], columns=[name.capitalize() for name in class_names])
        st.bar_chart(prob_df.T)
        
        # Show top influential words for this text
        feature_names = vectorizer.get_feature_names_out()
        coefs = model.coef_[0]  # positive = depression, negative = anxiety (maybe reversed)
        input_features = input_vec.toarray()[0]
        # Get words present in input
        present_indices = np.where(input_features > 0)[0]
        if len(present_indices) > 0:
            # Compute contribution: coefficient * feature_value
            contributions = [(feature_names[i], coefs[i] * input_features[i]) for i in present_indices]
            # Sort by absolute contribution
            contributions.sort(key=lambda x: abs(x[1]), reverse=True)
            top_words = contributions[:5]
            st.caption("Top contributing words:")
            for word, contrib in top_words:
                label = "🔵 Depression" if contrib > 0 else "🟡 Anxiety"
                st.write(f"{label}: `{word}` ({contrib:.3f})")
    else:
        st.info("👆 Enter text and click Classify.")

st.divider()
st.caption("⚠️ Disclaimer: Not a medical diagnostic device.")
