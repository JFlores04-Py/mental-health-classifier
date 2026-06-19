import streamlit as st
import pickle
import pandas as pd

st.set_page_config(page_title="Mental Health Classifier", layout="centered")
st.title("🧠 Mental Health Language Classifier")
st.markdown("Detects if a Reddit-style post indicates **Depression** or **Anxiety**.")

# Load the model and vectorizer
@st.cache_resource
def load_models():
    model = pickle.load(open("mental_health_model.pkl", "rb"))
    vectorizer = pickle.load(open("tfidf_vectorizer.pkl", "rb"))
    return model, vectorizer

model, vectorizer = load_models()
st.success("✅ Model loaded successfully!")

# User input
user_input = st.text_area("Paste a Reddit post or mental health discussion:", height=150)

if st.button("Classify"):
    if user_input.strip():
        # Transform and predict
        input_vec = vectorizer.transform([user_input])
        prediction = model.predict(input_vec)[0]
        proba = model.predict_proba(input_vec)[0]
        
        # Map to labels (YOUR MODEL USES 'anxiety' and 'depression')
        class_map = {0: "Depression", 1: "Anxiety"}
        result = class_map.get(prediction, "Unknown")
        
        st.subheader(f"Prediction: **{result}**")
        st.metric("Confidence Score", f"{max(proba)*100:.1f}%")
        
        # Show probability bar chart
        prob_df = pd.DataFrame([proba], columns=list(class_map.values()))
        st.bar_chart(prob_df.T)
    else:
        st.warning("Please enter some text.")