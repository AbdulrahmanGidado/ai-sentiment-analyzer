import streamlit as st
import joblib

# Load model and vectorizer
model = joblib.load("sentiment-model.pkl")
vectorizer = joblib.load("tfidf-vectorizer.pkl")

# Page config
st.set_page_config(page_title="Sentiment Analyzer", page_icon="💬")

# App title
st.title("AI Sentiment Analysis Tool")

st.write("Analyze text sentiment using Machine Learning")

# User input
user_input = st.text_area("Enter text here:")

# Predict button
if st.button("Analyze Sentiment"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        vector = vectorizer.transform([user_input])
        prediction = model.predict(vector)[0]
        probabilities = model.predict_proba(vector)[0]
        confidence = max(probabilities)
        
        if prediction.lower() == 'positive':
            st.success(f"😊 Positive, (Confidence: {confidence:.2f})", icon="✅")
        elif prediction.lower() == "negative":
            st.error(f"😠 Negative (Confidence: {confidence:.2f})")
        else:
            st.info(f"😐 Neutral (Confidence: {confidence:.2f})")