import streamlit as st
from transformers import pipeline
import joblib
import numpy as np
import re

# Load Fast model (my trained model)
model = joblib.load("model.pkl")
vectorizer = joblib.load("tfidf-vectorizer.pkl")

# Load Smart model (transformer)
@st.cache_resource
def load_transformer():
    return pipeline('sentiment-analysis', model="cardiffnlp/twitter-roberta-base-sentiment")

transformer_model = load_transformer()

mode = st.radio('Choose mode:', ['⚡ Fast', '🧠 Smart (Tranformer)'])

# Clean function
def clean(text):
    text = str(text).lower()                 # Standardize to lowercase
    text = re.sub(r'http\S+|www\S+|https\S+', '', text) # Remove URLs
    text = re.sub(r'@\w+', '', text)          # Remove Twitter handles (@mentions)
    text = re.sub(r'[^a-z\s]', ' ', text)     # Remove special characters and numbers
    text = ' '.join(text.split())             # Remove extra whitespace
    return text

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

        # ⚡ Fast
        if mode == '⚡ Fast':
            cleaned_text = clean(user_input)
            
            vector = vectorizer.transform([user_input])
            probabilities = model.predict_proba(vector)[0]
            confidence = np.max(probabilities)
            prediction = model.classes_[np.argmax(probabilities)]

            if confidence < 0.55:
                prediction = "neutral"


        # 🧠 SMART MODE
        else:
            result = transformer_model(user_input)[0]
            prediction = result['label']
            confidence = result['score']
        
        # Display result
        if prediction.lower() == 'positive':
            st.success(f"😊 Positive, (Confidence: {confidence:.2f})")
        elif prediction.lower() == "negative":
            st.error(f"😠 Negative (Confidence: {confidence:.2f})")
        else:
            st.info(f"😐 Neutral (Confidence: {confidence:.2f})")


