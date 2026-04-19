import streamlit as st
from transformers import pipeline
import joblib
import numpy as np
import re

# Load model and vectorizer
model = joblib.load("model.pkl")
vectorizer = joblib.load("tfidf-vectorizer.pkl")

@st.cache_resources
def load_transformer():
    return pipeline('sentiment-analysis', model="cardiffnlp/twitter-roberta-base-sentiment")

transformer_model = load_transformer()

# Clean function
# def clean(text):
#     text = str(text).lower()                 # Standardize to lowercase
#     text = re.sub(r'http\S+|www\S+|https\S+', '', text) # Remove URLs
#     text = re.sub(r'@\w+', '', text)          # Remove Twitter handles (@mentions)
#     text = re.sub(r'[^a-z\s]', ' ', text)     # Remove special characters and numbers
#     text = ' '.join(text.split())             # Remove extra whitespace
#     return text

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
        # cleaned_text = clean(user_input)
        # vector = vectorizer.transform([cleaned_text])
        # prediction = model.predict(vector)[0]
        # probabilities = model.predict_proba(vector)[0]
        # confidence = max(probabilities)

        # if confidence < 0.55:
        #     prediction = "neutral"
        result = transformer_model(user_input)[0]

        label = result['label']
        score = result['score']
        
        
        if label.lower() == 'positive':
            st.success(f"😊 Positive, (Confidence: {confidence:.2f})")
        elif label.lower() == "negative":
            st.error(f"😠 Negative (Confidence: {confidence:.2f})")
        else:
            st.info(f"😐 Neutral (Confidence: {confidence:.2f})")