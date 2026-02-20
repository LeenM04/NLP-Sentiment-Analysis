import streamlit as st
import joblib
import re
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

model_pipeline = joblib.load('sentiment_pipeline.pkl')

def clean_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return " ".join(words)

st.title("Sentiment Analysis - IMDB Reviews")
st.write("Enter a movie review to analyze its sentiment:")

user_input = st.text_area("Type your review here:")

if st.button("Analyze Sentiment"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        cleaned_text = clean_text(user_input)
        prediction = model_pipeline.predict([cleaned_text])[0]
        probability = model_pipeline.predict_proba([cleaned_text])[0]

        sentiment = "Positive 😀" if prediction == 1 else "Negative 😞"
        st.subheader("Prediction Result:")
        st.write(sentiment)
        st.write(f"Confidence: {max(probability):.2f}")
