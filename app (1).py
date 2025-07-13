import streamlit as st
import pandas as pd
import joblib
import string
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Load model and vectorizer
model = joblib.load("model_lr.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

def clean_text(text):
    if isinstance(text, str):
        text = text.lower()
        text = ''.join([c for c in text if c not in string.punctuation])
        words = text.split()
        words = [word for word in words if word not in stop_words]
        return ' '.join(words)
    return ""

# Streamlit UI
st.set_page_config(page_title="📰 Fake News Detection", layout="centered")
st.title("📰 Fake News Detection App")

# Section 1: Single Article
st.subheader("📌 Check a Single News Article or Headline")
user_input = st.text_area("Enter news text here")

if st.button("Predict Single"):
    cleaned = clean_text(user_input)
    vectorized = vectorizer.transform([cleaned])
    prediction = model.predict(vectorized)[0]
    confidence = model.predict_proba(vectorized)[0][prediction]
    if prediction == 1:
        st.success(f"🟢 Real News (Confidence: {confidence:.2f})")
    else:
        st.error(f"🔴 Fake News (Confidence: {confidence:.2f})")

# Section 2: Bulk CSV Upload
st.subheader("📁 Upload a CSV File with News Articles")

uploaded_file = st.file_uploader("Upload a CSV file (must have a 'text' column)", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    if 'text' not in df.columns:
        st.warning("⚠️ CSV must contain a column named 'text'.")
    else:
        df['cleaned'] = df['text'].apply(clean_text)
        vectors = vectorizer.transform(df['cleaned'])
        df['prediction'] = model.predict(vectors)
        df['confidence'] = model.predict_proba(vectors).max(axis=1)
        df['label'] = df['prediction'].apply(lambda x: 'Real' if x == 1 else 'Fake')

        st.success("✅ Prediction complete!")
        st.write(df[['text', 'label', 'confidence']])
