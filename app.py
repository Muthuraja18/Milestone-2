import os
import pandas as pd
import numpy as np
import requests
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import time
from textblob import TextBlob
import streamlit as st
import seaborn as sns
import plotly.express as px
from datetime import datetime, timedelta
import speech_recognition as sr
import tempfile

GROQ_API_KEY = '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'
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

csv_file_path = r"context.csv"
output_csv_path = r"contents (2).csv"

# === Load CSV ===
def load_csv_safely(file_path):
    try:
        df = pd.read_csv(file_path, encoding='latin1', on_bad_lines='skip')
        required_columns = ['question', 'product', 'price', 'features', 'ratings', 'discount']
        for column in required_columns:
            if column not in df.columns:
                raise Exception(f"Missing required column: {column}")
        if 'Timestamp' not in df.columns:
            df['Timestamp'] = pd.NaT
        return df
    except Exception as e:
        st.error(f"Error loading CSV: {e}")
        return None

dataset = load_csv_safely(csv_file_path)
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

def filter_data_by_date(data, date_filter):
    data['Timestamp'] = pd.to_datetime(data['Timestamp'], errors='coerce')
    if date_filter == "Today":
        start_date = datetime.now().replace(hour=0, minute=0, second=0)
        data = data[data['Timestamp'] >= start_date]
    elif date_filter == "One Week":
        start_date = datetime.now() - timedelta(weeks=1)
        data = data[data['Timestamp'] >= start_date]
    return data

def get_groq_response(query):
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "llama3-8b-8192",
        "messages": [{"role": "user", "content": query}]
    }
    try:
        response = requests.post(GROQ_API_URL, headers=headers, json=payload)
        response.raise_for_status()
        data = response.json()
        return data['choices'][0]['message']['content'] if 'choices' in data else "No response."
    except Exception as e:
        return f"Error: {e}"

def transcribe_audio_file(file):
    recognizer = sr.Recognizer()
    with sr.AudioFile(file) as source:
        audio = recognizer.record(source)
        try:
            return recognizer.recognize_google(audio)
        except Exception as e:
            return f"Speech recognition error: {e}"

def is_greeting(text):
    return any(greet in text.lower() for greet in ["hello", "hi", "hey", "good morning", "good evening"])

def respond_to_greeting():
    return "Hi there! How can I assist you today? 😊"

def extract_product_name(query):
    for product in dataset['product'].fillna('Unknown').astype(str):
        if product.lower() in query.lower():
            return product
    return None

def find_answer(query):
    if dataset is None:
        return "Dataset not loaded properly."
    query_embedding = embedding_model.encode([query])
    combined_columns = dataset['question'].fillna('') + " " + dataset['product'].fillna('') + " " + dataset['features'].fillna('')
    combined_embeddings = embedding_model.encode(combined_columns.tolist())
    similarities = cosine_similarity(query_embedding, combined_embeddings)
    closest_idx = np.argmax(similarities)
    if similarities[0][closest_idx] < 0.5:
        return "Sorry, no product found."
    row = dataset.iloc[closest_idx]
    save_query_to_csv(query, row['product'], row['price'], row['features'], row['ratings'], row['discount'])
    if "price" in query.lower():
        return f"The price of {row['product']} is {row['price']}"
    elif "features" in query.lower():
        return f"Features of {row['product']}: {row['features']}"
    elif "discount" in query.lower():
        return f"The discount on {row['product']} is {row['discount']}%"
    return f"Product: {row['product']}\nPrice: {row['price']}\nFeatures: {row['features']}\nRatings: {row['ratings']}\nDiscount: {row['discount']}%"

def save_query_to_csv(query, product, price, features, ratings, discount):
    new_entry = pd.DataFrame([{
        'question': query,
        'product': product,
        'price': price,
        'features': features,
        'ratings': ratings,
        'discount': discount,
        'Timestamp': datetime.now()
    }])
    new_entry.to_csv(output_csv_path, mode='a', header=not os.path.exists(output_csv_path), index=False)

def analyze_sentiment_with_emoji(text):
    score = TextBlob(text).sentiment.polarity
    if score > 0: return "Positive", score, "😊"
    elif score < 0: return "Negative", score, "😞"
    else: return "Neutral", score, "😐"

def recommend_products(query):
    if dataset is None:
        return []
    query_embedding = embedding_model.encode([query])
    embeddings = embedding_model.encode(dataset['product'].fillna('Unknown'))
    similarities = cosine_similarity(query_embedding, embeddings)
    indices = np.argsort(similarities[0])[-3:][::-1]
    return [dataset.iloc[i].to_dict() for i in indices]

def continuous_interaction():
    st.title("🎙️ AI Product Assistant - Upload Audio")
    audio_file = st.file_uploader("Upload your voice query (WAV/MP3)", type=["wav", "mp3", "m4a"])
    if audio_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(audio_file.read())
            tmp_path = tmp.name
        transcribed = transcribe_audio_file(tmp_path)
        st.success(f"Recognized Text: {transcribed}")

        if is_greeting(transcribed):
            st.info(respond_to_greeting())
        else:
            st.markdown("#### 🔍 Query Response")
            groq_reply = get_groq_response(transcribed)
            st.write(f"Groq API: {groq_reply}")
            product = extract_product_name(transcribed)
            if product:
                st.success(find_answer(transcribed))
            else:
                st.warning("Product not found directly, using best match.")
                st.info(find_answer(transcribed))
            sentiment, score, emoji = analyze_sentiment_with_emoji(transcribed)
            st.write(f"Sentiment: {sentiment} ({score:.2f}) {emoji}")
            st.markdown("---")
            st.markdown("### 🛍️ Recommendations")
            for rec in recommend_products(transcribed):
                st.write(rec)
                st.markdown("---")

def display_dashboard():
    st.title("📊 Product Dashboard")
    time_filter = st.sidebar.selectbox("Select time range", ["All Time", "Today", "One Week"])
    df = pd.read_csv(output_csv_path, on_bad_lines='skip')
    df['Timestamp'] = pd.to_datetime(df.get('Timestamp', pd.NaT))
    df = filter_data_by_date(df, time_filter)
    st.subheader("Recent Queries")
    st.write(df.tail(10))
    st.subheader("Sentiment Analysis")
    sentiments = df['question'].apply(lambda x: analyze_sentiment_with_emoji(x)[0]).value_counts()
    st.plotly_chart(px.pie(values=sentiments.values, names=sentiments.index, title="Sentiment Distribution"))
    df['sentiment_score'] = df['question'].apply(lambda x: analyze_sentiment_with_emoji(x)[1])
    st.plotly_chart(px.line(df, x='Timestamp', y='sentiment_score', title="Sentiment Over Time"))
    st.plotly_chart(px.bar(df['product'].value_counts().head(10), title="Top Products"))

if __name__ == '__main__':
    mode = st.sidebar.radio("Choose Mode", ["Speech Recognition", "Dashboard"])
    if mode == "Speech Recognition":
        continuous_interaction()
    else:
        display_dashboard()
