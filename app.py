import os
import pyaudio
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import time
import speech_recognition as sr
from textblob import TextBlob
import streamlit as st
import seaborn as sns
import plotly.express as px
import requests
from datetime import datetime, timedelta
import gspread
from google.oauth2.service_account import Credentials
from dotenv import load_dotenv  # For loading environment variables
import random  # For generating random customer IDs

# Load environment variables from a .env file

csv_file_path = r"C:\Users\Muthuraja\OneDrive\Attachments\Desktop\second\database1.csv"
output_csv_path = r"C:\Users\Muthuraja\OneDrive\Attachments\Desktop\second\Book4.csv"

SCOPE = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
CREDS_PATH = r"C:\Users\Muthuraja\Downloads\modern-cycling-444916-g6-82c207d3eb47.json"  # Path to your Google credentials JSON file

# Use the provided Groq API key (you can also store this in .env)
GROQ_API_KEY = "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"  
GROQ_API_URL = 'https://api.groq.com/openai/v1/chat/completions'

def initialize_google_sheets():
    credentials = Credentials.from_service_account_file(CREDS_PATH, scopes=SCOPE)
    try:
        client = gspread.authorize(credentials)
        sheet = client.open("CRM_Interactions").sheet1  # Using CRM_Interactions as the sheet name
        return sheet
    except gspread.exceptions.APIError as e:
        st.error(f"Google Sheets API error: {e}")
        return None

sheet = initialize_google_sheets()

def load_csv_safely(file_path):
    try:
        df = pd.read_csv(file_path, on_bad_lines='skip')
        required_columns = ['question', 'product', 'price', 'features', 'ratings', 'discount', 'customer_id']
        for column in required_columns:
            if column not in df.columns:
                raise Exception(f"CSV does not contain the required column: '{column}'. Please check your CSV.")
        
        if 'Timestamp' not in df.columns:
            df['Timestamp'] = pd.NaT  # Initialize Timestamp column if it doesn't exist
        
        return df
    except pd.errors.ParserError as e:
        st.error(f"Error reading CSV file: {e}")
        return None
    except Exception as e:
        st.error(f"An error occurred: {e}")
        return None

dataset = load_csv_safely(csv_file_path)

embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

def send_groq_request(query):
    headers = {
        'Authorization': f'Bearer {GROQ_API_KEY}',
        'Content-Type': 'application/json'
    }
    
    payload = {
        'query': query
    }
    
    try:
        response = requests.post(GROQ_API_URL, headers=headers, json=payload)
        response.raise_for_status()  # Will raise an HTTPError for bad responses (4xx or 5xx)
        return response.json()  # Return the response in JSON format
    except requests.exceptions.RequestException as e:
        st.error(f"Error communicating with Groq API: {e}")
        return None

def is_greeting(text):
    greetings = ["hello", "hi", "hey", "good morning", "good afternoon", "good evening", "hola"]
    return any(greeting in text.lower() for greeting in greetings)
    
def respond_to_greeting():
    st.write("Hi there! How can I assist you today? 😊")

def extract_product_name(query):
    for product in dataset['product'].fillna('Unknown').astype(str):
        if product.lower() in query.lower():
            return product
    return None

def handle_more_products_request(query):
    if "more products" in query.lower():
        more_products = dataset[['product', 'price', 'features', 'ratings', 'discount']].head(5)
        return f"Here are some more products you might like:\n{more_products}"
    return None

def find_answer(query):
    if "more products" in query.lower():
        return handle_more_products_request(query)
    
    if dataset is None:
        return "Dataset not loaded properly."
    
    query_embedding = embedding_model.encode([query])
    combined_columns = dataset['question'].fillna('') + " " + dataset['product'].fillna('') + " " + dataset['features'].fillna('') 
    combined_embeddings = embedding_model.encode(combined_columns.tolist())
    
    similarities = cosine_similarity(query_embedding, combined_embeddings)
    
    similarity_threshold = 0.5  
    closest_idx = np.argmax(similarities)  
    highest_similarity = similarities[0][closest_idx]  
    
    if highest_similarity < similarity_threshold:
        return "Sorry, no product found for your query."
    
    closest_question = dataset.iloc[closest_idx]
    product_name = closest_question['product']
    price = closest_question['price']
    features = closest_question['features']
    ratings = closest_question['ratings']
    discount = closest_question['discount']
    
    if 'Timestamp' not in closest_question.index:
        closest_question['Timestamp'] = datetime.now()

    save_query_to_csv(query, product_name, price, features, ratings, discount)

    if "price" in query.lower():
        return f"The price of {product_name} is {price}"
    elif "features" in query.lower():
        return f"Features of {product_name}: {features}"
    elif "discount" in query.lower():
        return f"The discount on {product_name} is {discount}%"
    else:
        return f"Product: {product_name}\nPrice: {price}\nFeatures: {features}\nRatings: {ratings}\nDiscount: {discount}%"

def save_query_to_csv(query, product_name, price, features, ratings, discount):
    new_entry = {
        'question': query,
        'product': product_name,
        'price': price,
        'features': features,
        'ratings': ratings,
        'discount': discount,
        'Timestamp': datetime.now(),
        'customer_id': random.randint(1000, 9999)  # Generate a random customer ID between 1000 and 9999
    }
    new_entry_df = pd.DataFrame([new_entry])
    new_entry_df.to_csv(output_csv_path, mode='a', header=not os.path.exists(output_csv_path), index=False)

def analyze_sentiment_with_emoji(text):
    blob = TextBlob(text)
    sentiment_score = blob.sentiment.polarity
    if sentiment_score > 0:
        sentiment = "Positive"
        emoji = "😊"
    elif sentiment_score < 0:
        sentiment = "Negative"
        emoji = "😞"
    else:
        sentiment = "Neutral"
        emoji = "😐"
    return sentiment, sentiment_score, emoji


def display_sentiment_pie_chart(sentiment_counts):
    sentiment_fig = px.pie(
        sentiment_counts, 
        names=sentiment_counts.index, 
        values=sentiment_counts.values, 
        title="Sentiment Distribution",
        hole=0.3  # For a donut chart (optional)
    )
    
    sentiment_fig.update_traces(textinfo='percent+label', pull=[0.1, 0.1, 0.1])
    
    return sentiment_fig

def display_dashboard():
    st.title("Product Dashboard")
    st.write("Welcome to the product query dashboard!")

    customer_ids = dataset['customer_id'].unique()
    selected_customer_id = st.sidebar.selectbox(
        "Select Customer ID", 
        ["All Customers"] + customer_ids.tolist()
    )

    time_filter = st.sidebar.selectbox(
        "Select time period", 
        ["All Time", "Today", "One Week"]
    )

    query_results_df = pd.read_csv(output_csv_path, on_bad_lines='skip')

    if 'Timestamp' not in query_results_df.columns:
        query_results_df['Timestamp'] = pd.to_datetime('now')

    if selected_customer_id != "All Customers":
        query_results_df = query_results_df[query_results_df['customer_id'] == selected_customer_id]

    query_results_df = filter_data_by_date(query_results_df, time_filter)

    st.subheader(f"Recent Queries Summary ({time_filter})")
    st.write(query_results_df.tail(10))

    sentiment_counts = query_results_df['question'].apply(lambda x: analyze_sentiment_with_emoji(x)[0]).value_counts()
    st.subheader(f"Sentiment Analysis Distribution ({time_filter})")
    st.write(sentiment_counts)

    sentiment_fig = display_sentiment_pie_chart(sentiment_counts)
    st.plotly_chart(sentiment_fig)

    query_results_df['sentiment_score'] = query_results_df['question'].apply(lambda x: analyze_sentiment_with_emoji(x)[1])

    sentiment_time_fig = px.line(
        query_results_df, 
        x='Timestamp', 
        y='sentiment_score', 
        title=f"Sentiment Score Over Time ({time_filter})"
    )
    st.plotly_chart(sentiment_time_fig)

    product_counts = query_results_df['product'].value_counts()
    st.subheader(f"Product Popularity ({time_filter})")
    st.write(product_counts)

    product_popularity_fig = px.pie(
        product_counts, 
        names=product_counts.index, 
        values=product_counts.values, 
        title=f"Product Popularity ({time_filter})"
    )
    st.plotly_chart(product_popularity_fig)

    recommended_products = query_results_df['product'].value_counts()
    st.subheader(f"Most Recommended Products ({time_filter})")
    st.write(recommended_products)

    recommended_products_fig = px.bar(
        recommended_products, 
        x=recommended_products.index, 
        y=recommended_products.values, 
        title=f"Top Recommended Products ({time_filter})"
    )
    st.plotly_chart(recommended_products_fig)


def filter_data_by_date(query_results_df, time_filter):
    if time_filter == "Today":
        today = datetime.now().date()
        query_results_df['Timestamp'] = pd.to_datetime(query_results_df['Timestamp']).dt.date
        query_results_df = query_results_df[query_results_df['Timestamp'] == today]
    elif time_filter == "One Week":
        one_week_ago = datetime.now() - timedelta(weeks=1)
        query_results_df['Timestamp'] = pd.to_datetime(query_results_df['Timestamp'])
        query_results_df = query_results_df[query_results_df['Timestamp'] > one_week_ago]
    return query_results_df

def continuous_interaction():
    recognizer = sr.Recognizer()
    microphone = sr.Microphone()
    
    st.write("Listening for your query...")

    while True:
        with microphone as source:
            recognizer.adjust_for_ambient_noise(source)
            audio = recognizer.listen(source)
        
        try:
            query = recognizer.recognize_google(audio)
            st.write(f"Your query: {query}")
            
            if is_greeting(query):
                respond_to_greeting()
            else:
                answer = find_answer(query)
                sentiment, score, emoji = analyze_sentiment_with_emoji(query)
                st.write(f"Answer: {answer}")
                st.write(f"Sentiment: {sentiment} {emoji}")
                st.write(f"Sentiment Score: {score}")
                
        except sr.UnknownValueError:
            st.write("Sorry, I couldn't understand that.")
        except sr.RequestError:
            st.write("Sorry, there was an error with the speech recognition service.")


if __name__ == "__main__":
    st.sidebar.title("Product Query Interface")
    mode = st.sidebar.selectbox("Select Mode", ["Speech Recognition", "Dashboard"])

    if mode == "Speech Recognition":
        if st.button('Start Listening'):
            continuous_interaction() 
    elif mode == "Dashboard":
        display_dashboard()
