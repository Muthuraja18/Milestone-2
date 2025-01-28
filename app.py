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
from datetime import datetime, timedelta
import gspread
from google.oauth2.service_account import Credentials

# Set up paths
csv_file_path = r"C:\Users\Muthuraja\OneDrive\Attachments\Desktop\second\context.csv"  # Path to your CSV file
output_csv_path = r"C:\Users\Muthuraja\OneDrive\Attachments\Desktop\second\contents.csv"  # Path to save query results

# Google Sheets setup
SCOPE = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
CREDS_PATH = r"C:\Users\Muthuraja\Downloads\modern-cycling-444916-g6-82c207d3eb47.json"  # Provide your Google credentials path

# Initialize Google Sheets connection
def initialize_google_sheets():
    credentials = Credentials.from_service_account_file(CREDS_PATH, scopes=SCOPE)
    try:
        client = gspread.authorize(credentials)
        sheet = client.open("infosys").sheet1  # Change Google Sheet name to "SalesStores"
        return sheet
    except gspread.exceptions.APIError as e:
        st.error(f"Google Sheets API error: {e}")
        return None

sheet = initialize_google_sheets()

# Function to safely load the CSV dataset
def load_csv_safely(file_path):
    try:
        df = pd.read_csv(file_path, on_bad_lines='skip')  # Skips malformed lines
        required_columns = ['question', 'product', 'price', 'features', 'ratings', 'discount']
        for column in required_columns:
            if column not in df.columns:
                raise Exception(f"CSV does not contain the required column: '{column}'. Please check your CSV.")
        
        if 'Timestamp' not in df.columns:
            df['Timestamp'] = pd.NaT  # Set it to NaT (Not a Time) initially
        
        return df
    except pd.errors.ParserError as e:
        st.error(f"Error reading CSV file: {e}")
        return None
    except Exception as e:
        st.error(f"An error occurred: {e}")
        return None

dataset = load_csv_safely(csv_file_path)  # Load the dataset safely
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')  # Pre-trained sentence transformer model

# Function to filter data by date
def filter_data_by_date(data, date_filter):
    data['Timestamp'] = pd.to_datetime(data['Timestamp'], errors='coerce')
    if date_filter == "Today":
        start_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        data = data[data['Timestamp'] >= start_date]
    elif date_filter == "One Week":
        start_date = datetime.now() - timedelta(weeks=1)
        data = data[data['Timestamp'] >= start_date]
    
    return data

# Function to recognize speech using SpeechRecognition and PyAudio in chunks
def listen_to_speech():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        recognizer.adjust_for_ambient_noise(source)
        st.write("Listening...")
        
        try:
            audio = recognizer.listen(source, timeout=5, phrase_time_limit=10)
            st.write("Recognizing...")
            text = recognizer.recognize_google(audio)
            st.write(f"Recognized: {text}")
            return text
        except sr.UnknownValueError:
            st.error("Sorry, I could not understand the audio.")
            return None
        except sr.RequestError:
            st.error("Could not request results from Google Speech Recognition service.")
            return None
        except Exception as e:
            st.error(f"An error occurred: {e}")
            return None

# Function to check if the text is a greeting
def is_greeting(text):
    greetings = ["hello", "hi", "hey", "good morning", "good afternoon", "good evening", "hola"]
    return any(greeting in text.lower() for greeting in greetings)

# Function to respond to greetings
def respond_to_greeting():
    st.write("Hi there! How can I assist you today? 😊")

# Function to extract the product name from the query
def extract_product_name(query):
    for product in dataset['product'].fillna('Unknown').astype(str):
        if product.lower() in query.lower():
            return product
    return None

# Function to find the best matching answer using embeddings
def find_answer(query):
    if dataset is None:
        return "Dataset not loaded properly."
    
    query_embedding = embedding_model.encode([query])
    dataset_embeddings = embedding_model.encode(dataset['question'].tolist())
    
    similarities = cosine_similarity(query_embedding, dataset_embeddings)
    closest_idx = np.argmax(similarities)
    
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

# Function to save the query and answer to 'context.csv'
def save_query_to_csv(query, product_name, price, features, ratings, discount):
    new_entry = {
        'question': query,
        'product': product_name,
        'price': price,
        'features': features,
        'ratings': ratings,
        'discount': discount,
        'Timestamp': datetime.now()
    }
    new_entry_df = pd.DataFrame([new_entry])
    new_entry_df.to_csv(output_csv_path, mode='a', header=not os.path.exists(output_csv_path), index=False)

# Function for sentiment analysis using TextBlob with emojis
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

# Function to provide product recommendations based on the query
def recommend_products(query):
    if dataset is None:
        return "Dataset not loaded properly."

    dataset['product'] = dataset['product'].fillna('Unknown').astype(str)
    query_embedding = embedding_model.encode([query])
    dataset_embeddings = embedding_model.encode(dataset['product'].tolist())
    similarities = cosine_similarity(query_embedding, dataset_embeddings)
    top_indices = np.argsort(similarities[0])[-3:][::-1]
    
    recommendations = []
    for idx in top_indices:
        product = dataset.iloc[idx]
        recommendations.append({
            'product': product['product'],
            'price': product['price'],
            'features': product['features'],
            'ratings': product['ratings'],
            'discount': product['discount']
        })
    
    while len(recommendations) < 3:
        recommendations.append({
            'product': 'No recommendation available',
            'price': 'N/A',
            'features': 'N/A',
            'ratings': 'N/A',
            'discount': 'N/A'
        })
    
    return recommendations

# Function to handle the entire continuous interaction loop
def continuous_interaction():
    st.title("Speech Recognition with Product Queries")
    if st.button("Start Speech Recognition"):
        while True:
            user_input = listen_to_speech()
            if user_input:
                if is_greeting(user_input):
                    respond_to_greeting()
                    continue
                product_name = extract_product_name(user_input)
                if product_name:
                    st.write(f"Let me check the details for {product_name}:")
                    product_details = dataset[dataset['product'].str.lower() == product_name.lower()]
                    if not product_details.empty:
                        product_info = product_details.iloc[0]
                        st.write(f"Product: {product_info['product']}")
                        st.write(f"Price: {product_info['price']}")
                        st.write(f"Features: {product_info['features']}")
                        st.write(f"Ratings: {product_info['ratings']}")
                        st.write(f"Discount: {product_info['discount']}%")
                    else:
                        st.write("Sorry, I couldn't find the product you're asking for.")
                else:
                    answer = find_answer(user_input)
                    st.write(f"Answer: {answer}")

                sentiment, sentiment_score, emoji = analyze_sentiment_with_emoji(user_input)
                st.write(f"Sentiment: {sentiment} (Score: {sentiment_score}) {emoji}")

                st.write("Here are some product recommendations based on your query: ")
                recommendations = recommend_products(user_input)
                for idx, rec in enumerate(recommendations, 1):
                    st.write(f"Recommendation {idx}:")
                    st.write(f"Product: {rec['product']}")
                    st.write(f"Price: {rec['price']}")
                    st.write(f"Features: {rec['features']}")
                    st.write(f"Ratings: {rec['ratings']}")
                    st.write(f"Discount: {rec['discount']}%")
                    st.write("---")

# Dashboard function with time filtering
def display_dashboard():
    st.title("Product Dashboard")
    st.write("Welcome to the product query dashboard!")
    
    time_filter = st.sidebar.selectbox(
        "Select time period", 
        ["All Time", "Today", "One Week"]
    )
    
    query_results_df = pd.read_csv(output_csv_path, on_bad_lines='skip')
    
    if 'Timestamp' not in query_results_df.columns:
        query_results_df['Timestamp'] = pd.to_datetime('now')
    
    query_results_df = filter_data_by_date(query_results_df, time_filter)
    
    st.subheader(f"Recent Queries Summary ({time_filter})")
    st.write(query_results_df.tail(10))
    
    sentiment_counts = query_results_df['question'].apply(lambda x: analyze_sentiment_with_emoji(x)[0]).value_counts()
    st.subheader(f"Sentiment Analysis Distribution ({time_filter})")
    st.write(sentiment_counts)
    
    sentiment_fig = px.pie(
        sentiment_counts, 
        names=sentiment_counts.index, 
        values=sentiment_counts.values, 
        title=f"Sentiment Distribution of Queries ({time_filter})"
    )
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

# Main code to run the app
if __name__ == '__main__':
    mode = st.sidebar.radio("Select Mode", ("Speech Recognition", "Dashboard"))
    
    if mode == "Speech Recognition":
        continuous_interaction()
    elif mode == "Dashboard":
        display_dashboard()
