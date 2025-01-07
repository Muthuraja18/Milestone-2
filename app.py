import time
import random
import speech_recognition as sr
import streamlit as st
from gtts import gTTS
import io
import soundfile as sf
import sounddevice as sd
import gspread
from google.oauth2.service_account import Credentials
from textblob import TextBlob
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import datetime

chat_model = GPT2LMHeadModel.from_pretrained('gpt2')
chat_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

recognizer = sr.Recognizer()

SCOPE = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
CREDS_PATH = r"C:\Users\Muthuraja\Downloads\modern-cycling-444916-g6-82c207d3eb47.json"

# Initialize Google Sheets connection
def initialize_google_sheets():
    credentials = Credentials.from_service_account_file(CREDS_PATH, scopes=SCOPE)
    try:
        client = gspread.authorize(credentials)
        sheet = client.open("Sales").sheet1
        return sheet
    except gspread.exceptions.APIError as e:
        st.error(f"Google Sheets API error: {e}")
        return None

sheet = initialize_google_sheets()

# Function to convert speech to text using microphone input
def listen_to_speech():
    with sr.Microphone() as source:
        recognizer.adjust_for_ambient_noise(source)
        st.write("Listening...")
        audio = recognizer.listen(source)
    try:
        text = recognizer.recognize_google(audio)
        return text
    except sr.UnknownValueError:
        return None

# Function to analyze sentiment chunk-wise with delay
def analyze_sentiment_chunks(text, chunk_size=20, delay=1):
    chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
    sentiments = []

    for chunk in chunks:
        blob = TextBlob(chunk)
        sentiment_polarity = blob.sentiment.polarity
        sentiment_description = "Positive" if sentiment_polarity > 0 else "Negative" if sentiment_polarity < 0 else "Neutral"
        sentiments.append((chunk, sentiment_polarity, sentiment_description))
        time.sleep(delay)
    
    return sentiments

# Function to generate random emoji based on sentiment
def get_random_emoji(sentiment):
    positive_emojis = ["😊", "😃", "😄", "😁", "🙂"]
    negative_emojis = ["😞", "😠", "😢", "😟", "😡"]
    neutral_emojis = ["😐", "😶", "🤔", "😑", "😏"]

    if sentiment == "Positive":
        return random.choice(positive_emojis)
    elif sentiment == "Negative":
        return random.choice(negative_emojis)
    else:
        return random.choice(neutral_emojis)

# Function to generate relevant answer to user input using GPT-2 model
def generate_relevant_answer_gpt2(user_input):
    input_ids = chat_tokenizer.encode(user_input, return_tensors='pt')
    chat_output = chat_model.generate(
        input_ids, 
        max_length=150, 
        temperature=0.9, 
        num_return_sequences=1, 
        no_repeat_ngram_size=2
    )
    return chat_tokenizer.decode(chat_output[0], skip_special_tokens=True)

# Function to convert text to speech using gTTS
def text_to_speech(text):
    if not text:
        text = "I'm sorry, I didn't quite catch that. Could you please repeat?"
    tts = gTTS(text=text, lang='en')
    audio_buffer = io.BytesIO()
    tts.write_to_fp(audio_buffer)
    audio_buffer.seek(0)
    audio_data, samplerate = sf.read(audio_buffer)
    return audio_data, samplerate

# Function to play audio using sounddevice
def play_audio(audio_data, samplerate):
    sd.play(audio_data, samplerate)
    sd.wait()

# Function to update Google Sheet
def update_sheet(transcript, sentiments, relevant_answer):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    for chunk, sentiment_polarity, sentiment_description in sentiments:
        sheet.append_row([timestamp, chunk, sentiment_description, sentiment_polarity, relevant_answer])

# Main function to handle Streamlit UI and interaction
def main():
    # Streamlit UI setup
    st.markdown("""
        <style>
            .chat-container { background-color: #f0f0f0; border-radius: 10px; padding: 20px; margin: 20px; }
            .user-message { font-weight: bold; color: #1f77b4; margin-bottom: 10px; }
            .chatbot-message { font-weight: bold; color: #ff7f0e; margin-bottom: 10px; }
        </style>
    """, unsafe_allow_html=True)
    
    st.title('Real-Time Sentiment Analysis Chatbot for Sales Calls')
    st.write("This system listens to your speech, analyzes sentiment, and detects sentiment shifts in real-time.")
    
    # Placeholders for live text, sentiment, and chatbot conversation
    text_placeholder = st.empty()
    sentiment_placeholder = st.empty()
    chat_placeholder = st.empty()

    # Initialize the conversation list
    conversation = []
    conversation.append('<div class="chatbot-message">Chatbot: Hello, how can I help you today? 😊</div>')
    chat_placeholder.markdown(f'<div class="chat-container">{"".join(conversation)}</div>', unsafe_allow_html=True)

    # Main loop to continuously listen and analyze speech
    if st.button("Press to Start Recording"):
        live_text = listen_to_speech()
        
        if live_text:
            # Display the user message
            conversation.append(f'<div class="user-message">User: {live_text}</div>')
            chat_placeholder.markdown(f'<div class="chat-container">{"".join(conversation)}</div>', unsafe_allow_html=True)
            
            # Perform sentiment analysis chunk-wise with delay
            sentiments = analyze_sentiment_chunks(live_text)

            # Display sentiment results chunk-wise
            for chunk, sentiment_polarity, sentiment_description in sentiments:
                emoji = get_random_emoji(sentiment_description)
                sentiment_placeholder.write(f"Chunk: {chunk}\nSentiment: {sentiment_description} {emoji} (Score: {sentiment_polarity:.2f})")
                time.sleep(1)

            # Generate relevant answer based on user input using GPT-2 model
            relevant_answer = generate_relevant_answer_gpt2(live_text)
            conversation.append(f'<div class="chatbot-message">Chatbot: {relevant_answer}</div>')
            chat_placeholder.markdown(f'<div class="chat-container">{"".join(conversation)}</div>', unsafe_allow_html=True)

            # Store conversation in Google Sheets
            if sheet:
                try:
                    update_sheet(live_text, sentiments, relevant_answer)
                except gspread.exceptions.APIError as e:
                    st.error(f"Google Sheets API error: {e}")

            # Convert chatbot response to speech and play it
            audio_data, samplerate = text_to_speech(relevant_answer)
            play_audio(audio_data, samplerate)

            # Optionally add the sentiment response to the conversation
            conversation.append(f'<div class="chatbot-message">Sentiment Analysis: {sentiment_description} {emoji} (Score: {sentiment_polarity:.2f})</div>')

if __name__ == "__main__":
    main()
