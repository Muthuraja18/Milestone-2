import streamlit as st
import PyPDF2
import speech_recognition as sr
from transformers import GPT2LMHeadModel, GPT2Tokenizer, pipeline
from sentence_transformers import SentenceTransformer
import gspread
from google.oauth2.service_account import Credentials
import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta


SCOPE = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
CREDS_PATH = r"C:\Users\Muthuraja\Downloads\modern-cycling-444916-g6-82c207d3eb47.json"  # Provide your Google credentials path


chat_model = GPT2LMHeadModel.from_pretrained('gpt2')
chat_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# Initialize Sentiment Analysis Pipeline using Hugging Face
sentiment_model = pipeline('sentiment-analysis')

# Initialize SentenceTransformer model for semantic search
sentence_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Function to initialize Google Sheets connection
def initialize_google_sheets():
    credentials = Credentials.from_service_account_file(CREDS_PATH, scopes=SCOPE)
    try:
        client = gspread.authorize(credentials)
        sheet = client.open("SalesStores").sheet1  # Change Google Sheet name to "SalesStores"
        return sheet
    except gspread.exceptions.APIError as e:
        st.error(f"Google Sheets API error: {e}")
        return None

sheet = initialize_google_sheets()

# Function to extract text from PDF using PyPDF2
def extract_pdf_text_with_pypdf(pdf_file):
    pdf_text = ""
    with io.BytesIO(pdf_file.read()) as pdf_data:
        pdf_reader = PyPDF2.PdfReader(pdf_data)
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            pdf_text += page.extract_text()
    return pdf_text.strip()

# Function to analyze sentiment using Hugging Face's pre-trained model
def analyze_sentiment(text):
    sentiment = sentiment_model(text)[0]  # Output is a list of dictionaries
    label = sentiment['label']
    score = sentiment['score']
    
    # Define sentiment labels
    if label == "POSITIVE" and score > 0.6:
        sentiment_description = "Positive"
    elif label == "NEGATIVE" and score > 0.6:
        sentiment_description = "Negative"
    else:
        sentiment_description = "Neutral"
    
    return score, sentiment_description

# Function to extract relevant chunks of text based on the question using Sentence Transformers
def extract_relevant_chunks(question, pdf_text, top_n=3):
    pdf_chunks = pdf_text.split('\n')  # Split the PDF text into chunks (each chunk is a line)

    # Encode the question and document chunks using Sentence Transformers
    question_embedding = sentence_model.encode([question])  # Get the embedding (vector) for the user's question
    chunk_embeddings = sentence_model.encode(pdf_chunks)  # Get embeddings (vectors) for each chunk of the PDF
    
    # Compute cosine similarities between the question and each chunk of text
    similarities = np.dot(chunk_embeddings, question_embedding.T).flatten()
    
    # Get the top N relevant chunks based on similarity scores
    relevant_indices = similarities.argsort()[-top_n:][::-1]
    relevant_chunks = [pdf_chunks[i] for i in relevant_indices]
    
    return relevant_chunks

# Function to generate relevant answers using GPT-2 based on the context chunks
def generate_relevant_answer(user_input, relevant_chunks):
    context = " ".join(relevant_chunks)  # Combine the relevant chunks into one long string (context)
    input_text = user_input + " " + context  # Combine the user input (question) with the context
    
    input_ids = chat_tokenizer.encode(input_text, return_tensors='pt')  # Tokenize the input text for GPT-2

    # Generate the answer using GPT-2 with num_return_sequences set to 1
    chat_output = chat_model.generate(
        input_ids, 
        max_new_tokens=150, 
        temperature=0.9, 
        num_return_sequences=1,  # Ensure only one answer is generated
        no_repeat_ngram_size=2
    )
    
    answer = chat_tokenizer.decode(chat_output[0], skip_special_tokens=True)  # Decode the output from GPT-2
    return answer

# Function to update Google Sheets with sentiment, product, and relevant answer only (no timestamp, query or response)
def update_sheet(sentiment_score, sentiment_description, relevant_answer, product_name):
    # Add a timestamp to each entry
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    sheet.append_row([timestamp, sentiment_description, sentiment_score, relevant_answer, product_name])

# Function to convert speech to text using microphone input
def listen_to_speech():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        recognizer.adjust_for_ambient_noise(source)
        st.write("Listening...")
        audio = recognizer.listen(source)
    try:
        text = recognizer.recognize_google(audio)
        return text
    except sr.UnknownValueError:
        return None

# Function to filter data by date range
def filter_data_by_date(data, date_filter):
    if date_filter == 'Today':
        today = datetime.today()
        start_date = today.replace(hour=0, minute=0, second=0, microsecond=0)
        data = data[data['Timestamp'] >= start_date]
    elif date_filter == 'One Week':
        one_week_ago = datetime.today() - timedelta(weeks=1)
        data = data[data['Timestamp'] >= one_week_ago]
    return data

# Dashboard functions
def display_dashboard():
    # Fetch data from Google Sheets
    if sheet:
        data = pd.DataFrame(sheet.get_all_records())

        # Ensure the Timestamp column exists and is in datetime format
        if 'Timestamp' in data.columns:
            data['Timestamp'] = pd.to_datetime(data['Timestamp'])

            # Add a date filter to the dashboard
            date_filter = st.selectbox("Filter by Date", ["All Time", "Today", "One Week"])

            # Filter data based on the selected date range
            if date_filter != "All Time":
                data = filter_data_by_date(data, date_filter)

            # Check if the required columns are present
            if 'Sentiment' in data.columns and 'Answer' in data.columns:
                # Filter by product (Amazon or Flipkart)
                product_filter = st.selectbox("Select Product", ["All", "Amazon", "Flipkart"])

                if product_filter != "All":
                    data = data[data['Product Name'] == product_filter]

                # Plot sentiment distribution
                sentiment_counts = data['Sentiment'].value_counts()

                # Plot Sentiment Distribution
                st.subheader("Sentiment Distribution")
                fig, ax = plt.subplots()
                sentiment_counts.plot(kind='bar', ax=ax)
                ax.set_ylabel("Frequency")
                ax.set_xlabel("Sentiment")
                st.pyplot(fig)

                # Call Statistics
                total_calls = len(data)
                avg_sentiment = data['Sentiment'].apply(lambda x: 1 if x == 'Positive' else -1 if x == 'Negative' else 0).mean()
                avg_sentiment = round(avg_sentiment, 2)

                st.subheader("Call Activity Statistics")
                st.write(f"Total Calls: {total_calls}")
                st.write(f"Average Sentiment: {avg_sentiment}")

                # Call History Table (now no timestamps or queries)
                st.subheader("Call History")
                st.write(data[['Sentiment', 'Answer', 'Product Name']])

                # Download option for the entire history (CSV)
                csv = data.to_csv(index=False)
                st.download_button(
                    label="Download Call History",
                    data=csv,
                    file_name="call_history.csv",
                    mime="text/csv"
                )

            else:
                st.error("The required columns (Sentiment, Answer) are not found in the data.")
                st.write("Check the data structure in the Google Sheet to make sure the columns are correct.")

# Main Streamlit UI and workflow
def main():
    st.title('Real-Time Customer Query Analysis & Call History')

    # Sidebar Navigation
    sidebar_option = st.sidebar.selectbox("Select an Option", ["Dashboard", "Call Analysis"])

    if sidebar_option == "Dashboard":
        display_dashboard()

    elif sidebar_option == "Call Analysis":
        # Upload PDF file
        uploaded_pdf = st.file_uploader("Upload a PDF file", type="pdf")
        if uploaded_pdf:
            pdf_text = extract_pdf_text_with_pypdf(uploaded_pdf)

            # Speech recognition button
            if st.button("Start Speech Recognition"):
                user_input = listen_to_speech()
                if user_input:
                    # Perform sentiment analysis using the new Hugging Face model
                    sentiment_score, sentiment_description = analyze_sentiment(user_input)

                    # Extract relevant chunks based on the question and PDF content
                    relevant_chunks = extract_relevant_chunks(user_input, pdf_text)

                    # Generate a relevant answer using GPT-2 based on the extracted context
                    relevant_answer = generate_relevant_answer(user_input, relevant_chunks)

              
                    st.write(f"Detected Sentiment: {sentiment_description} (Score: {sentiment_score:.2f})")
                    st.write(f"Answer: {relevant_answer[:150]}...")  # Limiting answer length to 150 characters

                    # Allow the user to select the product (Amazon or Flipkart)
                    product_name = st.selectbox("Select Product", ["Amazon", "Flipkart"])

                    # Store the query and the response in Google Sheets
                    if sheet:
                        update_sheet(sentiment_score, sentiment_description, relevant_answer, product_name)

                    st.write("Query and answer saved in Call History!")

if __name__ == "__main__":
    main()
