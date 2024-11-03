import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import re
import string
import os

# Set the environment variable within Python
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Load your saved model
import tensorflow as tf
model = tf.keras.models.load_model(r'C:\Users\ashis\model.h5')



# Load tokenizer
with open('C:/Users/ashis/tok.pkl', 'rb') as handle:

    tokenizer = pickle.load(handle)

# Preprocessing functions
def remove_URL(text):
    url = re.compile(r'https?://\S+|www\.\S+')
    return url.sub(r'', text)

def remove_emoji(text):
    emoji_pattern = re.compile(
        '['
        u'\U0001F600-\U0001F64F'  # emoticons
        u'\U0001F300-\U0001F5FF'  # symbols & pictographs
        u'\U0001F680-\U0001F6FF'  # transport & map symbols
        u'\U0001F1E0-\U0001F1FF'  # flags
        u'\U00002702-\U000027B0'
        u'\U000024C2-\U0001F251'
        ']+', flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)

def remove_html(text):
    html = re.compile(r'<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});')
    return re.sub(html, '', text)

def remove_punct(text):
    table = str.maketrans('', '', string.punctuation)
    return text.translate(table)

def preprocess_text(text):
    text = remove_URL(text)
    text = remove_emoji(text)
    text = remove_html(text)
    text = remove_punct(text)
    # Convert the text to sequences
    sequences = tokenizer.texts_to_sequences([text])
    # Pad sequences to ensure consistent length
    padded = pad_sequences(sequences, maxlen=100)  # Adjust maxlen as per model training
    return padded

# Function to predict sentiment
def predict_sentiment(text):
    # Preprocess the text
    processed_text = preprocess_text(text)
    # Get the model's prediction
    prediction = model.predict(processed_text)
    # Assume threshold of 0.5 for binary classification
    sentiment = "Positive" if prediction >= 0.5 else "Negative"
    return sentiment

# Streamlit app
st.title("Sentiment Analysis with LSTM Model")
st.write("Enter text to predict its sentiment.")

# Text input
input_text = st.text_area("Text Input", "Type your text here...")

# Predict button
if st.button("Predict"):
    # Ensure text is not empty
    if input_text.strip():
        # Predict sentiment
        sentiment = predict_sentiment(input_text)
        st.write(f"Predicted Sentiment: {sentiment}")
    else:
        st.write("Please enter some text.")
