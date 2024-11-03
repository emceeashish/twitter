Sentiment Analysis Using Deep Learning
Executive Summary
Topic: Sentiment Analysis using Deep Learning
Deep Learning Model: Long Short-Term Memory (LSTM)
Accuracy Achieved: 86%
Operations Performed:
Basic Exploratory Data Analysis (EDA)
Data Preprocessing
Model Creation
Parameter Tuning
Prediction
Introduction
Sentiment Analysis is a widely used application of data science, particularly for analyzing user sentiments on social media platforms. This project uses a Twitter dataset to classify sentiments as positive, negative, or neutral, leveraging deep learning models to achieve meaningful insights from user opinions. The dataset is available on Kaggle.

Project Structure
Data Preprocessing: Cleaning and transforming the Twitter dataset for modeling.
Modeling: Implementation of LSTM for sentiment classification.
Evaluation: Model evaluation to determine accuracy and optimize parameters.
Installation
To run this project, install the following libraries:

bash
Copy code
pip install numpy pandas matplotlib seaborn tensorflow keras streamlit
Running the Application
This project is implemented using Streamlit for an interactive web interface.

Run the Streamlit app with:
bash
Copy code
streamlit run app.py
Open the provided local URL in a browser to interact with the sentiment analysis model.
Usage
Data Preparation: Load and preprocess the dataset.
Model Training: Train the LSTM model on the preprocessed data.
Prediction: Use the trained model to predict sentiments for new data.
Results
The LSTM model achieved an accuracy of 86%, indicating robust performance in sentiment classification.

Acknowledgments
Dataset: Twitter Sentiment Dataset on Kaggle
Image Source: MonkeyLearn
