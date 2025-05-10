# Step 1: Install and Import Libraries
!pip install -q nltk
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import string

# NLP Libraries
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
nltk.download('stopwords')
nltk.download('wordnet')

# Modeling Libraries
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Step 2: Upload and Load Dataset from Colab
from google.colab import files
uploaded = files.upload()

import zipfile
import os

# Extract zip
with zipfile.ZipFile("archive-1.zip", 'r') as zip_ref:
    zip_ref.extractall("data")

# Load CSV
df = pd.read_csv("data/fake_news_dataset.csv")

# Step 3: Preprocessing
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\n', '', text)
    text = re.sub(r'\w*\d\w*', '', text)
    return text

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess(text):
    text = clean_text(text)
    return ' '.join([lemmatizer.lemmatize(word) for word in text.split() if word not in stop_words])

# Combine 'title' and 'text' if both exist
if 'title' in df.columns and 'text' in df.columns:
    df['content'] = df['title'].fillna('') + ' ' + df['text'].fillna('')
else:
    df['content'] = df['text']

# Apply preprocessing
df['clean_content'] = df['content'].apply(preprocess)

# Step 4: Vectorization
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df['clean_content']).toarray()
y = df['label'].values

# Step 5: Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 6: Train the model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Step 7: Evaluate the model
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Step 8: Sample Prediction
def predict_fake_news(text):
    processed = preprocess(text)
    vectorized = vectorizer.transform([processed]).toarray()
    prediction = model.predict(vectorized)
    return "Fake" if prediction[0] == 1 else "Real"

# Example
test_news = "Aliens have landed in Ohio and started farming."
print(f"Prediction for: '{test_news}'\nResult: {predict_fake_news(test_news)}")
