import re
import nltk
import numpy as np
import pandas as pd
import tensorflow as tf
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from tensorflow.keras.models import Sequential
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import one_hot
from sklearn.metrics import confusion_matrix, accuracy_score
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout

nltk.download('stopwords')
from google.colab import files
uploaded = files.upload()
import zipfile
import io

# The key should match the actual uploaded filename, which is 'archive.zip'
with zipfile.ZipFile(io.BytesIO(uploaded['archive.zip']), 'r') as zip_ref:
    zip_ref.extractall()
import pandas as pd
df = pd.read_csv('train.csv')
df.head()
X = df['title']
y = df['label']
ps = PorterStemmer()
corpus = []

for i in range(len(X)):
    # Convert X[i] to string explicitly to handle potential non-string values
    text = re.sub('[^a-zA-Z]', ' ', str(X[i]))
    text = text.lower().split()
    text = [ps.stem(word) for word in text if word not in stopwords.words('english')]
    corpus.append(' '.join(text))
vocab_size = 5000
sent_len = 20

one_hot_encoded = [one_hot(text, vocab_size) for text in corpus]
padded_docs = pad_sequences(one_hot_encoded, maxlen=sent_len, padding='post')
X_final = np.array(padded_docs)
y_final = np.array(y)

X_train, X_test, y_train, y_test = train_test_split(X_final, y_final, test_size=0.33, random_state=42)
embedding_dim = 40

model = Sequential()
model.add(Embedding(vocab_size, embedding_dim, input_length=sent_len))
model.add(Dropout(0.5))
model.add(LSTM(100))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))  # Add activation for binary classification

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()
model.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size=64, epochs=5)
y_pred = (model.predict(X_test) > 0.5).astype("int32")
print(confusion_matrix(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))
