from google.colab import files
uploaded = files.upload()  # Upload archive-1.zip
import zipfile
import os

zip_file_path = "archive-1.zip"
extraction_dir = "data"

# Extract zip contents
with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    zip_ref.extractall(extraction_dir)

# Ensure the directory exists
if not os.path.exists(extraction_dir):
    os.makedirs(extraction_dir)
import pandas as pd

csv_path = os.path.join(extraction_dir, "fake_news_dataset.csv")
print(f"Loading dataset from: {csv_path}")

df = pd.read_csv(csv_path)

# Display basic info
print("Columns:", df.columns.tolist())
df.dropna(inplace=True)
df.reset_index(drop=True, inplace=True)
df.head()
# Rename if needed (example)
# df.rename(columns={'news': 'text', 'category': 'label'}, inplace=True)

df = df[['text', 'label']]
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    df['text'], df['label'], test_size=0.2, random_state=42)
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)
from sklearn.linear_model import PassiveAggressiveClassifier

classifier = PassiveAggressiveClassifier(max_iter=50)
classifier.fit(X_train_vec, y_train)
from sklearn.metrics import accuracy_score, confusion_matrix

y_pred = classifier.predict(X_test_vec)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {round(accuracy * 100, 2)}%")
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# Automatically detect unique labels
labels = sorted(df['label'].unique())

# Compute confusion matrix
conf_mat = confusion_matrix(y_test, y_pred, labels=labels)

# Plot confusion matrix
sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues',
            xticklabels=labels, yticklabels=labels)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()
