import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Download NLTK resources (first time only)
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Read the dataset (ensure the path to the CSV file is correct)
df = pd.read_csv(r'C:\Users\Jabarlal\Desktop\reviews.csv')  # Update this with your file's location

# Inspect the first few rows of the dataset
print(df.head())

# Initialize NLTK resources
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Preprocessing function: remove stopwords, punctuation, lowercase, tokenize, and lemmatize
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    
    # Tokenize the text
    tokens = word_tokenize(text)
    
    # Remove stopwords and punctuation
    tokens = [word for word in tokens if word not in stop_words and word not in string.punctuation]
    
    # Lemmatize the tokens
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    
    # Join tokens back into a single string
    return ' '.join(tokens)

# Apply preprocessing to the review text
df['Cleaned Review'] = df['Review Text'].apply(preprocess_text)

# Vectorize the text using TF-IDF
vectorizer = TfidfVectorizer(max_features=5000)  # Limit to 5000 features (words)
X = vectorizer.fit_transform(df['Cleaned Review'])

# Define target variable (Sentiment: 1 for positive, 0 for negative)
df['Sentiment'] = df['Sentiment'].apply(lambda x: 1 if x == 'positive' else 0)

y = df['Sentiment']

# Split the data into training and testing sets (80-20 split)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the Logistic Regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model using accuracy, precision, recall, F1-score
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# Display evaluation metrics
print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1-Score: {f1}")

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print(f"Confusion Matrix:\n{conf_matrix}")

# Visualize the confusion matrix
plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Example of a review classification
sample_review = "I love this product! It works great and exceeds expectations."
processed_review = preprocess_text(sample_review)
sample_vector = vectorizer.transform([processed_review])
sample_pred = model.predict(sample_vector)

print(f"Sample Review: {sample_review}")
print(f"Predicted Sentiment: {'Positive' if sample_pred == 1 else 'Negative'}")
