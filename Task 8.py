#Section 1 - Feature Engineering

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

# Set a seed for reproducibility
np.random.seed(42)

# Generating random student scores for math, english, and science (between 50 and 100)
math_scores = np.random.randint(50, 101, size=1000)
english_scores = np.random.randint(50, 101, size=1000)
science_scores = np.random.randint(50, 101, size=1000)

# Generating random pass/fail status (0 or 1) based on total score
total_scores = math_scores + english_scores + science_scores
passed = (total_scores >= 250).astype(int)  # If total score >= 250, student passes (1), otherwise fails (0)

# Creating a DataFrame
df = pd.DataFrame({
    'math_score': math_scores,
    'english_score': english_scores,
    'science_score': science_scores,
    'passed': passed
})

# Feature Engineering: Creating a new feature - total_score
df['total_score'] = df['math_score'] + df['english_score'] + df['science_score']

# Define features (X) and target variable (y)
X = df[['math_score', 'english_score', 'science_score', 'total_score']]
y = df['passed']

# Splitting the data into training and testing sets (70% training, 30% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Scaling the features to standardize the data for better performance
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initializing the RandomForestClassifier model
model = RandomForestClassifier(random_state=42)

# Hyperparameter tuning using GridSearchCV
param_grid = {
    'n_estimators': [50, 100, 150],  # Number of trees in the forest
    'max_depth': [None, 10, 20, 30],  # Max depth of each tree
    'min_samples_split': [2, 5, 10],  # Min samples to split an internal node
    'min_samples_leaf': [1, 2, 4]  # Min samples required at a leaf node
}

# Performing GridSearchCV to find the best parameters
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
grid_search.fit(X_train_scaled, y_train)

# Output the best parameters found from GridSearchCV
print("Best Parameters from GridSearchCV:", grid_search.best_params_)

# Train the model with the best found parameters
best_model = grid_search.best_estimator_

# Predicting on the test set
y_pred = best_model.predict(X_test_scaled)

# Evaluating the model performance
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy of the model after tuning: {accuracy}")

--------------------------------------------------------------------------------

#Section 2 - Fraud Detection

# Importing necessary libraries
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV

# Load the dataset from the specified location
df = pd.read_csv(r'C:\Users\Jabarlal\Desktop\fraud_detection.csv')

# Inspect for missing values
print("Missing values:", df.isnull().sum())

# Label Encoding for categorical variables (Type column)
le = LabelEncoder()
df['Type'] = le.fit_transform(df['Type'])

# Feature Engineering: Create a new feature 'Amount Range'
def categorize_amount(amount):
    if amount < 100000:  # ₹100,000 (in thousands: ₹100)
        return 'Low'
    elif amount < 1000000:  # ₹1,000,000 (in thousands: ₹1000)
        return 'Medium'
    else:
        return 'High'

df['Amount Range'] = df['Amount (INR)'].apply(categorize_amount)

# Convert 'Amount Range' into numeric categories using Label Encoding
df['Amount Range'] = le.fit_transform(df['Amount Range'])

# Split the dataset into features and target variable
X = df.drop(columns=['Transaction ID', 'Is Fraud'])  # Features (excluding Transaction ID)
y = df['Is Fraud']  # Target variable

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train a Decision Tree Classifier
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)

# Make predictions
y_pred = clf.predict(X_test)

# Evaluate model performance with zero_division=1 to avoid ill-defined precision/recall
print("Model Performance Evaluation (Before Hyperparameter Tuning):")
print(classification_report(y_test, y_pred, zero_division=1))

# Hyperparameter Tuning with GridSearchCV
param_grid = {
    'max_depth': [3, 5, 10, 15, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Initialize GridSearchCV
grid_search = GridSearchCV(estimator=clf, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1)

# Fit the grid search
grid_search.fit(X_train, y_train)

# Print the best parameters from the grid search
print("Best parameters found through GridSearchCV: ", grid_search.best_params_)

# Evaluate the optimized model
best_clf = grid_search.best_estimator_
y_pred_optimized = best_clf.predict(X_test)

# Evaluate the optimized model's performance with zero_division=1 to avoid ill-defined precision/recall
print("Model Performance Evaluation (After Hyperparameter Tuning):")
print(classification_report(y_test, y_pred_optimized, zero_division=1))

