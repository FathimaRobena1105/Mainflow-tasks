#Sales Data#

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error

# Read the dataset
df = pd.read_csv('C:/Users/Jabarlal/Desktop/sales_data.csv')

# Convert the 'Date' column to datetime
df['Date'] = pd.to_datetime(df['Date'])

# Set the 'Date' column as the index
df.set_index('Date', inplace=True)

# Set the frequency of the date index (assuming daily frequency here)
df = df.asfreq('D')  # You can change 'D' to 'M' for monthly, etc.

# Inspect the first few rows of the dataset
print(df.head())

# Plot the sales data
plt.figure(figsize=(10,6))
plt.plot(df.index, df['Sales'], label='Sales')
plt.title('Sales Over Time')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.legend()
plt.show()

# ARIMA Model - Train and Forecast
model = ARIMA(df['Sales'], order=(5,1,0))  # Example ARIMA model (p,d,q)
model_fit = model.fit()

# Forecast the next 10 periods (for example)
forecast_steps = 10
forecast = model_fit.forecast(steps=forecast_steps)

# Plotting the forecast
plt.figure(figsize=(10,6))
plt.plot(df.index, df['Sales'], label='Historical Sales')
plt.plot(pd.date_range(df.index[-1], periods=forecast_steps+1, freq='D')[1:], forecast, color='red', label='Forecasted Sales')
plt.title('Sales Forecast')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.legend()
plt.show()

# Print forecasted values
print("Forecasted Sales:", forecast)

#Project Description: Predicting Heart Disease Using Logistic Regression#

# Import required libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv(r'C:\Users\Jabarlal\Desktop\heart_disease.csv')

# Data Preprocessing
# Handle categorical features (Gender column)
df['Gender'] = df['Gender'].map({'Male': 0, 'Female': 1})

# Features and target variable
X = df[['Age', 'Gender', 'Cholesterol', 'Blood Pressure']]  # Independent variables
y = df['Heart Disease']  # Dependent variable (target)

# Normalize numerical features using StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the dataset into training and testing sets (80-20 split)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train the Logistic Regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Model Evaluation
# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)

# Classification Report
class_report = classification_report(y_test, y_pred)

# Accuracy Score
accuracy = accuracy_score(y_test, y_pred)

# Display the results
print("Confusion Matrix:")
print(conf_matrix)
print("\nClassification Report:")
print(class_report)
print(f"\nAccuracy: {accuracy:.2f}")

# Plot Confusion Matrix
plt.figure(figsize=(6, 5))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False, xticklabels=['No Disease', 'Disease'], yticklabels=['No Disease', 'Disease'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

