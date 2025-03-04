# Part 1: Dimensionality Reduction using PCA

# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

# Load the Iris dataset
iris = load_iris()
X = iris.data  # The features
y = iris.target  # The labels

# Apply PCA to reduce dimensions to 2D
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Create a DataFrame to hold the reduced data for visualization
df_pca = pd.DataFrame(X_pca, columns=['PCA1', 'PCA2'])

# Scatter plot of the reduced dimensions
plt.figure(figsize=(8,6))
plt.scatter(df_pca['PCA1'], df_pca['PCA2'], c=y, cmap='viridis')
plt.title("PCA - 2D Visualization of Iris Dataset")
plt.xlabel('PCA1')
plt.ylabel('PCA2')
plt.colorbar(label='Species')
plt.show()

# Deliverables:
# 1. Reduced dataset (2D representation) is in df_pca.
# 2. Scatter plot visualizing the reduced dimensions.
# Part 2: Stock Price Prediction using ARIMA

# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error
from math import sqrt

# Load the stock data
stock_data = pd.read_csv('stock_prices.csv', parse_dates=['Date'])
stock_data['Date'] = pd.to_datetime(stock_data['Date'])

# Preprocessing
stock_data.set_index('Date', inplace=True)
stock_data = stock_data.sort_index()

# Check for missing values and handle them (if any)
stock_data = stock_data.ffill()  # Forward fill missing values

# Make sure the time series has a frequency (e.g., business day frequency 'B')
stock_data = stock_data.asfreq('B')

# Exploratory Data Analysis (EDA)
plt.figure(figsize=(10, 6))
plt.plot(stock_data['Close'], label='Close Price')
plt.title('Stock Closing Prices Over Time')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.legend()
plt.show()

# Feature Engineering: Create lag features
stock_data['Lag_1'] = stock_data['Close'].shift(1)
stock_data.dropna(inplace=True)

# Train-Test Split (80% for training, 20% for testing)
train_size = int(len(stock_data) * 0.8)
train, test = stock_data[:train_size], stock_data[train_size:].copy()  # Ensure 'test' is a copy, not a view

# ARIMA Model Training
# For simplicity, using default parameters (p=1, d=1, q=1) for ARIMA model.
model = ARIMA(train['Close'], order=(1,1,1))
model_fit = model.fit()

# Forecasting
forecast = model_fit.forecast(steps=len(test))

# Ensure we are modifying the copy of 'test' safely
test.loc[:, 'Predicted_Close'] = forecast  # Safe assignment to a copy of 'test'

# Plot forecast vs actual stock prices
plt.figure(figsize=(10, 6))
plt.plot(test['Close'], label='Actual Close Price')
plt.plot(test['Predicted_Close'], label='Predicted Close Price', linestyle='--')
plt.title('Stock Price Prediction: Actual vs Predicted')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.legend()
plt.show()

# Evaluation Metrics
mae = mean_absolute_error(test['Close'], test['Predicted_Close'])
rmse = sqrt(mean_squared_error(test['Close'], test['Predicted_Close']))
mape = np.mean(np.abs((test['Close'] - test['Predicted_Close']) / test['Close'])) * 100

print(f"MAE: {mae}")
print(f"RMSE: {rmse}")
print(f"MAPE: {mape}%")

# Deliverables:
# 1. Trained ARIMA model for stock forecasting.
# 2. Time-series plots comparing predictions vs. actual prices.
# 3. Insights on stock trends, seasonality, and forecast accuracy.
