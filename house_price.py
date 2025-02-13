# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
file_path = "C:/Users/Jabarlal/Desktop/house_prices.csv"
df = pd.read_csv(file_path)

# Step 1: Inspect the Dataset
# Check for missing values
print("Missing values:\n", df.isnull().sum())

# Step 2: Analyze distributions of numerical variables
# Plot distribution for Size and Price
sns.histplot(df['Size (sq ft)'], kde=True)
plt.title('Distribution of Size')
plt.show()

sns.histplot(df['Price ($)'], kde=True)
plt.title('Distribution of Price')
plt.show()

# Step 3: Identify potential outliers using boxplot
sns.boxplot(x=df['Size (sq ft)'])
plt.title('Boxplot for Size')
plt.show()

sns.boxplot(x=df['Price ($)'])
plt.title('Boxplot for Price')
plt.show()

# Step 4: Data Preprocessing
# Encode categorical 'Location' using One-Hot Encoding
df = pd.get_dummies(df, columns=['Location'], drop_first=True)

# Normalize numerical data using Standardization
scaler = StandardScaler()
df[['Size (sq ft)', 'Number of Rooms']] = scaler.fit_transform(df[['Size (sq ft)', 'Number of Rooms']])

# Step 5: Feature Selection
# Calculate correlations
corr_matrix = df.corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

# Step 6: Train-Test Split
X = df.drop('Price ($)', axis=1)  # Features
y = df['Price ($)']  # Target

# Split the data into 80% training and 20% testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 7: Train a Linear Regression Model
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Step 8: Model Evaluation
# Predicting the house prices on the test set
y_pred = regressor.predict(X_test)

# Calculate RMSE
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"Root Mean Squared Error (RMSE): {rmse}")

# Calculate R² (Coefficient of Determination)
r2 = r2_score(y_test, y_pred)
print(f"R²: {r2}")

# Visualizing the predicted vs actual prices
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted Prices")
plt.show()

# Feature Importance: Coefficients of the model
feature_importance = pd.DataFrame(regressor.coef_, X.columns, columns=["Coefficient"])
print("\nFeature Importance (Model Coefficients):")
print(feature_importance)
