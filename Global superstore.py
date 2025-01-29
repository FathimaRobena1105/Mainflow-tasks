# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
file_path = r'C:\Users\Jabarlal\Desktop\Global_superstore\superstore.csv'  # Update the file name if required
data = pd.read_csv(file_path)

# Display the first few rows of the dataset to understand its structure
print(data.head())

# Check basic info to find data types and missing values
print(data.info())

# Check for missing values
print(data.isnull().sum())

# Clean the data
# Handle missing values - filling with median for numerical and mode for categorical columns
data['Sales'] = data['Sales'].fillna(data['Sales'].median())
data['Profit'] = data['Profit'].fillna(data['Profit'].median())
data['Region'] = data['Region'].fillna(data['Region'].mode()[0])
data['Product Category'] = data['Product Category'].fillna(data['Product Category'].mode()[0])

# Remove duplicate rows if any
data = data.drop_duplicates()

# Detect and handle outliers using IQR (Interquartile Range) for continuous columns like 'Sales' and 'Profit'
def remove_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

data = remove_outliers(data, 'Sales')
data = remove_outliers(data, 'Profit')

# Statistical Analysis
print("Descriptive Statistics for Sales and Profit:")
print(data[['Sales', 'Profit']].describe())

# Correlation matrix to find relationships between numerical features
corr_matrix = data[['Sales', 'Profit', 'Discount']].corr()
print("Correlation Matrix:")
print(corr_matrix)

# Visualization

# 1. Histograms for numerical features like 'Sales' and 'Profit'
plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
sns.histplot(data['Sales'], kde=True, bins=30, color='blue')
plt.title('Distribution of Sales')

plt.subplot(1, 2, 2)
sns.histplot(data['Profit'], kde=True, bins=30, color='green')
plt.title('Distribution of Profit')

plt.tight_layout()
plt.show()

# 2. Boxplots for detecting outliers in 'Sales' and 'Profit'
plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
sns.boxplot(x=data['Sales'])
plt.title('Sales Boxplot')

plt.subplot(1, 2, 2)
sns.boxplot(x=data['Profit'])
plt.title('Profit Boxplot')

plt.tight_layout()
plt.show()

# 3. Heatmap for correlation between numerical features
plt.figure(figsize=(6, 4))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap')
plt.show()

# 4. Countplot to show sales by Region
plt.figure(figsize=(6, 4))
sns.countplot(x='Region', data=data, hue='Region', palette='viridis')  # Option 1, where hue is used
plt.title('Sales by Region')
plt.show()

# 5. Pie chart to show the sales distribution by Product Category
category_sales = data.groupby('Product Category')['Sales'].sum().sort_values(ascending=False)
plt.figure(figsize=(6, 4))
category_sales.plot.pie(autopct='%1.1f%%', startangle=90, cmap='Set3')
plt.title('Sales Distribution by Product Category')
plt.ylabel('')
plt.show()
