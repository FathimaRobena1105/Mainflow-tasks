# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. Data Loading
file_path = r'C:\Users\Jabarlal\Documents\python projects\student-mat.csv'
df = pd.read_csv(file_path)

# Display the first few rows of the dataset
df.head()

# 2. Data Exploration
# a. Check for missing values
missing_values = df.isnull().sum()

# b. Display column data types
data_types = df.dtypes

# c. Understand the dataset's size
data_shape = df.shape

# Display results for exploration
print("Missing Values per Column:\n", missing_values)
print("\nColumn Data Types:\n", data_types)
print("\nDataset Shape (Rows, Columns):", data_shape)

# 3. Data Cleaning
# a. Handle missing values (e.g., fill with median for numeric columns)
df = df.fillna(df.median(numeric_only=True))

# b. Remove duplicate entries
df = df.drop_duplicates()

# 4. Data Analysis
# a. Average score in math (G3)
average_score = df['G3'].mean()
print(f"Average Final Grade (G3): {average_score:.2f}")

# b. Number of students who scored above 15 in their final grade (G3)
students_above_15 = df[df['G3'] > 15].shape[0]
print(f"Number of students who scored above 15 in their final grade (G3): {students_above_15}")

# c. Correlation between study time (studytime) and final grade (G3)
correlation = df['studytime'].corr(df['G3'])
print(f"Correlation between study time and final grade: {correlation:.2f}")

# d. Which gender has a higher average final grade (G3)?
gender_avg_grade = df.groupby('sex')['G3'].mean()
print(f"Average Final Grade by Gender:\n{gender_avg_grade}")

# 5. Data Visualization
# a. Plot a histogram of final grades (G3)
plt.figure(figsize=(8, 6))
plt.hist(df['G3'], bins=20, color='skyblue', edgecolor='black')
plt.title('Distribution of Final Grades (G3)')
plt.xlabel('Final Grade (G3)')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

# b. Create a scatter plot between study time and final grade (G3)
plt.figure(figsize=(8, 6))
plt.scatter(df['studytime'], df['G3'], color='purple', alpha=0.6)
plt.title('Scatter Plot: Study Time vs Final Grade (G3)')
plt.xlabel('Study Time (hours per week)')
plt.ylabel('Final Grade (G3)')
plt.grid(True)
plt.show()

# c. Create a bar chart comparing the average scores of male and female students
plt.figure(figsize=(8, 6))
gender_avg_grade.plot(kind='bar', color=['lightblue', 'lightcoral'])
plt.title('Average Final Grade (G3) by Gender')
plt.xlabel('Gender')
plt.ylabel('Average Final Grade (G3)')
plt.xticks(rotation=0)
plt.show()
