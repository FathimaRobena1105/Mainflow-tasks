import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

# Step 1: Load the Dataset
file_path = r'C:\Users\Jabarlal\Desktop\customer_data.csv'
data = pd.read_csv(file_path)

# Step 2: Inspect the dataset
print(data.head())  # Checking the first few rows
print(data.info())  # Checking data types and null values
print(data.describe())  # Summary statistics to understand data range

# Step 3: Handle missing values (if any)
data = data.dropna()  # Drop rows with missing values
# Alternatively, you can use: data.fillna(method='ffill', inplace=True)

# Step 4: Drop the 'Customer ID' column as it's not useful for clustering
data = data.drop(columns=['Customer ID'], errors='ignore')

# Step 5: Standardize the data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data)

# Step 6: Determine the optimal number of clusters using the Elbow Method
wcss = []  # Within-cluster sum of squares
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=42)
    kmeans.fit(scaled_data)
    wcss.append(kmeans.inertia_)

# Plot the Elbow Method graph
plt.figure(figsize=(8, 6))
plt.plot(range(1, 11), wcss, marker='o')
plt.title('Elbow Method For Optimal Number of Clusters')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS (Within-Cluster Sum of Squares)')
plt.show()

# Based on the elbow method, choose the optimal number of clusters (say 5)
optimal_clusters = 5

# Step 7: Apply K-Means Clustering
kmeans = KMeans(n_clusters=optimal_clusters, init='k-means++', max_iter=300, n_init=10, random_state=42)
data['Cluster'] = kmeans.fit_predict(scaled_data)

# Step 8: Evaluate clustering using Silhouette Score
sil_score = silhouette_score(scaled_data, data['Cluster'])
print(f'Silhouette Score: {sil_score:.3f}')

# Step 9: Dimensionality Reduction using PCA for visualization
pca = PCA(n_components=2)
pca_components = pca.fit_transform(scaled_data)
data_pca = pd.DataFrame(data=pca_components, columns=['PCA1', 'PCA2'])

# Add the Cluster column from the original data to the PCA DataFrame
data_pca['Cluster'] = data['Cluster']

# Step 10: Visualization - 2D Scatter Plot of Clusters
plt.figure(figsize=(8, 6))
sns.scatterplot(x='PCA1', y='PCA2', hue='Cluster', data=data_pca, palette='viridis', s=100, alpha=0.7)
plt.title('Customer Segments in 2D (PCA)')
plt.show()

# Step 11: Additional Visualization - Pair Plot of Features Colored by Cluster
sns.pairplot(data, hue='Cluster', palette='viridis')
plt.show()

# Step 12: Centroids Visualization - Plot the centroids in PCA space
centroids = pca.transform(kmeans.cluster_centers_)
plt.figure(figsize=(8, 6))
sns.scatterplot(x='PCA1', y='PCA2', hue='Cluster', data=data_pca, palette='viridis', s=100, alpha=0.7)
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=300, marker='X', label='Centroids')
plt.legend()
plt.title('Customer Segments with Centroids')
plt.show()

# Step 13: Save the clustered dataset to a new CSV file
data.to_csv(r'C:\Users\Jabarlal\Desktop\customer_data_clustered.csv', index=False)

# Insights and Recommendations based on clustering
print("\nInsights and Recommendations:")
for i in range(optimal_clusters):
    cluster_data = data[data['Cluster'] == i]
    avg_age = cluster_data['Age'].mean()
    avg_income = cluster_data['Annual_Income_(k$)'].mean()
    avg_spending = cluster_data['Spending_Score'].mean()

    print(f"\nCluster {i}:")
    print(f"  - Average Age: {avg_age:.2f}")
    print(f"  - Average Annual Income: ${avg_income:.2f}")
    print(f"  - Average Spending Score: {avg_spending:.2f}")

# Based on the above data, you can create targeted strategies:
# Example Recommendations:
# 1. High-income customers with high spending: Target for premium offers and loyalty programs.
# 2. Low-income customers with high spending: Offer discounts or tailored deals to retain them.
# 3. Low spending customers with low income: Offer engagement programs or education about the product.

