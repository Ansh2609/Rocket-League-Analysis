import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
import seaborn as sns
from sklearn.metrics import silhouette_score


# Load the dataset with space-separated values
df = pd.read_csv('rocket_league_skillshots.data', delim_whitespace=True)

# Display the first few rows to inspect the structure
print(df.head())

# Check column names, data types, and if there are missing values
print(df.info())
print(df.describe())

# Fill missing numerical values with the column median
df.fillna(df.median(), inplace=True)

# Select relevant numerical columns
numerical_cols = ['BallAcceleration', 'Time', 'DistanceWall', 'DistanceCeil', 
                  'DistanceBall', 'PlayerSpeed', 'BallSpeed']

scaler = StandardScaler()
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

distortions = []
for k in range(1, 11):  # Test k values from 1 to 10
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(df[numerical_cols])
    distortions.append(kmeans.inertia_)

# Plot the Elbow Curve
plt.plot(range(1, 11), distortions, marker='o')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Distortion')
plt.title('Elbow Method')
plt.show()

kmeans = KMeans(n_clusters=3, random_state=42)  # Replace 3 with the optimal k
df['Cluster'] = kmeans.fit_predict(df[numerical_cols])


dbscan = DBSCAN(eps=0.5, min_samples=5)  # Adjust parameters for your dataset
df['Cluster_DBSCAN'] = dbscan.fit_predict(df[numerical_cols])

sns.scatterplot(x='BallAcceleration', y='PlayerSpeed', hue='Cluster_DBSCAN', data=df)
plt.title('DBSCAN Clustering')
plt.show()

# Calculate Silhouette Score for K-Means
silhouette_kmeans = silhouette_score(df[numerical_cols], df['Cluster'])
print(f"Silhouette Score for K-Means: {silhouette_kmeans}")

# Calculate Silhouette Score for DBSCAN
silhouette_dbscan = silhouette_score(df[numerical_cols], df['Cluster_DBSCAN'])
print(f"Silhouette Score for DBSCAN: {silhouette_dbscan}")

sns.scatterplot(x='BallAcceleration', y='PlayerSpeed', hue='Cluster', data=df, palette='Set1')
plt.title('K-Means Clustering')
plt.show()

cluster_means = df.groupby('Cluster').mean()
sns.heatmap(cluster_means, annot=True, cmap='coolwarm')
plt.title('Cluster Feature Averages')
plt.show()