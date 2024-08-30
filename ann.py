# lung_tumor_clustering.py

import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Step 1: Load the Data
data = pd.read_csv('annotations.csv')

# Step 2: Data Cleaning
# Check for missing values
if data.isnull().sum().any():
    raise ValueError("Missing values found in the data. Please clean the data.")

# Convert columns to appropriate data types if needed
data['coordX'] = data['coordX'].astype(float)
data['coordY'] = data['coordY'].astype(float)
data['coordZ'] = data['coordZ'].astype(float)
data['diameter_mm'] = data['diameter_mm'].astype(float)

# Step 3: Data Exploration
# Optional: Visualize the data points
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(data['coordX'], data['coordY'], data['coordZ'], c=data['diameter_mm'], cmap='viridis')
legend1 = ax.legend(*scatter.legend_elements(), title="Diameter_mm")
ax.add_artist(legend1)
plt.title("3D Scatter Plot of Tumor Coordinates")
plt.xlabel("coordX")
plt.ylabel("coordY")
ax.set_zlabel("coordZ")
plt.show()

# Step 4: Feature Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(data[['coordX', 'coordY', 'coordZ', 'diameter_mm']])

# Step 5: Clustering (if applicable)
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
kmeans.fit(X_scaled)
data['cluster'] = kmeans.labels_

# Step 6: Visualize Clusters
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(data['coordX'], data['coordY'], data['coordZ'], c=data['cluster'], cmap='viridis')
legend1 = ax.legend(*scatter.legend_elements(), title="Clusters")
ax.add_artist(legend1)
plt.title("3D Scatter Plot of Tumor Clusters")
plt.xlabel("coordX")
plt.ylabel("coordY")
ax.set_zlabel("coordZ")
plt.show()

# Optional: Save the results
data.to_csv('clustered_tumors.csv', index=False)
print("Clustered data saved to 'clustered_tumors.csv'")
