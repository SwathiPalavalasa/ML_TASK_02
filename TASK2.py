import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Load dataset
data = pd.read_csv("C:/Users/user/OneDrive/Desktop/mall_customers.csv")

# Select features
X = data[["Annual Income (k$)", "Spending Score (1-100)"]]

# KMeans model
kmeans = KMeans(n_clusters=5, random_state=42)
data["Cluster"] = kmeans.fit_predict(X)

# Plot clusters
plt.scatter(X["Annual Income (k$)"], X["Spending Score (1-100)"], 
            c=data["Cluster"], cmap="rainbow", s=50)
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], 
            c="black", marker="X", s=200, label="Centroids")
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.title("K-Means Customer Segmentation")
plt.legend()
plt.show()
