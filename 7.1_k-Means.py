import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

# Erzeuge 300 Datenpunkten und 4 Clustern
data, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)

# Wende K-Means Clustering an
kmeans = KMeans(n_clusters=4)
kmeans.fit(data)
predicted_labels = kmeans.predict(data)

# Zeige die Cluster und Zentroide
plt.scatter(data[:, 0], data[:, 1], c=predicted_labels, s=50, cmap='viridis')
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, alpha=0.75, marker='X')
plt.title("k-Means")
plt.show()