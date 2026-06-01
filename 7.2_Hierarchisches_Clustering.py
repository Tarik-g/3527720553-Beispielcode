import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.datasets import make_blobs

# Erzeuge einen Beispieldatensatz
data, _ = make_blobs(n_samples=100, centers=3, cluster_std=0.60, random_state=0)

# Anwendung des hierarchischen Clustering
linked = linkage(data, 'single')

# Zeichnen des Dendrogramms
plt.figure(figsize=(10, 7))
dendrogram(linked)
plt.title('Hierarchisches Clustering - Dendrogramm')
plt.xlabel('Datenpunkte')
plt.ylabel('Euklidische Entfernungen')
plt.show()