import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.datasets import make_moons

# Erzeuge einen Beispieldatensatz
data, _ = make_moons(n_samples=300, noise=0.05, random_state=0)

# Anwendung von DBSCAN
dbscan = DBSCAN(eps=0.3, min_samples=5)
predicted_labels = dbscan.fit_predict(data)

# Visualisierung
plt.scatter(data[:, 0], data[:, 1], c=predicted_labels, cmap='viridis')
plt.title('DBSCAN-Clustering')
plt.show()