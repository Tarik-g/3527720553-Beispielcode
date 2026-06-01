from sklearn import svm
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles

# Generiere nicht lineare Daten
np.random.seed(0)
X, Y = make_circles(n_samples=100, noise=0.05)

# Erstelle SVM-Modell mit RBF-Kernel
model = svm.SVC(kernel='rbf', C=1.0, gamma='scale')

# Trainiere das Modell
model.fit(X, Y)

# Zeichne die Datenpunkte als Kreuze und Kreise
plt.scatter(X[Y==0, 0], X[Y==0, 1], c='blue', marker='x')
plt.scatter(X[Y==1, 0], X[Y==1, 1], c='red', marker='o')

# Zeichne die Entscheidungsgrenze
ax = plt.gca()
xlim = ax.get_xlim()
ylim = ax.get_ylim()

# Erstelle Gitter zum Auswerten des Modells
xx = np.linspace(xlim[0], xlim[1], 30)
yy = np.linspace(ylim[0], ylim[1], 30)
YY, XX = np.meshgrid(yy, xx)
xy = np.vstack([XX.ravel(), YY.ravel()]).T
Z = model.decision_function(xy).reshape(XX.shape)

# Zeichne die Entscheidungsgrenze und die Margen
ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5, linestyles=['--', '-', '--'])

plt.title("Support Vector Machine mit RBF-Kernel")
plt.show()