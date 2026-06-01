import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# Erstelle einige zweidimensionale Daten
np.random.seed(0)
X = np.r_[np.random.randn(20, 2) - [2, 2], np.random.randn(20, 2) + [2, 2]]
Y = [-1] * 20 + [1] * 20

# Trainiere eine lineare SVM
n_samples = len(X)
# P ist eine NxN Matrix mit xi.xj Produkt
P = np.outer(Y,Y) * np.dot(X,X.T)

def objective(a):
    return 0.5 * np.dot(a, np.dot(a, P)) - np.sum(a)

def constraint(a):
    return np.dot(a, Y)

a0 = np.zeros(n_samples)
bounds = [(0, None) for _ in range(n_samples)]
constraints = {'type': 'eq', 'fun': constraint}

solution = minimize(objective, a0, bounds=bounds, constraints=constraints)

# Lagrange multiplikatoren
a = np.ravel(solution.x)

# Support Vektoren haben nicht null lagrange multiplikatoren
sv = a > 1e-5
ind = np.arange(len(a))[sv]
indices = np.where(sv)[0]
a = a[indices]
X_sv = X[indices]
Y_sv = np.array(Y)[indices]

# Berechne den Bias
b = 0
for n in range(len(a)):
    b += Y_sv[n]
    b -= np.sum(a * Y_sv * np.dot(X_sv, X_sv[n]))
b /= len(a)

# Berechne das Gewichtsvektor
w = np.zeros(2)
for n in range(len(a)):
    w += a[n] * Y_sv[n] * X_sv[n]

# Zeichne die Datenpunkte und die SVM Grenzlinie
plt.figure(figsize=(10, 8))

# Zeichne die Datenpunkte als Kreuze und Kreise
for (x1, x2), y in zip(X, Y):
    if y == -1:
        plt.scatter(x1, x2, c='b', marker='x')  # Kreuze für Klasse -1
    else:
        plt.scatter(x1, x2, c='r', marker='o')  # Kreise für Klasse 1

# Zeichne die SVM Grenzlinie
ax = plt.gca()
xlim = ax.get_xlim()
ylim = ax.get_ylim()

xx = np.linspace(xlim[0], xlim[1], 30)
yy = np.linspace(ylim[0], ylim[1], 30)
YY, XX = np.meshgrid(yy, xx)
xy = np.vstack([XX.ravel(), YY.ravel()]).T
Z = np.dot(xy, w) + b
Z = Z.reshape(XX.shape)
ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5, linestyles=['--', '-', '--'])
plt.title("Support Vector Machine")
plt.show()
