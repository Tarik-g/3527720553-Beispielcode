import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# Erzeugen von Daten
np.random.seed(0)
X = np.sort(5 * np.random.rand(80, 1), axis=0)
y = np.sin(X).ravel() + np.random.randn(80) * 0.1

# Umwandlung der Daten für ein Polynom zweiten Grades
polynomial_features = PolynomialFeatures(degree=2)
X_poly = polynomial_features.fit_transform(X)

# Polynomiale Regression
regressor = LinearRegression()
regressor.fit(X_poly, y)

# Vorhersagen und Plotten
y_pred = regressor.predict(X_poly)
plt.scatter(X, y, color='blue')
plt.plot(X, y_pred, color='red')
plt.title("Polynomiale Regression")
plt.xlabel("X")
plt.ylabel("y")
plt.show()