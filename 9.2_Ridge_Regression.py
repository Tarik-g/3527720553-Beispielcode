import numpy as np
from sklearn.linear_model import Ridge
from sklearn.datasets import make_regression

# Erzeugen von Daten
X, y = make_regression(n_samples=200, n_features=10, noise=0.5, random_state=42)

# Ridge-Regression mit Regularisierungsparameter von 1.0
ridge = Ridge(alpha=1.0)
ridge.fit(X, y)

# Ausgabe der Koeffizienten
print(ridge.coef_)