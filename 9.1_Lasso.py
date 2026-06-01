import numpy as np
from sklearn.linear_model import Lasso
from sklearn.datasets import make_regression
# Erzeugen von Daten
X, y = make_regression(n_samples=200, n_features=10, noise =0.5, random_state=42)
# Lasso Regression mit einem Regularisierungsparameter von 1.0
lasso = Lasso(alpha=1.0) 
lasso.fit(X, y)
# Ausgabe der Koeffizienten
print(lasso.coef_)