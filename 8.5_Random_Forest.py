from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
# Daten laden
iris = load_iris () 
X = iris.data
y = iris.target
# Random-Forest-Modell erstellen
clf = RandomForestClassifier(n_estimators=100)
# Modell trainieren
clf.fit(X, y)
# Vorhersage für einen neuen Datenpunkt
sample_data = [[5.1, 3.5, 1.4, 0.2]] 
prediction = clf.predict(sample_data) 
print("Predicted class:", prediction)