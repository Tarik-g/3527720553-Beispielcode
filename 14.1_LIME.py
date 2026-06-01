import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.linear_model import LinearRegression
from sklearn.metrics.pairwise import euclidean_distances

# Daten erstellen
np.random.seed(0)
n_samples = 5000
X = np.random.randint(2, size=(n_samples, 5))  # 5 binäre Merkmale
# Kredit wird vergeben, wenn mindestens 3 Kriterien erfüllt sind und kein Schufa-Eintrag vorhanden ist
y = ((np.sum(X[:, :4], axis=1) >= 3) & (X[:, 4] == 1)).astype(int)

# Neuronales Netzwerk trainieren
model = Sequential()
model.add(Dense(10, input_dim=5, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X, y, epochs=10, verbose=0)

# Instanz auswählen, für die wir die Vorhersage erklären möchten
instance = np.array([[1, 0, 1, 1, 1]])

# Instanz stören und Vorhersagen erhalten
num_perturbations = 1000
perturbed_instances = np.random.randint(2, size=(num_perturbations, 5))
perturbed_predictions = model.predict(perturbed_instances).ravel()

# Gewichte basierend auf der Nähe zur ursprünglichen Instanz berechnen
weights = np.exp(-euclidean_distances(perturbed_instances, instance))

# Interpretierbares Modell (Lineare Regression) trainieren
interpretable_model = LinearRegression()
interpretable_model.fit(perturbed_instances, perturbed_predictions, sample_weight=weights.ravel())

# Die Erklärung sind die Koeffizienten des linearen Modells
explanation = interpretable_model.coef_
print("Erklärung:", explanation)

# Visualisierung
features = ['Wohnort', 'Alter', 'Berufstätigkeit', 'Gehalt', 'kein Schufa-Eintrag']
plt.bar(features, explanation)
plt.xlabel('Merkmale')
plt.ylabel('Wichtigkeit')
plt.title('Erklärung der Kreditvergabe mit LIME')
plt.xticks(fontsize=8)  # Schriftgröße der x-Achsen-Ticks ändern
plt.yticks(fontsize=8)  # Schriftgröße der y-Achsen-Ticks ändern
plt.tight_layout() 
plt.show()
