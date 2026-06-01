import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras

# Simulierter Datensatz
N = 1000
X = np.random.randn(N, 2) # zufällige Punkte
# Zielfunktion (Target) definieren:
# Sind die Punkte im Einheitskreis, dann y=1 sonst y=0
y = (X[:, 0]**2 + X[:, 1]**2 < 1.0).astype(int)

# Aufteilen des Datensatzes in Trainings- und Testdaten
X_train, X_test = X[:800], X[800:]
y_train, y_test = y[:800], y[800:]

# Modells mit 2 Schichten: erst 10 Neuronen, dann 1 Neuron
model = keras.Sequential([
    keras.layers.Dense(10, input_shape=(2,), activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])

# Kompilieren des Modells (binary_crossentropy wird zur Klassifizierung genommen)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Training des Modells
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))
# Fehler plotten, wie er kleiner wird
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'],"--")
plt.title('Modellfehler')
plt.ylabel('Fehler')
plt.xlabel('Epoche')
plt.legend(['Trainingsdaten', 'Testdaten'], loc='center right')
plt.show()