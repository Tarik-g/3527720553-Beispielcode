import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

# Anzahl Klassen
num_classes = 10
# Eingabedaten 28x28 Pixel x 1 Grauwert
input_shape = (28, 28, 1)
# lade Trainings und Testdatensets
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
# skaliere auf Werte zwischen 0 und 1
x_train = x_train / 255
x_test = x_test / 255

# Die shape anpassen auf (28, 28, 1)
x_train = x_train.reshape(60000, 28, 28, 1)
x_test = x_test.reshape(10000,28,28,1)

# die Nummer der Klassen muss in binäre 
# one-hot-Vektoren umgewandelt werden
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# definiere das Netzwerk
model = keras.Sequential([
  keras.Input(shape=input_shape),
  layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
  layers.MaxPooling2D(pool_size=(2, 2)),
  layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
  layers.MaxPooling2D(pool_size=(2, 2)),
  layers.Flatten(),
  layers.Dropout(0.5),
  layers.Dense(num_classes, activation="softmax"),
])
model.summary()
# Modell compilieren...
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
# ... und trainieren ...
history = model.fit(x_train, y_train, batch_size=128, epochs=15, validation_split=0.1)
# Am Ende auswerten, accuracy plotten...
plt.plot(history.history['accuracy'], label='Trainingsgenauigkeit')
plt.plot(history.history['val_accuracy'], linestyle='dashed', label='Validierungsgenauigkeit')
plt.title('Genauigkeit des Modells')
plt.ylabel('Genauigkeit')
plt.xlabel('Epoche')
plt.legend(loc='center right')
plt.savefig("cnn_accuracy.svg")
plt.show()

# ... und Loss plotten
plt.plot(history.history['loss'], label='Trainingsverlust')
plt.plot(history.history['val_loss'], linestyle='dashed', label='Validierungsverlust')
plt.title('Verlust des Modells')
plt.ylabel('Verlust')
plt.xlabel('Epoche')
plt.legend(loc='center right')
plt.show()