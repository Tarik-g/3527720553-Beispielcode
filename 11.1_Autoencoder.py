import numpy as np
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input
import matplotlib.pyplot as plt

# Laden Sie den Fashion MNIST-Datensatz
(x_train, _), (x_test, _) = fashion_mnist.load_data()

# Normalisieren Sie die Pixelwerte auf [0, 1]
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.

# Ändern Sie die Form der Bilder in eine eindimensionale Darstellung
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

# Definieren Sie die Schichten des Autoencoders
input_img = Input(shape=(784,))
encoded = Dense(128, activation='relu')(input_img)
encoded = Dense(64, activation='relu')(encoded)
encoded = Dense(32, activation='relu')(encoded)
decoded = Dense(64, activation='relu')(encoded)
decoded = Dense(128, activation='relu')(decoded)
decoded = Dense(784, activation='sigmoid')(decoded)

# Konstruieren und kompilieren Sie den Autoencoder
autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adam', loss='mean_squared_error')

# Trainieren Sie den Autoencoder
autoencoder.fit(x_train, x_train, epochs=50, batch_size=256, shuffle=True, validation_data=(x_test, x_test))

# Verwenden Sie den Autoencoder, um die Testbilder zu rekonstruieren
reconstructed_imgs = autoencoder.predict(x_test)

# Zeichnen Sie die Original- und rekonstruierten Bilder
n = 10  # Anzahl der anzuzeigenden Bilder
plt.figure(figsize=(20, 4))
for i in range(n):
    # Originalbild
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(x_test[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # Rekonstruiertes Bild
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(reconstructed_imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()
