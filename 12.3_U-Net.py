import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
from keras.layers import Add

(train_data, _), (test_data, _) = fashion_mnist.load_data()

# Daten normalisieren nach [0, 1]
train_data = train_data.astype('float32') / 255.0
test_data = test_data.astype('float32') / 255.0

# Reshape, um nur eine Dimension zu haben
train_data = train_data.reshape(-1, 28, 28, 1)
test_data = test_data.reshape(-1, 28, 28, 1)

# Zu Trainingsdaten und Testdaten Rauschen hinzufügen
noise_factor = 0.5
train_noisy = train_data + noise_factor * np.random.normal(size=train_data.shape)
test_noisy =  test_data  + noise_factor * np.random.normal(size=test_data.shape)
# ... und auf Werte zwischen 0 und 1 begrenzen
train_noisy = np.clip(train_noisy, 0.0, 1.0)
test_noisy = np.clip(test_noisy, 0.0, 1.0)

# die U-Net-Modelldefinition
def unet_model(input_shape):
    input_img = Input(shape=input_shape)

    # Encoder
    x1 = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
    x2 = MaxPooling2D((2, 2), padding='same')(x1)
    x3 = Conv2D(64, (3, 3), activation='relu', padding='same')(x2)
    x4 = MaxPooling2D((2, 2), padding='same')(x3)

    # Bottleneck
    bn = Conv2D(128, (3, 3), activation='relu', padding='same')(x4)

    # Decoder
    x5 = Conv2D(64, (3, 3), activation='relu', padding='same')(bn)
    x6 = UpSampling2D((2, 2))(x5)
    # Hinzufügen von Residual-Verbindungen von x3
    x6 = Add()([x6, x3])

    x7 = Conv2D(32, (3, 3), activation='relu', padding='same')(x6)
    x8 = UpSampling2D((2, 2))(x7)
    # Hinzufügen von Residual-Verbindungen von x1
    x8 = Add()([x8, x1])

    x9 = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x8)
    
    return Model(input_img, x9)

model = unet_model((28, 28, 1))
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(train_noisy, train_data, epochs=10, batch_size=128, validation_data=(test_noisy, test_data))

denoised_images = model.predict(test_noisy)
# Plot
n = 10
plt.figure(figsize=(20, 6))
for i in range(1, n + 1):
    ax = plt.subplot(2, n, i)
    plt.imshow(test_noisy[i].reshape(28, 28), cmap='gray')
    ax.set_title('verrauscht')

    ax = plt.subplot(2, n, i + n)
    plt.imshow(denoised_images[i].reshape(28, 28), cmap='gray')
    ax.set_title('entrauscht')
plt.show()