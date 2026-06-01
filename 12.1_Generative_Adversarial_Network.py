import tensorflow as tf
from tensorflow.keras.layers import Dense, LeakyReLU, Reshape, Flatten
from tensorflow.keras.models import Sequential
import matplotlib.pyplot as plt
import numpy as np

# Lade die MNIST Ziffern und normiere sie
(x_train, _), (_, _) = tf.keras.datasets.mnist.load_data()

x_train = x_train.reshape(x_train.shape[0], 28 * 28).astype('float32')
x_train = (x_train - 127.5) / 127.5  # normiere die Bilder auf [-1, 1]

buffer_size = x_train.shape[0]
batch_size = 256
train_dataset = tf.data.Dataset.from_tensor_slices(x_train).shuffle(buffer_size).batch(batch_size)

# Generator
def create_generator():
    model = Sequential([
        Dense(256, input_shape=(100,), activation=LeakyReLU(0.2)),
        Dense(512, activation=LeakyReLU(0.2)),
        Dense(1024, activation=LeakyReLU(0.2)),
        Dense(28 * 28, activation='tanh')
    ])
    return model

# Discriminator
def create_discriminator():
    model = Sequential([
        Dense(1024, input_shape=(28 * 28,), activation=LeakyReLU(0.2)),
        Dense(512, activation=LeakyReLU(0.2)),
        Dense(256, activation=LeakyReLU(0.2)),
        Dense(2, activation='softmax')
    ])
    return model

generator = create_generator()
discriminator = create_discriminator()

# Loss und Optimierer
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

# Training eines Schrittes
@tf.function
def train_step(images):
    noise = tf.random.normal([batch_size, 100])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)

        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)

        gen_loss = cross_entropy(tf.ones_like(fake_output), fake_output)
        disc_loss = cross_entropy(tf.ones_like(real_output), real_output) + cross_entropy(tf.zeros_like(fake_output), fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

# Training Schleife
def train(dataset, epochs):
    for epoch in range(epochs):
        print(f'Epoch {epoch+1}')
        for batch in dataset:
            train_step(batch)

epochs = 100
train(train_dataset, epochs)

# Generiere 100 Bilder
num_images = 100
noise = tf.random.normal([num_images, 100])
generated_images = generator(noise, training=False)

# Skaliere und formatiere die generierten Bilder
generated_images = (generated_images + 1) / 2
generated_images = generated_images.numpy().reshape(num_images, 28, 28)

# Visualisiere die Bilder
rows, cols = 10, 10
fig, axes = plt.subplots(rows, cols, figsize=(15, 15))

for i in range(rows):
    for j in range(cols):
        ax = axes[i, j]
        ax.imshow(generated_images[i * cols + j], cmap='gray')
        ax.axis('off')

plt.show()

