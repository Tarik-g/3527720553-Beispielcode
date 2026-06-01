import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

class PosEnc(layers.Layer):
    def __init__(self, **kwargs):
        super(PosEnc, self).__init__(**kwargs)

    def build(self, input_shape):
        _, seq_len, d_model = input_shape
        self.positional_encoding = self.get_pos_enc(seq_len, d_model)
        super(PosEnc, self).build(input_shape)

    def call(self, x):
        return x + self.positional_encoding

    @staticmethod
    def get_pos_enc(seq_len, d_model):
        angles = np.arange(seq_len)[:, np.newaxis] / np.power(10000, 2 * np.arange(d_model)[np.newaxis, :] // d_model)
        angles[:, 0::2] = np.sin(angles[:, 0::2])
        angles[:, 1::2] = np.cos(angles[:, 1::2]) 
        return tf.cast(angles[np.newaxis, ...], tf.float32)

# -- generiert Buchstabensequenzen x und Targetbuchstaben y
def gen_train_data(text, tokenizer, seq_len):
    encoded = tokenizer.texts_to_sequences([text])[0]
    sequences = []
    for i in range(seq_len, len(encoded)):
        sequence = encoded[i-seq_len:i+1]
        sequences.append(sequence)
    sequences = np.array(sequences)
    x, y = sequences[:, :-1], sequences[:, -1] 
    return x, y

def create_transformer_model(vocab_size, d_model, nhead, max_seq_len, mask):
    inputs = tf.keras.Input(shape=(max_seq_len,))
    embedding = layers.Embedding(input_dim=vocab_size, output_dim=d_model)(inputs)
    pos_encoding = PosEnc()(embedding)
    x = pos_encoding

    # Multi-Head Attention mit Residual-Verbindung
    attention_output = layers.MultiHeadAttention(num_heads=nhead, key_dim=d_model // nhead)(x, x, attention_mask=mask)
    x = layers.Add()([x, attention_output])
    x = layers.LayerNormalization()(x)
    x = layers.Dropout(0.1)(x)

    # Zwei Dense Layer mit Residual-Verbindung
    d_1 = layers.Dense(d_model, activation='relu')(x)
    d_2 = layers.Dense(d_model, activation='relu')(d_1)
    x = layers.Add()([x, d_2])
    x = layers.LayerNormalization()(x)
    x = layers.Dropout(0.1)(x)

    logits = layers.Dense(vocab_size, activation='softmax')(x[:, -1, :])

    model = tf.keras.Model(inputs=inputs, outputs=logits)
    return model

# --- Parameter
train_text = "Welches Tier ist das größte? Der Wal. Welches Tier ist das kleinste? Der Einzeller."
seq_len = 32    # Sequenzlänge
batch_size = 32 # Batch-Länge
epochs = 100    # Trainingsepochen

# --- Der Tokenizer codiert folgende Zeichen
chars = "\n,.;:-/!?$&'ABCDEFGHIJKLMNOPQRSTUVWXYZ abcdefghijklmnopqrstuvwxyzßöäüÖÄÜ0123456789"
tokenizer = Tokenizer(char_level=True, filters='', lower=False)
tokenizer.fit_on_texts(chars)

# --- Die Maske ist in Keras vom Datentyp bool
mask = np.ones((seq_len, seq_len), dtype=bool)
mask[np.triu_indices(seq_len, 1)] = False

# --- erzeuge Trainingsdaten
x_train, y_train = gen_train_data(train_text, tokenizer, seq_len)

# --- erzeuge das Transformer-Model
vocab_size = len(tokenizer.word_index)+1
d_model = vocab_size # Dimension der Ausgabe des Modells
nhead = 4  # 1 x Multi-Head Attention mit 4 Köpfen
model = create_transformer_model(vocab_size, d_model, nhead, seq_len, mask)
model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=['accuracy'])

# --- Trainiere das Modell
model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size)

# --- Generiere den Text
text = "Welches Bier ist das kleinste?"
for _ in range(seq_len):
    enc_txt = tokenizer.texts_to_sequences([text])[0]
    padded_txt = pad_sequences([enc_txt], maxlen=seq_len, padding='pre', truncating='pre')
    logits = model.predict(padded_txt) # aktivieren
    next_char = np.argmax(logits[0, :]) # Besten nehmen
    next_char = tokenizer.index_word[next_char]
    text += next_char # Buchstabe anhängen
    print("Generierter Text:",text)
    if next_char=='.': # Punkt = Stopp!
        break