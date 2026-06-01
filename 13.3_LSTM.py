import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import LSTM, Dense

# Generiere unrealistische Aktienpreise
np.random.seed(0)
time_steps = 300
total_samples = 1000

# Verlauf eines erfundenen Börsenkurses
# mittels Wiener Prozess
N = 1000
T = 1.0
t = np.linspace(0.0, T, N)
dt = T / N
dW = np.sqrt(dt) * np.random.normal(size=N)
prices = np.cumsum(dW)

# Erzeuge Trainingssequenz
X = [prices[i:i+time_steps] for i in range(total_samples - time_steps)]
y = prices[time_steps:]

# Konvertieren und umformen für LSTM
X = np.array(X).reshape(-1, time_steps, 1)
y = np.array(y)

# Initialisiere sequenzielles Modell
model = Sequential()

# Füge Layer mit 5 LSTM-Zellen hinzu
model.add(LSTM(20, activation='tanh', input_shape=(time_steps, 1)))

# Füge eine Ausgabeeinheit hinzu
model.add(Dense(1))

# Kompiliere das Modell mit "mean squared error" loss und ADAM optimizer
model.compile(optimizer='adam', loss='mse')

# Trainiere das Modell
model.fit(X, y, epochs=10, verbose=1)

# Generiere Vorhersagen
future_predictions = []
current_sequence = X[-1].reshape(-1, time_steps, 1)

for _ in range(300):
    # Sage den zukünftigen Preis vorher
    future_price = model.predict(current_sequence)[0][0]
    future_predictions.append(future_price)

    # Ergänze die Sequenz durch die Zukunftsprognose
    current_sequence = np.roll(current_sequence, -1)
    current_sequence[-1][-1] = future_price

# Börsenkurs samt 30 Zukunftsprognosen plotten
plt.figure(figsize=(10, 6))
plt.plot(range(total_samples), prices, color='blue', label='Originalpreis')
plt.plot(range(total_samples, total_samples + 300), future_predictions, color='red', linestyle='dashed', label='Zukunftsvorhersage')
plt.legend()
plt.show()