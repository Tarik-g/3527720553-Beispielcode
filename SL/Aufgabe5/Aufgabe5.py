# Aufgabe 5
# Rekurrentes Neuronales Netz mit 2 Neuronen und tanh-Aktivierungsfunktion

import math
import matplotlib.pyplot as plt

class NN:
    def __init__(self):
        # Anfangswerte beider Neuronen
        self.o1 = 0.0
        self.o2 = 0.0

        # Verbindungsgewichte zwischen den Neuronen
        # w11: o1 -> o1, w12: o2 -> o1
        # w21: o1 -> o2, w22: o2 -> o2
        self.w11 = -4
        self.w12 = 1.5
        self.w21 = -1.5
        self.w22 = 0

        # Bias-Gewichte (Schwellwert) pro Neuron
        self.bias1 = -3.37
        self.bias2 = 0.125

        # Ausgabeverläufe für den Plot
        self.history_o1 = []
        self.history_o2 = []

    def activate(self):
        # Neuen Output berechnen – beide gleichzeitig aus den alten Werten
        # o_neu = tanh(w11*o1 + w12*o2 + bias)
        new_o1 = math.tanh(self.w11*self.o1 + self.w12*self.o2 + self.bias1)
        new_o2 = math.tanh(self.w21*self.o1 + self.w22*self.o2 + self.bias2)

        # Erst nach der Berechnung überschreiben (simultanes Update)
        self.o1, self.o2 = new_o1, new_o2

        # Werte für spätere Visualisierung speichern
        self.history_o1.append(new_o1)
        self.history_o2.append(new_o2)

# Netz erstellen und 100 Zeitschritte simulieren
n = NN()
for _ in range(10):
    n.activate()

# Ausgabeverläufe beider Neuronen plotten
plt.plot(n.history_o1, label="o1")
plt.plot(n.history_o2, label="o2")
plt.legend()
plt.xlabel("Zeitschritt")
plt.ylabel("Ausgabe (tanh)")
plt.title("Rekurrentes Neuronales Netz (2 Neuronen)")
plt.show()