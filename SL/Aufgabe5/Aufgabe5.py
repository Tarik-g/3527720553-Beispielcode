# Aufgabe 5
# Programmieren Sie ein rekurrentes Neuronales Netz aus zwei
# Neuronen mit Tangens hyperbolicus Transferfunktion:
# FORMEL
# und folgenden Gewichten:
# wbias1 = −3.37, wbias2 = 0.125,
# w11 = −4, w12 = 1.5,
# w21 = −1.5, w22 = 0
# und den Anfangswerte: o1 = 0.0, o2 = 0.0
# Zeichnen Sie den Output o1 und o2 der beiden Neuronen mit Hilfe
# von Matplotlib in ein zweidimensionales Diagramm.

import math
import matplotlib.pyplot as plt

class NN:
    def __init__(self):
        self.o1 = 0.0
        self.o2 = 0.0

        # korrekte Gewichte
        self.w11 = -4
        self.w12 = 1.5
        self.w21 = -1.5
        self.w22 = 0

        self.bias1 = -3.37
        self.bias2 = 0.125

        self.history_o1 = []
        self.history_o2 = []

    def activate(self):
        new_o1 = math.tanh(self.w11*self.o1 + self.w12*self.o2 + self.bias1)
        new_o2 = math.tanh(self.w21*self.o1 + self.w22*self.o2 + self.bias2)

        self.o1, self.o2 = new_o1, new_o2

        self.history_o1.append(new_o1)
        self.history_o2.append(new_o2)

n = NN()

# Simulation über Zeit
for _ in range(100):
    n.activate()

# Plot
plt.plot(n.history_o1, label="o1")
plt.plot(n.history_o2, label="o2")
plt.legend()
plt.xlabel("Zeit")
plt.ylabel("Output")
plt.title("Rekurrentes Neuronales Netz (2 Neuronen)")
plt.show()