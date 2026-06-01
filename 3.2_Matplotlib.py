import matplotlib.pyplot as plt
import numpy as np

# generiere die Daten
x = np.linspace(0, 10, 100)
y = np.sin(x)

# Erzeuge einen plot
plt.plot(x, y)

# Damit setzt an den Titel und die Label
plt.title("Sinus-Funktion")
plt.xlabel("x-Achse")
plt.ylabel("y-Achse")

# zeige den plot
plt.show()