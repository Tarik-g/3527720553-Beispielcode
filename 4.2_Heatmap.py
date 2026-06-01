import matplotlib.pyplot as plt
import numpy as np

# Beispieldaten generieren
data = np.random.rand(10, 10)

# Zeichne Heatmap ...
fig, ax = plt.subplots()
im = ax.imshow(data, cmap="YlGnBu")

# ... mit Text
for i in range(data.shape[0]):
    for j in range(data.shape[1]):
        if data[i,j] > 0.5: # Farbe abh. vom Hintergrund
            text_color = 'white' # für dunklen Hintergrund
        else:
            text_color = 'black' # für hellen Hintergrund
        text = ax.text(j, i, "{:.2f}".format(data[i, j]), ha="center", va="center", color=text_color)

# und eine Farbleiste
cbar = ax.figure.colorbar(im, ax=ax)

# Titel und Achsenbeschriftung
ax.set_title("Beispiel-Heatmap")
ax.set_xlabel("x-Label")
ax.set_ylabel("y-Label")

# Plot zeichnen
plt.show()