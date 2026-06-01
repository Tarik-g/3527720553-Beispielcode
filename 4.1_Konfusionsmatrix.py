import matplotlib.pyplot as plt
import numpy as np

# Beispieldaten kreieren
actual_labels = ["Hund", "Hund", "Katze", "Katze", "Maus", "Maus", "Maus"]
predicted_labels = ["Hund", "Hund", "Katze", "Hund", "Maus", "Katze", "Maus"]

# Konfusionsmatrix zeichnen
labels = np.unique(actual_labels)
confusion_matrix = np.zeros((len(labels), len(labels)))
for i in range(len(actual_labels)):
    actual_index = np.where(labels == actual_labels[i])[0][0]
    predicted_index = np.where(labels == predicted_labels[i])[0][0]
    confusion_matrix[actual_index][predicted_index] += 1

# zeichne die Konfusionsmatrix
fig, ax = plt.subplots()
im = ax.imshow(confusion_matrix)

# Label und Farbbalken hinzufügen
ax.set_xticks(np.arange(len(labels)))
ax.set_yticks(np.arange(len(labels)))
ax.set_xticklabels(labels)
ax.set_yticklabels(labels)
im = ax.imshow(confusion_matrix, cmap='YlGnBu')
plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
for i in range(len(labels)):
    for j in range(len(labels)):
        text_color = "white" if confusion_matrix[i, j] > confusion_matrix.max() / 2 else "black"
        text = ax.text(j, i, int(confusion_matrix[i, j]), ha="center", va="center", color=text_color)
cbar = ax.figure.colorbar(im, ax=ax)

# Achenbeschriftung und Titel
ax.set_title("Konfusionsmatrix")
ax.set_xlabel("vorhergesagte Klasse")
ax.set_ylabel("wirkliche Klasse")

# zeige den Plot
plt.show()