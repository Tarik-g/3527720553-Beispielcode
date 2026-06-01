# Importiere die notwendigen Bibliotheken
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

# Erstelle ein Beispiel-Array (Tabelle) für Training
data = [
    [20, 65, 10, 0, 1],  # Schönes Wetter, wir spielen Fußball
    [25, 80, 5, 1, 0],   # Es regnet, wir spielen nicht
    [18, 70, 15, 0, 1],  # Noch gutes Wetter, wir spielen
    [10, 90, 20, 1, 0],  # Kalt und regnerisch, wir spielen nicht
    [15, 85, 25, 0, 0],  # Zu windig, wir spielen nicht
    [22, 60, 5, 0, 1],   # Schönes Wetter, wir spielen
    [30, 50, 10, 0, 0],  # Zu heiß, wir spielen nicht
    [20, 70, 8, 0, 1]    # Schönes Wetter, wir spielen
]

# Trenne die Merkmale (Features) und das Ziel (Target)
X = [row[:-1] for row in data]
y = [row[-1] for row in data]

# Erstelle und trainiere den Entscheidungsbaum
tree_classifier = DecisionTreeClassifier()
tree_classifier.fit(X, y)

# Visualisiere den Baum
plt.figure(figsize=(10, 7))
plot_tree(tree_classifier, filled=True, feature_names=['Temperatur', 'Luftfeuchtigkeit', 'Windgeschwindigkeit', 'Regen'], class_names=['Nicht spielen', 'Spielen'],fontsize=10)
plt.show()
