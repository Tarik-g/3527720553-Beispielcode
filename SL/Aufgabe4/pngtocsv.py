from PIL import Image
import numpy as np

# --- Bild laden und zu Graustufen konvertieren ---
img = Image.open("SL\Aufgabe4\globe.png").convert('L')  # <-- Pfad anpassen
arr = np.array(img)

# --- Als CSV speichern ---
np.savetxt("SL\Aufgabe4\\bild.csv", arr, delimiter=",", fmt="%d")

print(f"Gespeichert: {arr.shape[0]} x {arr.shape[1]} Pixel → bild.csv")