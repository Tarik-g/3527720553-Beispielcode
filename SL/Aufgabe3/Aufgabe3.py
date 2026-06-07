# Restricted Boltzmann Machine (RBM) als einfacher Autoencoder
# Aufgabe: MNIST-Ziffern encodieren & rekonstruieren (nur 100 Hidden-Neuronen)

import numpy as np
import pygame

# --------------------------------------------------------------------------------------------------
# Fenster
# --------------------------------------------------------------------------------------------------
xmax = 600
ymax = 200
pygame.init()
screen = pygame.display.set_mode((xmax, ymax))

# --------------------------------------------------------------------------------------------------
# Hilfsfunktionen
# --------------------------------------------------------------------------------------------------
def sigmoid(x):
    # Quetscht beliebige Werte auf [0, 1]
    return 1.0 / (1.0 + np.exp(-x))

# --------------------------------------------------------------------------------------------------
# RBM KLASSE
# --------------------------------------------------------------------------------------------------
class RBM:
    
    def __init__(self, visible=28*28, hidden=100):
        # visible = 784 Pixel-Neuronen, hidden = 100 komprimierte Features
        self.visible = visible
        self.hidden = hidden

        # Gewichte: kleine Zufallswerte (symmetrie brechen)
        self.W = np.random.normal(0, 0.01, size=(hidden, visible))
        # Bias: anfangs null
        self.v_bias = np.zeros(visible)
        self.h_bias = np.zeros(hidden)

    def hidden_prob(self, v):
        # Sichtbar → Versteckt: Encoding
        return sigmoid(self.W @ v + self.h_bias)

    def visible_prob(self, h):
        # Versteckt → Sichtbar: Decoding (transponierte Gewichte)
        return sigmoid(self.W.T @ h + self.v_bias)

    def contrastive_divergence(self, v0, lr=0.1):
        # === Contrastive Divergence (CD-1): der Kern des RBM-Lernens ===

        # Schritt 1: Positiv-Phase – reales Bild encodieren
        h0 = sigmoid(self.W @ v0 + self.h_bias)

        # Schritt 2: Rekonstruktion – aus Hidden zurück ins Bildraum
        v1 = sigmoid(self.W.T @ h0 + self.v_bias)

        # Schritt 3: Negativ-Phase – Hidden aus Rekonstruktion
        h1 = sigmoid(self.W @ v1 + self.h_bias)

        # Schritt 4: Gewichte anpassen
        # Positiv: Original stärken, Negativ: Rekonstruktion schwächen
        # outer() = äußeres Produkt → liefert Gewichtsmatrix aus zwei Vektoren
        self.W += lr * (np.outer(h0, v0) - np.outer(h1, v1))
        self.v_bias += lr * (v0 - v1)
        self.h_bias += lr * (h0 - h1)

        return h0, v1  # Hidden-Aktivierungen & rekonstruiertes Bild
        
    def train(self, v0, lr=0.1):
        return self.contrastive_divergence(v0, lr)

def main():
    # MNIST laden: jede Zeile = ein Bild, 784 Pixelwerte (0–255)
    test_data = np.loadtxt("SL/Aufgabe3/mnist.csv", delimiter=",")
    
    # RBM anlegen: 784 → 100 → 784
    RBMaschine = RBM()

    pattern = 0
    endlos = True

    while endlos:

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                endlos = False

        # ------------------------------------------------------------------
        # INPUT: Bild laden & als Rot-Kanal links anzeigen
        # ------------------------------------------------------------------
        screen.fill((0, 0, 0))

        ii = 0
        in_vec = np.zeros(28*28)

        for y in range(28):
            for x in range(28):

                val = test_data[pattern, ii] / 255.0  # Normierung auf [0,1]
                in_vec[ii] = val
                ii += 1

                pygame.draw.rect(
                    screen,
                    (int(255 * val), 0, 0),  # rot
                    pygame.Rect(x * 5, y * 5, 5, 5)
                )

        # ------------------------------------------------------------------
        # TRAINING: ein Schritt CD-1 → gibt Hidden & Rekonstruktion zurück
        # ------------------------------------------------------------------
        hid_vec, out_vec = RBMaschine.train(in_vec, 0.1)

        # ------------------------------------------------------------------
        # HIDDEN: 100 Neuronen als 10×10 Gitter in Grün (Mitte)
        # ------------------------------------------------------------------
        jj = 0
        for y in range(10):
            for x in range(10):

                val = hid_vec[jj]
                jj += 1

                pygame.draw.rect(
                    screen,
                    (0, int(255 * val), 0),  # grün
                    pygame.Rect(200 + x * 10, y * 10, 10, 10)
                )

        # ------------------------------------------------------------------
        # OUTPUT: rekonstruiertes Bild als Blau-Kanal rechts
        # ------------------------------------------------------------------
        kk = 0
        for y in range(28):
            for x in range(28):

                val = out_vec[kk]
                kk += 1

                pygame.draw.rect(
                    screen,
                    (0, 0, int(255 * val)),  # blau
                    pygame.Rect(400 + x * 5, y * 5, 5, 5)
                )

        # ------------------------------------------------------------------
        # Alle 100 Bilder kurze Pause zum Anschauen
        # ------------------------------------------------------------------
        if pattern % 100 == 0:
            pygame.time.delay(500)

        pattern = (pattern + 1) % len(test_data)  # nächstes Bild (zyklisch)
        pygame.display.flip()

main()