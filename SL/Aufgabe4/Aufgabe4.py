# Aufgabe 4
# Implementieren Sie den zweidimensionalen Pooling Layer eines
# Convolutional Networks. Laden Sie dazu ein Schwarz-Weiß Bild
# (z.B. im PGM Format) und erzeugen Sie ein kleineres Bild indem
# Sie die 2x2 Pooling Maske mit Stride 2 (heißt in Schrittweite 2)
# über das Bild schieben.

import numpy as np
import pygame

# --- Bild laden und normalisieren ---
# Das Bild liegt als CSV vor, wobei jeder Wert ein Graustufenpixel (0-255) ist.
# Division durch 255 normalisiert die Werte auf den Bereich [0.0, 1.0].
img = np.loadtxt("SL\Aufgabe4\\bild.csv", delimiter=",") / 255.0

# Bilddimensionen auslesen: H = Höhe in Pixeln, W = Breite in Pixeln
H, W = img.shape

# --- Ausgabegröße des Pooling-Ergebnisses berechnen ---
# Mit 2x2-Pooling und Stride 2 halbiert sich die Auflösung in beiden Richtungen.
# Beispiel: 100x80 Bild → 50x40 nach dem Pooling
pH = H // 2  # Höhe des gepoolten Bildes
pW = W // 2  # Breite des gepoolten Bildes

# --- Darstellungsgröße (Skalierungsfaktor) automatisch bestimmen ---
# Ziel: Das Originalbild soll möglichst groß dargestellt werden,
# aber in ein 800x600-Fenster passen. min() wählt den kleinsten
# passenden Faktor, damit nichts abgeschnitten wird. max(..., 1)
# stellt sicher, dass der Faktor nie unter 1 fällt (kein Schrumpfen).
scale = min(800 // W, 600 // H, 5)
scale = max(scale, 1)

# --- Fenstergröße berechnen ---
# Original und gepooltes Bild werden nebeneinander angezeigt,
# mit einem Abstand von 20 Pixeln dazwischen und 20 Pixeln Rand oben/unten.
# Original:      W * scale breit
# Lücke:         20 Pixel
# Pooling:       W * scale breit (gleiche Darstellungsgröße wie Original)
# → Gesamt: W * scale * 2 + 40 (je 20 Pixel Puffer links/rechts der Lücke)
xmax = W * scale * 2 + 40
ymax = H * scale + 20

# --- Pygame initialisieren und Fenster öffnen ---
pygame.init()
screen = pygame.display.set_mode((xmax, ymax))


def main():
    endlos = True
    while endlos:
        # --- Ereignisse verarbeiten ---
        # Auf das Schließen-Ereignis (X-Button) reagieren
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                endlos = False

        # Hintergrund schwarz füllen (löscht den vorherigen Frame)
        screen.fill((0, 0, 0))

        # --- Originalbild links zeichnen ---
        # Jeder Pixelwert (0.0–1.0) wird zurück auf 0–255 skaliert
        # und als gefülltes Rechteck (scale × scale Pixel) gezeichnet.
        for y in range(H):
            for x in range(W):
                val = int(255 * img[y, x])
                pygame.draw.rect(screen, (val, val, val),
                                 pygame.Rect(x * scale, y * scale, scale, scale))

        # --- Max-Pooling 2x2 mit Stride 2 (NumPy-Trick) ---
        # Schritt 1: Bild auf gerade Dimensionen zuschneiden
        #            (falls H oder W ungerade, fällt die letzte Zeile/Spalte weg)
        # Schritt 2: reshape(pH, 2, pW, 2) teilt das Bild in pH×pW Blöcke à 2×2.
        #            Achsenbedeutung nach reshape: (Block-Zeile, lokale Zeile,
        #                                          Block-Spalte, lokale Spalte)
        # Schritt 3: .max(axis=(1, 3)) nimmt das Maximum über die lokalen
        #            Achsen 1 und 3, also über die 4 Pixel jedes 2×2-Blocks.
        # Ergebnis:  pH × pW Matrix mit dem Maximalwert jedes Blocks
        maxPool = img[:H//2*2, :W//2*2].reshape(pH, 2, pW, 2).max(axis=(1, 3))

        # --- Gepooltes Bild rechts neben dem Original zeichnen ---
        # X-Versatz: Original (W * scale) + Lücke (20 Pixel)
        # Das gepoolte Bild hat zwar weniger Pixel, wird aber mit demselben
        # scale-Faktor gezeichnet → sichtbar kleiner als das Original.
        for y in range(pH):
            for x in range(pW):
                val = int(255 * maxPool[y, x])
                pygame.draw.rect(screen, (val, val, val),
                                 pygame.Rect(W * scale + 20 + x * scale, y * scale, scale, scale))

        # Fertigen Frame auf dem Bildschirm anzeigen
        pygame.display.flip()


main()