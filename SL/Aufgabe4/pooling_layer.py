"""
Aufgabe 4: 2D Pooling Layer eines Convolutional Networks
=========================================================
Implementierung von Max-Pooling und Average-Pooling mit einer
2x2 Maske und Stride 2 für Schwarz-Weiß Bilder (PGM Format).
"""

import sys
import os


# ─────────────────────────────────────────────
#  PGM Lesen / Schreiben (ohne externe Libs)
# ─────────────────────────────────────────────

def read_pgm(filename):
    """Liest ein PGM-Bild (P2 ASCII oder P5 Binary) und gibt
    (Pixel-Matrix als 2D-Liste, max_val) zurück."""
    with open(filename, 'rb') as f:
        raw = f.read()

    lines = []
    i = 0
    # Zeichen für Zeichen einlesen und Kommentare überspringen
    while i < len(raw):
        if raw[i:i+1] == b'#':
            while i < len(raw) and raw[i:i+1] != b'\n':
                i += 1
        else:
            lines.append(raw[i:i+1])
        i += 1
    content = b''.join(lines)

    tokens = content.split()
    magic   = tokens[0].decode()
    width   = int(tokens[1])
    height  = int(tokens[2])
    max_val = int(tokens[3])

    pixels = []
    if magic == 'P2':          # ASCII
        flat = [int(t) for t in tokens[4:]]
    elif magic == 'P5':        # Binary
        # Rohe Bytes nach dem Header
        header_end = content.index(tokens[3]) + len(tokens[3]) + 1
        flat = list(content[header_end:header_end + width * height])
    else:
        raise ValueError(f"Unbekanntes PGM-Format: {magic}")

    for r in range(height):
        row = flat[r * width:(r + 1) * width]
        pixels.append(row)

    return pixels, max_val, width, height


def write_pgm(filename, pixels, max_val=255):
    """Schreibt eine 2D-Liste als PGM P2 (ASCII)."""
    height = len(pixels)
    width  = len(pixels[0])
    with open(filename, 'w') as f:
        f.write(f"P2\n{width} {height}\n{max_val}\n")
        for row in pixels:
            f.write(" ".join(str(v) for v in row) + "\n")


# ─────────────────────────────────────────────
#  2D Pooling Layer
# ─────────────────────────────────────────────

def pooling_2d(image, pool_size=2, stride=2, mode='max'):
    """
    Wendet einen 2D-Pooling-Layer auf ein Graustufenbild an.

    Parameter:
    ----------
    image     : 2D-Liste mit Pixelwerten
    pool_size : Größe der quadratischen Pooling-Maske (Standard: 2)
    stride    : Schrittweite (Standard: 2)
    mode      : 'max' für Max-Pooling, 'avg' für Average-Pooling

    Rückgabe:
    ---------
    2D-Liste mit den gepoolten Pixelwerten
    """
    height = len(image)
    width  = len(image[0])

    # Ausgabegröße berechnen
    out_h = (height - pool_size) // stride + 1
    out_w = (width  - pool_size) // stride + 1

    output = []

    for r in range(out_h):
        row_out = []
        for c in range(out_w):
            # Ausschnitt (Pooling-Fenster) extrahieren
            window = []
            for pr in range(pool_size):
                for pc in range(pool_size):
                    row_idx = r * stride + pr
                    col_idx = c * stride + pc
                    window.append(image[row_idx][col_idx])

            # Pooling-Operation anwenden
            if mode == 'max':
                value = max(window)
            elif mode == 'avg':
                value = round(sum(window) / len(window))
            else:
                raise ValueError(f"Unbekannter Modus: '{mode}'. Wähle 'max' oder 'avg'.")

            row_out.append(value)
        output.append(row_out)

    return output


# ─────────────────────────────────────────────
#  Hilfsfunktionen für die Ausgabe
# ─────────────────────────────────────────────

def print_image(image, title="Bild"):
    """Gibt ein Bild als formatierte Tabelle auf der Konsole aus."""
    print(f"\n{'─' * 50}")
    print(f"  {title}  ({len(image[0])}x{len(image)} Pixel)")
    print('─' * 50)
    for row in image:
        print("  " + "  ".join(f"{v:3d}" for v in row))
    print('─' * 50)


def print_pooling_step(image, pool_size, stride, r, c, window, result, mode):
    """Zeigt einen einzelnen Pooling-Schritt detailliert."""
    row_start = r * stride
    col_start = c * stride
    print(f"    Fenster [{row_start}:{row_start+pool_size}, "
          f"{col_start}:{col_start+pool_size}]  →  "
          f"Werte {window}  →  "
          f"{'max' if mode == 'max' else 'avg'} = {result}")


# ─────────────────────────────────────────────
#  Hauptprogramm
# ─────────────────────────────────────────────

def main():
    # ── Eingabedatei bestimmen ──
    if len(sys.argv) >= 2:
        input_file = sys.argv[1]
    else:
        # Kleines 8×8 Demo-Bild erzeugen, wenn keine Datei angegeben
        input_file = "demo_input.pgm"
        demo_pixels = [
            [10,  20,  30,  40,  50,  60,  70,  80],
            [15,  25,  35,  45,  55,  65,  75,  85],
            [90, 100, 110, 120, 130, 140, 150, 160],
            [95, 105, 115, 125, 135, 145, 155, 165],
            [170,180, 190, 200, 210, 220, 230, 240],
            [175,185, 195, 205, 215, 225, 235, 245],
            [10,  50,  90, 130, 170, 210, 250,  30],
            [20,  60, 100, 140, 180, 220, 240,  40],
        ]
        write_pgm(input_file, demo_pixels)
        print(f"Demo-Bild erzeugt: '{input_file}' (8×8 Pixel)")

    # ── Bild laden ──
    print(f"\n{'═' * 50}")
    print("  Aufgabe 4: 2D Pooling Layer")
    print(f"{'═' * 50}")
    print(f"\nLade Bild: '{input_file}'")

    image, max_val, width, height = read_pgm(input_file)
    print(f"Bildgröße: {width}×{height} Pixel, Maximalwert: {max_val}")

    print_image(image, "Eingabebild")

    # ── Parameter ──
    pool_size = 2
    stride    = 2

    print(f"\nPooling-Parameter:")
    print(f"  Pool-Größe : {pool_size}×{pool_size}")
    print(f"  Stride     : {stride}")

    # ── Max-Pooling ──
    print(f"\n{'─' * 50}")
    print("  MAX-POOLING  (Detail)")
    print('─' * 50)

    out_h = (height - pool_size) // stride + 1
    out_w = (width  - pool_size) // stride + 1
    print(f"  Ausgabegröße: {out_w}×{out_h} Pixel\n")

    for r in range(out_h):
        for c in range(out_w):
            window = []
            for pr in range(pool_size):
                for pc in range(pool_size):
                    window.append(image[r*stride+pr][c*stride+pc])
            print_pooling_step(image, pool_size, stride, r, c,
                               window, max(window), 'max')

    max_pooled = pooling_2d(image, pool_size, stride, mode='max')
    print_image(max_pooled, "Ergebnis: Max-Pooling")

    # ── Average-Pooling ──
    print(f"\n{'─' * 50}")
    print("  AVERAGE-POOLING  (Detail)")
    print('─' * 50)
    print(f"  Ausgabegröße: {out_w}×{out_h} Pixel\n")

    for r in range(out_h):
        for c in range(out_w):
            window = []
            for pr in range(pool_size):
                for pc in range(pool_size):
                    window.append(image[r*stride+pr][c*stride+pc])
            avg = round(sum(window) / len(window))
            print_pooling_step(image, pool_size, stride, r, c,
                               window, avg, 'avg')

    avg_pooled = pooling_2d(image, pool_size, stride, mode='avg')
    print_image(avg_pooled, "Ergebnis: Average-Pooling")

    # ── Ergebnisse speichern ──
    out_max = "output_max_pooling.pgm"
    out_avg = "output_avg_pooling.pgm"
    write_pgm(out_max, max_pooled, max_val)
    write_pgm(out_avg, avg_pooled, max_val)

    print(f"\n✔ Max-Pooling  gespeichert: '{out_max}'")
    print(f"✔ Avg-Pooling  gespeichert: '{out_avg}'")

    # ── Zusammenfassung ──
    print(f"\n{'═' * 50}")
    print("  Zusammenfassung")
    print(f"{'═' * 50}")
    print(f"  Eingabe  : {width}×{height}  =  {width * height}  Pixel")
    print(f"  Ausgabe  : {out_w}×{out_h}   =  {out_w * out_h}  Pixel")
    reduction = (1 - (out_w * out_h) / (width * height)) * 100
    print(f"  Reduktion: {reduction:.0f} %  (bei Stride {stride}, Pool {pool_size}×{pool_size})")
    print()


if __name__ == "__main__":
    main()
