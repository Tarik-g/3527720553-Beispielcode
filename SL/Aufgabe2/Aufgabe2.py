# Aufgabe 2
# Unter den Beispielen aus meinem Buch finden Sie sowohl einen
# Backpropagation-Algorithmus als auch einen Beispielcode für den
# iRProp−-Algorithmus. Verändern Sie den Backpropagation-
# Algorithmus so, dass er den Gradientenabstieg vollständig mit
# iRProp− macht.

import numpy as np

# Sigmoide Aktivierungsfunktion und ihre Ableitung
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def deriv_sigmoid(x):
    return x * (1 - x)

# XOR-Daten (Bias bereits im Input enthalten)
inp = np.array([
    [1, 0, 0],
    [1, 0, 1],
    [1, 1, 0],
    [1, 1, 1]
])
target = np.array([[0], [1], [1], [0]])

# Architektur des neuronalen Netzes
inp_size = 3   # Eingabeneuronen
hid_size = 4   # Hidden-Neuronen
out_size = 1   # Ausgabeneuron

# Gewichte zufällig initialisieren
# w0 = Gewichte Input → Hidden Layer
w0 = np.random.randn(inp_size, hid_size) * 0.5

# w1 = Gewichte Hidden → Output Layer (+ Bias)
w1 = np.random.randn(hid_size + 1, out_size) * 0.5


## ---------------- iRProp- Parameter ----------------

# Jedes Gewicht hat seine EIGENE Schrittweite (delta).
# Anders als beim normalen Gradientenabstieg (ein globales Lernrate für alle),
# passt iRProp- die Schrittweite pro Gewicht individuell an.
# Startwert 0.1 für alle Gewichte → alle starten gleich, entwickeln sich dann unterschiedlich.
d0 = np.ones_like(w0) * 0.1   # Schrittweiten für w0 (Input → Hidden), selbe Form wie w0
d1 = np.ones_like(w1) * 0.1   # Schrittweiten für w1 (Hidden → Output), selbe Form wie w1

# Die Gradienten des VORHERIGEN Schritts werden gespeichert,
# um sie mit dem aktuellen Gradienten zu vergleichen.
# Entscheidend ist das VORZEICHEN: hat sich der Gradient gedreht?
#   → gleiches Vorzeichen: wir sind noch auf dem richtigen Weg → schneller werden
#   → anderes Vorzeichen: wir haben ein Minimum übersprungen → langsamer werden
# Startwert 0 (= kein vorheriger Gradient im ersten Schritt)
prev_g0 = np.zeros_like(w0)
prev_g1 = np.zeros_like(w1)

# Faktoren, mit denen die Schrittweite multipliziert wird:
eta_plus  = 1.2   # Gradient hat gleiches Vorzeichen → Schrittweite um 20% erhöhen
eta_minus = 0.5   # Gradient hat anderes Vorzeichen  → Schrittweite halbieren

# Schrittweiten dürfen nicht unbegrenzt wachsen oder gegen 0 schrumpfen,
# da das Training sonst explodiert oder komplett stagniert.
d_max = 50    # Obergrenze: verhindert, dass ein Gewicht riesige Sprünge macht
d_min = 1e-6  # Untergrenze: verhindert, dass die Schrittweite effektiv 0 wird

# ---------------------------------------------------


def forward():
    L0 = inp  # Eingabedaten (4×3): alle 4 XOR-Beispiele auf einmal (Batch)

    # Matrixmultiplikation: jedes Eingabemuster × Gewichtsmatrix → gewichtete Summe
    # sigmoid drückt das Ergebnis in den Bereich (0, 1)
    # L1 hat Form (4×4): 4 Beispiele, 4 Hidden-Neuronen
    L1 = sigmoid(L0 @ w0)

    # iRProp- braucht kein explizites Bias-Gewicht in w0,
    # stattdessen wird eine Spalte mit 1en vor L1 gehängt.
    # L1b hat Form (4×5): Spalte 0 = Bias (immer 1), Spalten 1–4 = Hidden-Aktivierungen
    L1b = np.concatenate([np.ones((L1.shape[0], 1)), L1], axis=1)

    # Gleiche Logik wie oben: L1b × w1 → sigmoid → Ausgabe
    # L2 hat Form (4×1): eine Wahrscheinlichkeit pro XOR-Beispiel
    L2 = sigmoid(L1b @ w1)

    return L0, L1, L1b, L2  # alle Layer zurückgeben, da Backprop sie alle braucht


for i in range(100):  # 100 Trainingsdurchläufe (Epochen)

    L0, L1, L1b, L2 = forward()  # Vorwärtsdurchlauf: Ausgabe berechnen

    # Wie weit liegt die Ausgabe vom Zielwert entfernt?
    error = target - L2           # Differenz pro Beispiel (positiv = zu klein, negativ = zu groß)
    loss  = np.mean(error ** 2)   # Wert der die Güte des Netzes beschreibt

    # --- Backpropagation: Gradienten berechnen ---

    # Gradient am Ausgang: Fehler × Ableitung der Sigmoidfunktion am Ausgang
    # "Wie stark muss sich L2 ändern, damit der Fehler kleiner wird?"
    dL2 = error * deriv_sigmoid(L2)   # Form (4×1)

    # Gradient für w1: L1b.T @ dL2
    # "Wie stark trägt jedes Gewicht in w1 zum Fehler bei?"
    g1 = L1b.T @ dL2                  # Form (5×1), ein Gradientenwert pro Gewicht in w1

    # Fehler zurück durch w1 in den Hidden Layer propagieren
    # [:, 1:] entfernt die Bias-Spalte (Bias hat keine Vorgängerschicht)
    # × deriv_sigmoid(L1): Ableitung der Sigmoidfunktion im Hidden Layer
    dL1 = (dL2 @ w1.T)[:, 1:] * deriv_sigmoid(L1)   # Form (4×4)

    # Gradient für w0: analog zu g1
    g0 = L0.T @ dL1                   # Form (3×4), ein Gradientenwert pro Gewicht in w0


    # --- iRProp- Update für w1 ---

    # Vorzeichenvergleich: prev_g1 * g1
    #   > 0 → beide gleich  → richtiger Weg, beschleunigen
    #   < 0 → Vorzeichenwechsel → Minimum übersprungen, bremsen
    #   = 0 → erster Schritt oder Gradient war 0
    sign = prev_g1 * g1

    for idx in np.ndindex(w1.shape):   # jedes Gewicht einzeln aktualisieren
        if sign[idx] > 0:
            d1[idx] = min(d1[idx] * eta_plus, d_max)    # Schrittweite erhöhen (max. d_max)
            w1[idx] += np.sign(g1[idx]) * d1[idx]       # Schritt in Gradientenrichtung
            prev_g1[idx] = g1[idx]                       # aktuellen Gradienten merken

        elif sign[idx] < 0:
            d1[idx] = max(d1[idx] * eta_minus, d_min)   # Schrittweite verkleinern (min. d_min)
            # Kein Gewichtsupdate! (das „−" in iRProp−)
            prev_g1[idx] = 0   # Gradient auf 0 setzen → nächster Schritt landet im else-Zweig

        else:
            # Erster Schritt: einfach in Gradientenrichtung gehen, Schrittweite unverändert
            w1[idx] += np.sign(g1[idx]) * d1[idx]
            prev_g1[idx] = g1[idx]


    # --- iRProp- Update für w0 (identische Logik wie w1) ---

    sign0 = prev_g0 * g0

    for idx in np.ndindex(w0.shape):
        if sign0[idx] > 0:
            d0[idx] = min(d0[idx] * eta_plus, d_max)
            w0[idx] += np.sign(g0[idx]) * d0[idx]
            prev_g0[idx] = g0[idx]

        elif sign0[idx] < 0:
            d0[idx] = max(d0[idx] * eta_minus, d_min)
            prev_g0[idx] = 0   # Gradient zurücksetzen, kein Gewichtsupdate

        else:
            w0[idx] += np.sign(g0[idx]) * d0[idx]
            prev_g0[idx] = g0[idx]


    if i % 5 == 0:   # alle 5 Iterationen Zwischenstand ausgeben
        print(f"Iteration {i}")
        print(f"Loss: {loss:.6f}")
        print(f"Output:\n{L2}\n")


# Finaler Test mit den trainierten Gewichten
_, _, _, L2 = forward()

print("Final Output (probabilities):")
print(L2)                          # Rohausgabe: Werte zwischen 0 und 1

print("\nFinal Output (class labels):")
print((L2 >= 0.5).astype(int))    # Schwellwert 0.5 → 0 oder 1 (XOR-Ergebnis)