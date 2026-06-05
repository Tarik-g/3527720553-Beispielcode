# Aufgabe 2
# Unter den Beispielen aus meinem Buch finden Sie sowohl einen
# Backpropagation Algorithmus, als auch einen Beispielcode für den
# iRProp− Algorithmus. Verändern Sie den Backpropagation
# Algorithmus so, dass er den Gradientenabstieg vollständig mit
# iRProp− macht.

import numpy as np

# Sigmoide Aktivierungsfunktion und ihre Ableitung
def sigmoid(x):
    return 1 / (1 + np.exp(-x)) # Sigmoidfunktion

def deriv_sigmoid(x):
    return x * (1 - x) # Ableitung der Sigmoiden

# XOR Daten (Bias im Input enthalten)
inp = np.array([
    [1, 0, 0],
    [1, 0, 1],
    [1, 1, 0],
    [1, 1, 1]
])

target = np.array([[0], [1], [1], [0]])

# Die Architektur des neuronalen Netzes
inp_size = 3   # Eingabeneuronen
hid_size = 4   # Hidden-Neuronen
out_size = 1   # Ausgabeneuron


# Gewichte zufällig initialisieren (Mittelwert = 0)
# w0 = Gewichte Input → Hidden Layer
w0 = np.random.randn(inp_size, hid_size) * 0.5
# w1 = Gewichte Hidden → Output Layer
w1 = np.random.randn(hid_size + 1, out_size) * 0.5  # +1 wegen Bias in Hidden Layer

# iRProp Parameter
# Schrittweiten (Update-Magnitude)
d0 = np.ones_like(w0) * 0.1
d1 = np.ones_like(w1) * 0.1

# alte Gradienten (für Vorzeichenvergleich)
prev_g0 = np.zeros_like(w0)
prev_g1 = np.zeros_like(w1)
#[[0, 0],
# [0, 0]]

eta_plus = 1.2 # beschleunigen bei gleichem Vorzeichen
eta_minus = 0.5 # abbremsen bei Vorzeichenwechsel
d_max = 50 # maximale Schrittweite
d_min = 1e-6 # minimale Schrittweite


def forward():
    L0 = inp

    # Hidden Layer
    L1 = sigmoid(L0 @ w0)

    # Bias hinzufügen
    L1b = np.concatenate([np.ones((L1.shape[0], 1)), L1], axis=1)

    # Output
    L2 = sigmoid(L1b @ w1)

    return L0, L1, L1b, L2


for i in range(60):

    L0, L1, L1b, L2 = forward()

    error = target - L2
    loss = np.mean(error ** 2)

    # ---------- Gradienten ----------
    dL2 = error * deriv_sigmoid(L2)
    g1 = L1b.T @ dL2

    dL1 = (dL2 @ w1.T)[:, 1:] * deriv_sigmoid(L1)
    g0 = L0.T @ dL1

    # ---------- iRProp Update w1 ----------
    sign = prev_g1 * g1

    for idx in np.ndindex(w1.shape):
        if sign[idx] > 0:
            d1[idx] = min(d1[idx] * eta_plus, d_max)
            w1[idx] += np.sign(g1[idx]) * d1[idx]
            prev_g1[idx] = g1[idx]

        elif sign[idx] < 0:
            d1[idx] = max(d1[idx] * eta_minus, d_min)
            prev_g1[idx] = 0

        else:
            w1[idx] += np.sign(g1[idx]) * d1[idx]
            prev_g1[idx] = g1[idx]

    # ---------- iRProp Update w0 ----------
    sign0 = prev_g0 * g0

    for idx in np.ndindex(w0.shape):
        if sign0[idx] > 0:
            d0[idx] = min(d0[idx] * eta_plus, d_max)
            w0[idx] += np.sign(g0[idx]) * d0[idx]
            prev_g0[idx] = g0[idx]

        elif sign0[idx] < 0:
            d0[idx] = max(d0[idx] * eta_minus, d_min)
            prev_g0[idx] = 0

        else:
            w0[idx] += np.sign(g0[idx]) * d0[idx]
            prev_g0[idx] = g0[idx]

    # ---------- Debug Prints ----------
    if i % 5 == 0:
        print(f"Iteration {i}")
        print(f"Loss: {loss:.6f}")
        print(f"Output:\n{L2}\n")


# ---------- Test ----------
_, _, _, L2 = forward()
print("Final Output:")
print(L2)

# ---------- Test ----------
_, _, _, L2 = forward()

print("Final Output (probabilities):")
print(L2)

print("\nFinal Output (class labels):")
print((L2 >= 0.5).astype(int))