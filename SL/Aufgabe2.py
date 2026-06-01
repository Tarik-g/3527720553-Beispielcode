import numpy as np

# Sigmoide Aktivierungsfunktion und ihre Ableitung
def sigmoid(x):
    return 1 / (1 + np.exp(-x)) # Sigmoidfunktion

def deriv_sigmoid(x):
    return x * (1 - x) # Ableitung der Sigmoiden

# Das XOR-Problem, input [bias, x, y] und Target-Daten
inp    = np.array([[1,0,0], [1,0,1], [1,1,0], [1,1,1]])
target = np.array([[0], [1], [1], [0]])

# Die Architektur des neuronalen Netzes
inp_size = 3   # Eingabeneuronen
hid_size = 4   # Hidden-Neuronen
out_size = 1   # Ausgabeneuron

# Gewichte zufällig initialisieren (Mittelwert = 0)
w0 = np.random.random((inp_size, hid_size)) - 0.5
w1 = np.random.random((hid_size, out_size)) - 0.5

# NEU
# --------------------------------------------------------------------------------------------------------------------------------------
# direkt aus 10.3_RProp_Neuron.py übernommen:
# iRProp- Parameter
eta_plus  = 1.2    # Faktor zur Vergrößerung der Schrittweite
eta_minus = 0.5    # Faktor zur Verkleinerung der Schrittweite
delta_max = 50.0   # Maximale Schrittweite
delta_min = 1e-6   # Minimale Schrittweite
delta_init = 0.125 # Anfangs-Schrittweite
 
# Schrittweitenmatrizen (eine pro Gewichtsmatrix)
delta_w0 = np.full_like(w0, delta_init)
delta_w1 = np.full_like(w1, delta_init)
 
# Gradienten des vorherigen Schritts (mit 0 initialisieren) = kein Vorzeichenwechsel im ersten Schritt
grad_w0_old = np.zeros_like(w0)
grad_w1_old = np.zeros_like(w1)
# --------------------------------------------------------------------------------------------------------------------------------------


# Netzwerk trainieren
for i in range(100000):

    # Vorwärtsaktivierung
    L0 = inp
    L1 = sigmoid(np.matmul(L0, w0))
    L1[0] = 1 # Bias-Neuron in der Hiddenschicht
    L2 = sigmoid(np.matmul(L1, w1))

    # Fehler berechnen
    L2_error = target - L2

    # Backpropagation
    L2_delta = L2_error * deriv_sigmoid(L2)
    L1_error = np.matmul(L2_delta, w1.T)
    L1_delta = L1_error * deriv_sigmoid(L1)
	
    # NEU
    # --------------------------------------------------------------------------------------------------------------------------------------
    # Gradienten der Gewichte (negatives Vorzeichen wegen Fehler = target - output)
    grad_w1_new = -np.matmul(L1.T, L2_delta)   # dE/dw1
    grad_w0_new = -np.matmul(L0.T, L1_delta)   # dE/dw0

    # iRProp- Gewichtsaktualisierung
    for grad_new, grad_old, delta, w in [
        (grad_w1_new, grad_w1_old, delta_w1, w1),
        (grad_w0_new, grad_w0_old, delta_w0, w0),
    ]:
        # Schrittweite anpassen
        sign_change = grad_old * grad_new
        pos = sign_change > 0 # Minimum noch nicht übersprungen → Schrittweite vergrößern
        neg = sign_change < 0 # Minimum übersprungen → Schrittweite verkleinern.
 
        delta[pos] = np.minimum(delta[pos] * eta_plus,  delta_max)
        delta[neg] = np.maximum(delta[neg] * eta_minus, delta_min)
 
        # iRProp-: bei Vorzeichenwechsel Gradienten auf 0 setzen
        grad_new[neg] = 0.0
 
        # Gewichte aktualisieren
        w -= delta * np.sign(grad_new)
 
    # Alte Gradienten speichern
    grad_w1_old = grad_w1_new.copy()
    grad_w0_old = grad_w0_new.copy()
    # --------------------------------------------------------------------------------------------------------------------------------------

# Netzwerk testen
L0 = inp
L1 = sigmoid(np.matmul(inp, w0))
L1[0] = 1 # Bias-Neuron in der Hiddenschicht 
L2 = sigmoid(np.matmul(L1, w1))

print(L2)