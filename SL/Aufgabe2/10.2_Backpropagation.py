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

    # Gewichte aktualisieren 
    learnrate = 0.1
    w1 += learnrate * np.matmul(L1.T, L2_delta)
    w0 += learnrate * np.matmul(L0.T, L1_delta)

# Netzwerk testen
L0 = inp
L1 = sigmoid(np.matmul(inp, w0))
L1[0] = 1 # Bias-Neuron in der Hiddenschicht 
L2 = sigmoid(np.matmul(L1, w1))

print(L2)