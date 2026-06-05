

# Gewichte zufällig initialisieren (Mittelwert = 0)
# w0 = Gewichte Input → Hidden Layer
w0 = np.random.random((inp_size, hid_size)) - 0.5
# w1 = Gewichte Hidden → Output Layer
w1 = np.random.random((hid_size, out_size)) - 0.5

# NEU
# --------------------------------------------------------------------------------------------------------------------------------------
# iRProp- ersetzt Gradientenabstieg durch adaptive Schrittweiten, nutzt nur das Vorzeichen des Gradienten

# direkt aus 10.3_RProp_Neuron.py übernommen:
# iRProp- Parameter
eta_plus  = 1.2    # Faktor zur Vergrößerung der Schrittweite
eta_minus = 0.5    # Faktor zur Verkleinerung der Schrittweite
delta_max = 1      # Maximale Schrittweite
delta_min = 1e-6   # Minimale Schrittweite
delta_init = 0.125 # Anfangs-Schrittweite
 
# Schrittweitenmatrizen (eine pro Gewichtsmatrix)
# delta_w*: individuelle Schrittweite pro Gewicht (iRProp)
# selber Array aber mit delta_init gefüllt, damit alle Gewichte mit der gleichen Anfangs-Schrittweite starten
delta_w0 = np.full_like(w0, delta_init)
delta_w1 = np.full_like(w1, delta_init)
#[[0.125 0.125 0.125 0.125]
 #[0.125 0.125 0.125 0.125]
 #[0.125 0.125 0.125 0.125]]
 
# zeros_like: nötig, weil wir Arrays gleicher Form wie die Gewichte brauchen
# grad_w*_old: speichert vorherigen Gradienten für Vorzeichenvergleich
# Gradienten des vorherigen Schritts (mit 0 initialisieren) = kein Vorzeichenwechsel im ersten Schritt
grad_w0_old = np.zeros_like(w0)
grad_w1_old = np.zeros_like(w1)
#[[0, 0],
# [0, 0]]
# --------------------------------------------------------------------------------------------------------------------------------------


# Netzwerk trainieren
for i in range(100000):
    # Vorwärtsaktivierung
    L0 = inp
    L1 = sigmoid(np.matmul(L0, w0))
    #L1[:, 0] = 1  # FIX: erstes Neuron aller Samples auf 1 (Bias-Spalte) Bias-Neuron in der Hiddenschicht
    L1[0] = 1
    L2 = sigmoid(np.matmul(L1, w1))

    # Fehler berechnen
    L2_error = target - L2

    # Backpropagation
    L2_delta = L2_error * deriv_sigmoid(L2)
    L1_error = np.matmul(L2_delta, w1.T)
    L1_delta = L1_error * deriv_sigmoid(L1)

    loss = np.mean(np.abs(L2_error))
    print(f"Schritt {i}: Loss = {loss:.4f}, L2 = {L2.flatten()}")
	
    # NEU
    # --------------------------------------------------------------------------------------------------------------------------------------
    # Berechnet, wie stark jedes Gewicht den Fehler beeinflusst
    # (Gradient = Richtung, in die der Fehler am stärksten steigt)
    grad_w1_new = np.matmul(L1.T, L2_delta)   # Gradient für w1 (Hidden → Output)
    grad_w0_new = np.matmul(L0.T, L1_delta)   # Gradient für w0 (Input → Hidden)

    # iRProp- Gewichtsaktualisierung
    # Für jede Gewichtsmatrix (w1 und w0) wird das gleiche Update gemacht
    for grad_new, grad_old, delta, w in [
        (grad_w1_new, grad_w1_old, delta_w1, w1),
        (grad_w0_new, grad_w0_old, delta_w0, w0),
    ]:
        # Schrittweite anpassen
        sign_change = grad_old * grad_new
        pos = sign_change > 0 # Minimum noch nicht übersprungen → Schrittweite vergrößern
        neg = sign_change < 0 # Minimum übersprungen → Schrittweite verkleinern.
 
        
        # Bei stabiler/konstanter Gradientenrichtung:
        # Schrittweite erhöhen (beschleunigt Konvergenz)
        delta[pos] = np.minimum(delta[pos] * eta_plus,  delta_max)
        # Bei Vorzeichenwechsel:
        # Schrittweite reduzieren (vorsichtigere Updates)
        delta[neg] = np.maximum(delta[neg] * eta_minus, delta_min)
 
        # iRProp-: nach Vorzeichenwechsel wird der aktuelle Gradient ignoriert
        # verhindert ein sofortiges Zurückspringen in die neue Richtung
        grad_new[neg] = 0.0
 
        # Gewichte aktualisieren
        # Nur die Richtung (Vorzeichen) des Gradienten zählt
        # Schrittgröße kommt nur von delta
        w[:] -= delta * np.sign(grad_new)
 
    # Alte Gradienten speichern
    grad_w1_old = grad_w1_new.copy()
    grad_w0_old = grad_w0_new.copy()
    # --------------------------------------------------------------------------------------------------------------------------------------

# Netzwerk testen
L0 = inp
L1 = sigmoid(np.matmul(inp, w0))
# L1[:, 0] = 1  # FIX : auch beim Testen konsistent Bias-Neuron in der Hiddenschicht 
L1[0] = 1
L2 = sigmoid(np.matmul(L1, w1))

print(L2)