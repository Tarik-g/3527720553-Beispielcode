import numpy as np

def train_hopfield(patterns):
    dim = patterns.shape[1]
    weights = np.zeros((dim, dim)) # Gewichte am Anfang null
    
    for pattern in patterns:
        weights += np.outer(pattern, pattern) # jedes mit jedem
    np.fill_diagonal(weights, 0) # Keine Selbstrückkopplung

    return weights / patterns.shape[0]

def recall_hopfield(weights, patterns, iterations=5):
    recalled = np.copy(patterns)
    for iteration in range(iterations):
        for i in range(recalled.shape[0]):
            recalled[i] = np.where(np.dot(weights, recalled[i]) < 0, -1, 1)
    return recalled

# Trainingsmuster (Eingabe)
patterns = np.array([[1, -1,  1, -1,  1, -1,  1, -1],
                     [-1, 1, -1,  1, -1,  1, -1,  1]])

# Trainieren des Hopfield-Netzwerks
weights = train_hopfield(patterns)

# Testmuster (leicht von den Trainingsmustern abweichend)
test_patterns = np.array([[ 1,  1,  1, -1,  1, -1,  1, -1],
                          [-1, -1, -1,  1, -1,  1, -1,  1]])

# Recall-Phase (Erinnerung)
recalled_patterns = recall_hopfield(weights, test_patterns)

print("Rekonstruiertes Muster:")
print(recalled_patterns)
