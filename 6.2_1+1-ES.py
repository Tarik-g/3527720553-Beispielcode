import numpy as np
import random
def objective_function(x):
    """Die Zielfunktion, die wir optimieren möchten."""
    return x**2

def one_plus_one_es(max_iterations=100, initial_solution=10.0, initial_step_size=1.0, step_size_adaptation=0.85):
    """Ein einfacher (1+1)-ES Algorithmus."""
    
    # Initialisiere den aktuellen Punkt und die Schrittweite
    x = initial_solution
    sigma = initial_step_size
    
    for iteration in range(max_iterations):
        # Erzeuge einen Nachkommen durch Mutation
        y = x + sigma * random.gauss(0, 1)
        
        # Wenn der Nachkomme besser ist, aktualisiere den aktuellen Punkt
        if objective_function(y) < objective_function(x):
            x = y
            sigma /= step_size_adaptation  # Erhöhe die Schrittweite
        else:
            sigma *= step_size_adaptation  # Verringere die Schrittweite
        
        # Ausgabe der aktuellen Lösung und des Funktionswerts
        print(f"Iteration {iteration + 1}: x = {x:.4f}, f(x) = {objective_function(x):.4f}")
    
    return x

# Führe den (1+1)-ES Algorithmus aus
best_solution = one_plus_one_es()
print(f"\nBeste gefundene Lösung: x = {best_solution:.4f}, f(x) = {objective_function(best_solution):.4f}")

