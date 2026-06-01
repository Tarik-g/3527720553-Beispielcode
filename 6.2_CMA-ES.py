import numpy as np
from cmaes import CMA

# Rosenbrock Funktion
def rosenbrock(x1, x2):
    return (1 - x1) ** 2 + 10 * (x2 - x1**2) ** 2

if __name__ == "__main__":
    optimizer = CMA(mean=np.zeros(2), sigma=1.3, population_size = 10)

    # optimize over 50 generations
    for gen in range(50):
        solutions = []
        for _ in range(optimizer.population_size):
            x = optimizer.ask()
            val = rosenbrock(x[0], x[1])
            solutions.append((x, val))
            print(f"#{gen} {val} (x1={x[0]}, x2 = {x[1]})")
        optimizer.tell(solutions)