import numpy as np
import itertools
from math import comb
import matplotlib.pyplot as plt

# Ein einfaches Modell zur Vorhersage des Hauspreises
def house_price_model(features):
    # Annahme: Der Preis steigt mit der Größe des Hauses und sinkt mit der Entfernung zum Stadtzentrum
    return features[0] * 3000 - features[1] * 10000

def shapley_value(model, base_features, feature_index):
    all_features = list(range(len(base_features)))
    n = len(all_features)
    total_value = 0

    for subset_size in range(n):
        for subset in itertools.combinations(all_features, subset_size):
            if feature_index not in subset:
                without_feature = list(subset)
                with_feature = list(subset) + [feature_index]
                
                prediction_without = model([base_features[i] if i in without_feature else 0 for i in all_features])
                prediction_with = model([base_features[i] if i in with_feature else 0 for i in all_features])
                
                marginal_contribution = (prediction_with - prediction_without) * comb(n - 1, subset_size)
                total_value += marginal_contribution
    
    return total_value / (2 ** (n - 1))

# Basis-Features: Größe des Hauses (in Quadratmetern) und Entfernung zum Stadtzentrum (in Kilometern)
base_features = [150, 5]
shap_values = [shapley_value(house_price_model, base_features, i) for i in range(len(base_features))]

# Ergebnisse anzeigen
print("Shapley-Werte:", shap_values)

# Mit Matplotlib visualisieren
plt.bar(range(len(base_features)), shap_values)
plt.xticks(range(len(base_features)), ['Größe des Hauses', 'Entfernung zum Stadtzentrum'])
plt.ylabel('Shapley-Wert')
plt.title('Shapley-Werte für jedes Feature')
plt.show()
