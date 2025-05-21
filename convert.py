import numpy as np
import pandas as pd

# Charger les fichiers .npy
X = np.load("X.npy")
y = np.load("y.npy")

# S'assurer que y est en colonne
y = y.reshape(-1, 1) if y.ndim == 1 else y

# Concaténer X et y
data = np.hstack((X, y))

# Créer un DataFrame
columns = [f"feature_{i}" for i in range(X.shape[1])] + ["target"]
df = pd.DataFrame(data, columns=columns)

# Sauvegarder le tout dans un fichier CSV
df.to_csv("genes_combined.csv", index=False)

print("✅ Fichier 'genes_combined.csv' créé avec succès.")
