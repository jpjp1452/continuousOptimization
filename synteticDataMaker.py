import numpy as np
import pandas as pd
import os

# Crée un dossier de sortie
output_dir = "synthetic_datasets"
os.makedirs(output_dir, exist_ok=True)

# Paramètres de génération
n_samples = 1000
feature_ranges = range(50, 1051, 50)  # de 5 à 1000, pas de 50
noise_std = 0.1  # bruit ajouté

for n_features in feature_ranges:
    # Génération aléatoire des données
    X = np.random.randn(n_samples, n_features)
    print(X)
    exit(0)
    w = np.random.randn(n_features, 1)
    noise = np.random.randn(n_samples, 1) * noise_std
    y = X @ w + noise  # produit scalaire + bruit

    # Conversion en DataFrame
    df_X = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(n_features)])
    df_y = pd.DataFrame(y, columns=["target"])
    df_full = pd.concat([df_X, df_y], axis=1)

    # Sauvegarde en CSV
    file_name = f"synthetic_data_{n_features}_features.csv"
    df_full.to_csv(os.path.join(output_dir, file_name), index=False)
    print(f"✅ Généré : {file_name}")

print("\n✅ Tous les datasets synthétiques ont été créés.")
