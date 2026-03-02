import os
import numpy as np
import pandas as pd
from sklearn.manifold import trustworthiness

DATA_PATH = "data/city_lifestyle_dataset.csv"   
OUTPUTS_PATH = "outputs"

print("Chargement des données originales...")

df = pd.read_csv(DATA_PATH)


X_original = df.select_dtypes(include=[np.number]).to_numpy()

print(f"X_original shape = {X_original.shape}")

methods = {
    "PCA": "pca_emb_2d.csv",
    "t-SNE": "tsne_emb_2d.csv",
    "UMAP": "umap_emb_2d.csv",
}

print("\n===== Comparaison des méthodes de réduction de dimension =====\n")
results = {}

for method_name, filename in methods.items():
    filepath = os.path.join(OUTPUTS_PATH, filename)
    if not os.path.exists(filepath):
        print(f"[{method_name}] Fichier non trouvé : {filepath}, ignoré.")
        continue

    emb = pd.read_csv(filepath)

    
    if {"x", "y"}.issubset(emb.columns):
        X_reduced = emb[["x", "y"]].to_numpy()
    else:
        X_reduced = emb.select_dtypes(include=[np.number]).iloc[:, :2].to_numpy()

    if X_reduced.shape[0] != X_original.shape[0]:
        raise ValueError(
            f"{method_name}: embedding a {X_reduced.shape[0]} lignes mais X_original en a {X_original.shape[0]}."
        )

    score = trustworthiness(X_original, X_reduced, n_neighbors=10)
    results[method_name] = score
    print(f"  {method_name:<8} → Trustworthiness : {score:.4f}")

if results:
    print("\n===== Classement =====")
    for rank, (method, score) in enumerate(sorted(results.items(), key=lambda x: x[1], reverse=True), 1):
        print(f"  {rank}. {method:<8} : {score:.4f}")

    best_method, best_score = max(results.items(), key=lambda x: x[1])
    print(f"\nLa meilleure méthode  est : {best_method} (trustworthiness = {best_score:.4f})")