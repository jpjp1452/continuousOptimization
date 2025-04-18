import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler

# Chargement des données
X, y = load_diabetes(return_X_y=True)
scaler = StandardScaler()
X = scaler.fit_transform(X)
y = y - y.mean()

# Paramètres
n_iter = 100
alpha = 0.1  # régularisation (lambda)
L = np.linalg.norm(X, ord=2) ** 2  # Lipschitz constant
t = 1.0 / L  # step size

def soft_thresholding(x, thresh):
    return np.sign(x) * np.maximum(np.abs(x) - thresh, 0.)

# ISTA
def ista(X, y, alpha, n_iter=100):
    w = np.zeros(X.shape[1])
    history = []
    for _ in range(n_iter):
        grad = X.T @ (X @ w - y)
        w = soft_thresholding(w - t * grad, alpha * t)
        history.append(np.linalg.norm(X @ w - y) ** 2)
    return w, history

# FISTA
def fista(X, y, alpha, n_iter=100):
    w = np.zeros(X.shape[1])
    z = w.copy()
    history = []
    t_k = 1
    for _ in range(n_iter):
        grad = X.T @ (X @ z - y)
        w_new = soft_thresholding(z - t * grad, alpha * t)
        t_k_new = (1 + np.sqrt(1 + 4 * t_k ** 2)) / 2
        z = w_new + ((t_k - 1) / t_k_new) * (w_new - w)
        w = w_new
        t_k = t_k_new
        history.append(np.linalg.norm(X @ w - y) ** 2)
    return w, history

# Exécution
w_ista, loss_ista = ista(X, y, alpha, n_iter)
w_fista, loss_fista = fista(X, y, alpha, n_iter)

# Référence scikit-learn
lasso = Lasso(alpha=alpha, max_iter=10000)
lasso.fit(X, y)
w_sklearn = lasso.coef_

# Comparaison des résultats
print("Erreur ISTA :", np.linalg.norm(w_ista - w_sklearn))
print("Erreur FISTA:", np.linalg.norm(w_fista - w_sklearn))

# Affichage des courbes de convergence
plt.plot(loss_ista, label="ISTA")
plt.plot(loss_fista, label="FISTA")
plt.xlabel("Itérations")
plt.ylabel("Erreur quadratique")
plt.title("Convergence de ISTA vs FISTA")
plt.legend()
plt.grid()
plt.show()
