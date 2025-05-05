import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Lasso


LASSO=0
ELASTICNET=1



# Charger les données California Housing
data = fetch_california_housing()

A = data.data
b = data.target

# Normalisation
scaler = StandardScaler()
A = scaler.fit_transform(A)

# Train/test split
A_train, A_test, b_train, b_test = train_test_split(A, b, test_size=0.2, random_state=42)



def gradientCurrying(A, b):
    def g_grad(w):
        return A.T @ (A @ w - b)
    return g_grad
def soft_thresholding(x, lam):
    return np.sign(x) * np.maximum(np.abs(x) - lam, 0.0)
    




def ISTA(A, b, lam1=0.1, MODE=LASSO, lam2=0.01, max_iter=50000, tol=1e-16, adaptive_step=False):
    if(MODE != LASSO and MODE != ELASTICNET):
        raise ValueError("MODE must be either LASSO or ELASTICNET")
    if(max_iter <= 0):
        raise ValueError("max_iter must be positive")
    grad = gradientCurrying(A, b)
    
    
    n = A.shape[1]
    x_i = np.zeros(n)
    L = np.linalg.norm(A.T @ A, 2)
    t = 1.0 / L
    
    logs =[x_i]
    for k in range(max_iter):
        x_i_old = x_i.copy()
        
        if(MODE==LASSO):
            # Gradient step
            x_i = x_i - t * grad(x_i)
            # Proximal operator (soft-thresholding)
            x_i = soft_thresholding(x_i, lam1*t)

        else:
            # Gradient step
            x_i = x_i - t * (grad(x_i) + 2 * lam2 * x_i)
            # Proximal operator (soft-thresholding)
            x_i = soft_thresholding(x_i, lam1*t)
        # Critère d'arrêt
        
        if(adaptive_step):
            # Mise à jour de la taille du pas
            Lk = np.linalg.norm(grad(x_i) - grad(x_i_old), 2) / np.linalg.norm(x_i - x_i_old, 2)
            if Lk > L / 2:
                L = Lk
                t = 1.0 / L
        logs.append(x_i)
        if np.linalg.norm(x_i - x_i_old) < tol:
            break
        
    return x_i,logs








def fista(A, b, lam1=0.1, MODE=LASSO, lam2=0.01, max_iter=50000, tol=1e-16):
    if(MODE != LASSO and MODE != ELASTICNET):
        raise ValueError("MODE must be either LASSO or ELASTICNET")
    if(max_iter <= 0):
        raise ValueError("max_iter must be positive")
    """
    FISTA pour le problème LASSO: min_x 0.5*||Ax - b||^2 + lambda * ||x||_1

    Paramètres :
    A : matrice (m x n)
    b : vecteur (m,)
    lambda_reg : paramètre de régularisation
    L : constante de Lipschitz de grad f (typiquement, L = plus grande valeur propre de A^T A)
    """
    n = A.shape[1]
    x_i = np.zeros(n)
    y = x_i.copy()
    t = 1

    grad = gradientCurrying(A, b)
    logs =[x_i]
    for k in range(max_iter):
        x_old = x_i.copy()


        # Étape de mise à jour via l’opérateur proximal (soft-thresholding ici)
        
        if(MODE==LASSO):
            x_i = soft_thresholding(y - (1/L) * (grad(y)), lam1/L)
        else:
            x_i = soft_thresholding(y - (1/L) * (grad(y)+2 * lam2 *y), lam1/L)
            


        # Accélération de Nesterov
        t_new = (1 + np.sqrt(1 + 4 * t**2)) / 2
        y = x_i + ((t - 1) / t_new) * (x_i - x_old)
        t = t_new

        logs.append(x_i)
        # Critère d'arrêt
        if np.linalg.norm(x_i - x_old) < tol:
            break

    return x_i,logs



def gradient_descent(A, b,lam=0.00001, max_iter=10000, tol=1e-16):
    n = A.shape[1]
    x_i = np.zeros(n)
    grad = gradientCurrying(A, b)
    
    logs = [x_i.copy()]
    for k in range(max_iter):
        x_old = x_i.copy()
        
        # Descente de gradient
        x_i = x_i - lam * grad(x_i)
        
        logs.append(x_i.copy())
        
        # Critère d'arrêt
        if np.linalg.norm(x_i - x_old) < tol:
            break

    return x_i,logs


# Initialisation
x0 = np.zeros(A_train.shape[1])
L = np.linalg.norm(A_train.T @ A_train, 2)
t = 1.0 / L  
lam1 = 0.1
lam2 = 0.01
CURRENT_MODE = LASSO  # Choisir entre LASSO et ELASTICNET

# Exécution ISTA
x_hat_ista, logs_ista = ISTA(A_train, b_train, lam1=lam1, MODE=CURRENT_MODE, lam2=lam2, max_iter=10000, tol=1e-16)
x_hat_fista, logs_fista = fista(A_train, b_train, lam1=lam1, MODE=CURRENT_MODE, lam2=lam2, max_iter=10000, tol=1e-16)
x_hat_gd, logs_gd = gradient_descent(A_train, b_train, max_iter=10000, tol=1e-16)



def mse(A, b):
    def f(x):
        return mean_squared_error(b, A @ x)
    
    return f

mseMapFunc = mse(A_train, b_train)

mse_Per_iter_ista = [mseMapFunc(x) for x in logs_ista]
mse_Per_iter_fista = [mseMapFunc(x) for x in logs_fista]
mse_Per_iter_gd = [mseMapFunc(x) for x in logs_gd]

import matplotlib.pyplot as plt
plt.figure(figsize=(12, 6))
plt.plot(mse_Per_iter_ista, label="ISTA")
plt.plot(mse_Per_iter_fista, label="FISTA")
plt.plot(mse_Per_iter_gd, label="GD")
plt.legend()
plt.title("MSE vs Iterations")
plt.xlabel("Iterations")
plt.ylabel("MSE")
plt.yscale("log")  # Set y-axis to logarithmic scale
plt.grid()
plt.show()




# Évaluation
y_pred = A_train @ x_hat_ista
mse_test = mean_squared_error(b_train, y_pred)
print("==> Résultats de l'ISTA")
print(f"MSE : {mse_test:.4f}")

y_pred_fista = A_train @ x_hat_fista
mse_test_fista = mean_squared_error(b_train, y_pred_fista)
print("==> Résultats du FISTA")
print(f"MSE : {mse_test_fista:.4f}")


# Lasso sklearn : alpha = lam / n_samples
alpha = lam1 / A_train.shape[0]

model = Lasso(alpha=alpha, fit_intercept=False, max_iter=10000)
model.fit(A_train, b_train)

# Comparaison des résultats
y_pred_sklearn = model.predict(A_train)
mse_sklearn = mean_squared_error(b_train, y_pred_sklearn)

print("==> Résultats de comparaison")
print(f"MSE sklearn : {mse_sklearn:.4f}")
print(f"MSE ISTA : {mse_test:.4f}")
print(f"MSE FISTA : {mse_test_fista:.4f}")
print(f"Différence ISTA vs sklearn : {mse_test - mse_sklearn}")
print(f"Différence FISTA vs sklearn : {mse_test_fista - mse_sklearn}")
print(f"Différence ISTA vs FISTA : {mse_test - mse_test_fista}")
print(f"Différence nb itérations ISTA vs FISTA : {len(logs_ista) - len(logs_fista)}")

