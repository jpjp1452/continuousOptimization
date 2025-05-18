import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Lasso,Ridge,ElasticNet
from sklearn.datasets import fetch_california_housing
from data_preprocessing import *
import matplotlib.pyplot as plt



"""

# Soft-thresholding
            eta = 0.99
            Xplus = soft_thresholding(x_i- grad(x_i)*sizeStep, lam1 *sizeStep)
            while g(Xplus) > g(x_i) + grad(x_i).T @ (Xplus - x_i) + (1 / (2 * sizeStep)) * np.linalg.norm(Xplus - x_i)**2:
                sizeStep *= eta
                print("sizeStep",sizeStep)
                Xplus = soft_thresholding(x_i - grad(x_i) *sizeStep, lam1 *sizeStep)
            print("___________________________________________________")
            x_i = Xplus




# Charger les données California Housing
data = fetch_california_housing()

A = data.data
b = data.target

# Normalisation
scaler = StandardScaler()
A = scaler.fit_transform(A)

# Train/test split
A_train, A_test, b_train, b_test = train_test_split(A, b, test_size=0.2, random_state=42)
"""



#A_train,b_train,b_mean, b_std=load_and_preprocess_Insurance_data(test_size=0.8, random_state=42)
A_train, b_train, b_mean, b_std = load_and_preprocess_Housing_data(test_size=0.9, random_state=42)
# A_train, b_train, b_mean, b_std = load_and_preprocess_Student_Performance_data(test_size=0.8, random_state=42)



LASSO=0
RIDGE=1
ELASTICNET=2





def soft_thresholding(x, lam):
    return np.sign(x) * np.maximum(np.abs(x) - lam, 0.0)




def backtracking_line_search(x_i,gradX_i,gFunc,sizeStep,eta=0.5,c=0.0001):
    g0 = gFunc(x_i)
    while True:
        x_trial = x_i - gradX_i * sizeStep
        g_trial = gFunc(x_trial)
        if g_trial <= g0 - c *sizeStep * np.dot(gradX_i, gradX_i):
            break
        sizeStep *= eta
    return sizeStep


def ISTA(A, b,ObjectiveFunction,MODE,lam1,lam2, max_iter, tol, adaptive_step=False):
    if(MODE != LASSO and MODE != ELASTICNET and MODE != RIDGE):
        raise ValueError("MODE must be either LASSO or ELASTICNET")
    def grad(x):
        # Gradient de ||Ax - b||²
        return (A.T @ (A @ x - b))
    
    n = A.shape[1]
    x_i = np.zeros(n)
    L = np.linalg.norm(A.T @ A, 2)

    def g(x):
        # 0.5 * ||Ax - b||²
        r = A @ x - b
        return 0.5 * np.linalg.norm(r, 2)**2

    sizeStep = 1 / (L)
    logs =[x_i]
    for k in range(max_iter):
        x_i_old = x_i.copy()
        if(MODE==LASSO):
            # Soft-thresholding
            
            sizeStep = backtracking_line_search(x_i,grad(x_i),g,sizeStep)
            x_i = soft_thresholding(x_i - grad(x_i) *sizeStep, lam1 *sizeStep)

        elif(MODE==RIDGE):
            # Ridge regression
            sizeStep = backtracking_line_search(x_i,grad(x_i),g,sizeStep)
            #x_i = x_i -  sizeStep  * (grad(x_i) + lam2 * x_i)
            lam1 = 0
            x_i = soft_thresholding(x_i -  sizeStep  * (grad(x_i) + lam2 * x_i), lam1 *  sizeStep )/ (1+lam2*sizeStep)
        else:
            # ElasticNet
            sizeStep = backtracking_line_search(x_i,grad(x_i),g,sizeStep)
            x_i = soft_thresholding(x_i -  sizeStep  * (grad(x_i) + lam2 * x_i), lam1 *  sizeStep )/ (1+lam2*sizeStep)
        """
        if(adaptive_step):
            # update step
            Lk = np.linalg.norm(grad(x_i) - grad(x_i_old), 2) / np.linalg.norm(x_i - x_i_old, 2)
            if Lk > L / 2:  
                L = Lk
                t = 1.0 / L
        """
        logs.append(x_i)
        #stop criterion 
        if np.abs(ObjectiveFunction(x_i) - ObjectiveFunction(x_i_old)) < tol:
            break
        
    return x_i,logs









def fista(A, b,ObjectiveFunction,MODE,lam1,lam2, max_iter, tol):
    if(MODE != LASSO and MODE != ELASTICNET and MODE != RIDGE):
        raise ValueError("MODE must be either LASSO or ELASTICNET")
    if(max_iter <= 0):
        raise ValueError("max_iter must be positive")
    def grad(x):
        # Gradient de ||Ax - b||²
        return (A.T @ (A @ x - b))


    n = A.shape[1]
    x_i = np.zeros(n)
    y = x_i.copy()
    t = 1
    L = np.linalg.norm(A_train.T @ A_train, 2)
    


    def g(x):
        # ½||Ax - b||²
        r = A @ x - b
        return np.dot(r, r)


    def F(x):
        # objectif complet
        return g(x) + lam1 * np.linalg.norm(x, 1)

    sizeStep = 1 / L

    logs =[x_i]
    for k in range(max_iter):
        x_old = x_i.copy()
        if(MODE==LASSO):
            sizeStep = backtracking_line_search(y,grad(y),g,sizeStep)
            x_i = soft_thresholding(y- grad(y)*sizeStep, lam1 *sizeStep)
        elif(MODE==RIDGE):
            sizeStep = backtracking_line_search(y,grad(y),g,sizeStep)
            #x_i = y - sizeStep  * (grad(y)+lam2 *y)/ (1+lam2*sizeStep)
            lam1 = 0
            x_i = soft_thresholding(y - sizeStep  * (grad(y)+lam2 *y), lam1 *sizeStep ) / (1+lam2*sizeStep)

        else:
            sizeStep = backtracking_line_search(y,grad(y),g,sizeStep)
            x_i = soft_thresholding(y - sizeStep  * (grad(y)+lam2 *y), lam1 *sizeStep ) / (1+lam2*sizeStep)
            
        t_new = (1 + np.sqrt(1 + 4 * t**2)) / 2
        y = x_i + ((t - 1) / t_new) * (x_i - x_old)
        t = t_new

        logs.append(x_i)
        #stop criterion 
        if np.abs(ObjectiveFunction(x_i) - ObjectiveFunction(x_old)) < tol:
            break
    return x_i,logs
    
def subgradient_descent(A, b,ObjectiveFunction,MODE,lam1,lam2, max_iter, tol):
    if(MODE != LASSO and MODE != ELASTICNET and MODE != RIDGE):
        raise ValueError("MODE must be either LASSO or ELASTICNET")
    def grad(x):
        return(A.T @ (A @ x - b))

    n = A.shape[1]
    x_i = np.zeros(n)
    
    #eta0 = 0.001
    # calcul Lipschitz
    L = np.linalg.norm(A, ord=2)**2
    eta0 = 1.0 / L

    logs = [x_i.copy()]
    for k in range(1, max_iter+1):
        x_old = x_i.copy()

        if(MODE==LASSO):
            # LASSO
            z = np.sign(x_i)
            zeros = (x_i == 0)
            z[zeros] = np.random.uniform(-1, 1, size=zeros.sum())
            eta_k = eta0 / (k**0.75)
            # mise à jour
            x_i = x_i - eta_k * (grad(x_i) + lam1 * z)
        elif(MODE==RIDGE):
            # RIDGE
            eta_k = eta0 / (k**0.75)
            x_i = x_i - eta_k * (grad(x_i) + lam2 * x_i)
        else:
            # ELASTICNET
            eta_k = eta0 / (k**0.75)
            z = np.sign(x_i)
            zeros = (x_i == 0)
            z[zeros] = np.random.uniform(-1, 1, size=zeros.sum())
            x_i = x_i - eta_k * (grad(x_i) + lam2 * x_i + lam1 * z)


        logs.append(x_i.copy())
        if np.abs(ObjectiveFunction(x_i) - ObjectiveFunction(x_old)) < tol:
            break

    return x_i, logs

def gradient_descent(A, b,ObjectiveFunction,MODE,lam2,max_iter, tol):
    if(MODE != RIDGE):
        raise ValueError("MODE must be RIDGE")
    def grad(x):
        return(A.T @ (A @ x - b))

    n = A.shape[1]
    x_i = np.zeros(n)
    
    def g(x):
        # 0.5 * ||Ax - b||²
        r = A @ x - b
        return 0.5 * np.linalg.norm(r, 2)**2


    #eta0 = 0.001
    # calcul Lipschitz
    L = np.linalg.norm(A, ord=2)**2
    stepsize = 1.0 / L
    logs = [x_i.copy()]
    for k in range(1, max_iter+1):
        x_old = x_i.copy()
        if(MODE==RIDGE):
            # RIDGE
            stepsize = backtracking_line_search(x_i,grad(x_i)+lam2*x_i,g,stepsize)
            x_i = x_i - stepsize * (grad(x_i) + lam2 * x_i)
        
        logs.append(x_i.copy())
        if np.abs(ObjectiveFunction(x_i) - ObjectiveFunction(x_old)) < tol:
            break

    return x_i, logs






# Initialization
x0 = np.zeros(A_train.shape[1])
max_iter = 10000
tolerance = 1e-4

nb_features = A_train.shape[1]
nb_samples = A_train.shape[0]
lambdaMax = np.max(np.abs(A_train.T @ b_train)) / nb_samples

lam1 = 0.01
lam2 = 0.01
CURRENT_MODE = LASSO # Choose between LASSO, ELASTICNET, RIDGE


if CURRENT_MODE == LASSO:
    # LASSO  ||Ax - b||² + λ||x||₁
    objectiveFunction = lambda x: 0.5 * np.linalg.norm(A_train@x-b_train, 2)**2 + lam1 * np.linalg.norm(x, 1)
elif CURRENT_MODE == RIDGE:
    # RIDGE  ||Ax - b||² + λ||x||₂²
    objectiveFunction = lambda x: 0.5 * np.linalg.norm(A_train@x-b_train, 2)**2 + 0.5*lam2 * np.linalg.norm(x, 2)**2
elif CURRENT_MODE == ELASTICNET:
    # ELASTICNET ||Ax - b||² + λ₁||x||₁ + λ₂||x||₂²
    objectiveFunction = lambda x: 0.5 * np.linalg.norm(A_train@x-b_train, 2)**2 + lam1 * np.linalg.norm(x, 1) + 0.5*lam2 * np.linalg.norm(x, 2)**2



# Exécution ISTA



x_hat_ista, logs_ista = ISTA(A_train, b_train,objectiveFunction,CURRENT_MODE ,lam1,lam2, max_iter,tolerance, adaptive_step=False)
print("ISTA converged in", len(logs_ista), "iterations")
print(objectiveFunction(x_hat_ista))

x_hat_fista, logs_fista = fista(A_train, b_train,objectiveFunction,CURRENT_MODE ,lam1,lam2, max_iter,tolerance)
print("FISTA converged in", len(logs_fista), "iterations")
x_hat_sgd, logs_sgd = subgradient_descent(A_train, b_train,objectiveFunction,CURRENT_MODE ,lam1,lam2, max_iter,tolerance)
print("Subgradient Descent converged in", len(logs_sgd), "iterations")

if CURRENT_MODE == RIDGE:
    x_hat_gd, logs_gd = gradient_descent(A_train, b_train,objectiveFunction,CURRENT_MODE ,lam2, max_iter,tolerance)
    print("GD converged in", len(logs_gd), "iterations")
    mse_Per_iter_gd = [objectiveFunction(x) for x in logs_gd]
    


mse_Per_iter_ista = [objectiveFunction(x) for x in logs_ista]
mse_Per_iter_fista = [objectiveFunction(x) for x in logs_fista]
mse_Per_iter_sgd = [objectiveFunction(x) for x in logs_sgd]





plt.figure(figsize=(12, 6))
plt.plot(range(1, len(mse_Per_iter_sgd) + 1),mse_Per_iter_sgd, label="Subgradient Descent")
plt.plot( range(1, len(mse_Per_iter_ista) + 1),mse_Per_iter_ista, label="ISTA")
plt.plot( range(1, len(mse_Per_iter_fista) + 1),mse_Per_iter_fista, label="FISTA")
if( CURRENT_MODE == RIDGE):
    plt.plot( range(1, len(mse_Per_iter_gd) + 1),mse_Per_iter_gd, label="Gradient Descent")

plt.legend()
plt.title("MSE vs Iterations")
plt.xlabel("Iterations")
plt.ylabel("MSE")
plt.yscale('log')
plt.xscale('log')
plt.grid()
plt.show()
plt.savefig("i.png")




# Évaluation



mse_test_ista = mean_squared_error(b_train, A_train @ x_hat_ista)
print("==> Results ofISTA")
print(f"MSE : {mse_test_ista:.4f}")

mse_test_fista = mean_squared_error(b_train, A_train @ x_hat_fista)
print("==> Results of FISTA" )
print(f"MSE : {mse_test_fista:.4f}")


mse_subgradient = mean_squared_error(b_train, A_train @ x_hat_sgd)
print("==> Results of Subgradient Descent")
print(f"MSE : {mse_subgradient:.4f}")

if CURRENT_MODE == LASSO:
    # Lasso sklearn : alpha = lam / n_samples
    alpha = lam1 / A_train.shape[0]
    model = Lasso(alpha=alpha, fit_intercept=False, max_iter=max_iter )
    model.fit(A_train, b_train)
elif CURRENT_MODE == RIDGE:
    # Ridge sklearn : alpha = lam / n_samples
    alpha = lam2 / A_train.shape[0]
    model = Ridge(alpha=alpha, fit_intercept=False, max_iter=max_iter )
    model.fit(A_train, b_train)
elif CURRENT_MODE == ELASTICNET:    
    # ElasticNet sklearn : alpha = lam1 / n_samples, l1_ratio = lam1 / (lam1 + lam2)
    alpha = lam1 / A_train.shape[0]
    l1_ratio = lam1 / (lam1 + lam2)
    model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, fit_intercept=False, max_iter=max_iter )
    model.fit(A_train, b_train)




# Comparaison des résultats
mse_sklearn = mean_squared_error(b_train, model.predict(A_train))



print("==> Results comparison")
print(f"MSE sklearn : {mse_sklearn:.4f}")
print(f"MSE ISTA : {mse_test_ista:.4f}")
print(f"MSE FISTA : {mse_test_fista:.4f}")
print(f"Difference ISTA vs sklearn : {mse_test_ista - mse_sklearn}")
print(f"Difference FISTA vs sklearn : {mse_test_fista - mse_sklearn}")
print(f"Difference Subgradient vs sklearn : {mse_subgradient - mse_sklearn}")
print(f"Difference ISTA vs Subgradient : {mse_test_ista - mse_subgradient}")
print(f"Difference ISTA vs FISTA : {mse_test_ista - mse_test_fista}")
print(f"Difference amount of iteration ISTA vs FISTA : {len(logs_ista) - len(logs_fista)}")


