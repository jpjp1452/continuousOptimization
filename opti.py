import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Lasso,Ridge,ElasticNet,LinearRegression
from sklearn.datasets import fetch_california_housing
from data_preprocessing import *
import matplotlib.pyplot as plt
import time




train_size = 0.9
random_state = 42
#A_train,b_train,b_mean, b_std=load_and_preprocess_Insurance_data(train_size=train_size, random_state=random_state)
#NAME_DATASET = "Insurance" 
A_train, b_train, b_mean, b_std = load_and_preprocess_Housing_data(train_size=train_size, random_state=random_state)
NAME_DATASET = "Housing" # Choose between "Insurance", "Student_Performance", "Housing", "car", "synthetic_data_1000_features"
#A_train, b_train, b_mean, b_std = load_and_preprocess_Student_Performance_data(train_size=train_size, random_state=random_state)
#NAME_DATASET = "Student_Performance"
#A_train, b_train,b_mean, b_std = load_and_preprocess_data(csv_path="synthetic_datasets/synthetic_data_1000_features.csv",targetColumn = 'target',train_size=train_size, random_state=random_state)


LASSO=0
RIDGE=1
ELASTICNET=2
LEASTSQUARES=3

# Initialization
max_iter = 4000
tolerance = 1e-4


nb_features = A_train.shape[1]
nb_samples = A_train.shape[0]
lambdaMax = np.max(np.abs(A_train.T @ b_train)) / nb_samples

lam1 = 8 
lam2 = 2

lambdaString = f"lambda1_{lam1}_lambda2_{lam2}"
m = 10 # Number of past iterations to consider in L-BFGS

CURRENT_MODE = RIDGE# Choose between LASSO, ELASTICNET, RIDGE


def soft_thresholding(x, tau):
    return np.sign(x) * np.maximum(np.abs(x) - tau, 0.0)




def backtracking_line_search(x_i,gradX_i,gFunc,sizeStep,eta=0.5,c=0.0001):
    g0 = gFunc(x_i)
    while True:
        x_trial = x_i - gradX_i * sizeStep
        g_trial = gFunc(x_trial)
        if g_trial <= g0 - c *sizeStep * np.dot(gradX_i, gradX_i):
            break
        sizeStep *= eta
    return sizeStep

def line_search_Wolfe(f, grad, x, d, alpha0=1.0, c1=1e-4, c2=0.9):
        alpha = alpha0
        fx = f(x); gdx = grad(x).T @ d
        while True:
            # if Armijo condition or curvature condition is not satisfied
            if f(x + alpha*d) > fx + c1*alpha*gdx or grad(x + alpha*d).T @ d < c2*gdx:
                alpha *= 0.5
            
            else:
                return alpha








def ISTA(functionInfo):

    grad = functionInfo["gradient"]
    g = functionInfo["g(x)"]
    x_i = functionInfo["x_0"].copy()
    L = functionInfo["Lipschitz"]
    prox = functionInfo["prox Operator"]
    StopCriterionFunction = functionInfo["stop criterion"]
    max_iter = functionInfo["max_iter"]
    tol = functionInfo["tolerance"]

    sizeStep = 1
    logs =[x_i]
    for _ in range(max_iter):
        x_old = x_i.copy()

        sizeStep = backtracking_line_search(x_i,grad(x_i),g,sizeStep)
        x_i = prox(x_i - grad(x_i) *sizeStep,sizeStep)
        logs.append(x_i)
        #stop criterion 
        if StopCriterionFunction(x_i, x_old, tol):
            break
    return x_i,logs

def FISTA(functionInfo):
    grad = functionInfo["gradient"]
    g = functionInfo["g(x)"]
    x_i = functionInfo["x_0"].copy()
    L = functionInfo["Lipschitz"]
    prox = functionInfo["prox Operator"]
    StopCriterionFunction = functionInfo["stop criterion"]
    max_iter = functionInfo["max_iter"]
    tol = functionInfo["tolerance"]


    y = x_i.copy()
    sizeStep = 1
    print("sizeStep",sizeStep)
    logs =[x_i]
    t=1
    for k in range(max_iter):
        x_old = x_i.copy()
        sizeStep = backtracking_line_search(y,grad(y),g,sizeStep)
        x_i = prox(y - ( grad(y)) *sizeStep,sizeStep)
        t_new = (1 + np.sqrt(1 + 4 * t**2)) / 2
        y = x_i + ((t - 1) / t_new) * (x_i - x_old)
        t = t_new
        logs.append(x_i)
        #stop criterion 
        if StopCriterionFunction(x_i, x_old, tol):
            break
    return x_i,logs



def subgradient_descent(INFOFunction):
    grad = INFOFunction["gradient"]
    g = INFOFunction["g(x)"]
    x_i = INFOFunction["x_0"].copy()
    L = INFOFunction["Lipschitz"]
    StopCriterionFunction = INFOFunction["stop criterion"]
    max_iter = INFOFunction["max_iter"]
    tol = INFOFunction["tolerance"]
    subgradientX_i = INFOFunction["subgradient_descent"]	

    stepsize = 1/L
    logs =[x_i]
    for k in range(1, max_iter+1):
        x_old = x_i.copy()
        #stepsize = backtracking_line_search(x_i,grad(x_i),g,stepsize)
        x_i = x_i - stepsize * (grad(x_i)+subgradientX_i(x_i))
        stepsize = (1/(L))/ (k**0.5)
        logs.append(x_i)
        #stop criterion 
        if StopCriterionFunction(x_i, x_old, tol):
            break
    return x_i,logs




def gradient_descent(INFOFunction):
    grad = INFOFunction["gradient"]
    g = INFOFunction["g(x)"]
    x_i = INFOFunction["x_0"].copy()
    L = INFOFunction["Lipschitz"]
    StopCriterionFunction = INFOFunction["stop criterion"]
    max_iter = INFOFunction["max_iter"]
    tol = INFOFunction["tolerance"]

    stepsize = 1 
    logs =[x_i]
    for _ in range(max_iter):
        x_old = x_i.copy()
        stepsize = backtracking_line_search(x_i,grad(x_i),g,stepsize)
        x_i = x_i - stepsize * grad(x_i)
        logs.append(x_i)
        #stop criterion 
        if StopCriterionFunction(x_i, x_old, tol):
            break
    return x_i,logs

def LBFGS(INFOFunction):
    grad = INFOFunction["gradient"]
    g = INFOFunction["g(x)"]
    x_i = INFOFunction["x_0"].copy()
    L = INFOFunction["Lipschitz"]
    StopCriterionFunction = INFOFunction["stop criterion"]
    max_iter = INFOFunction["max_iter"]
    tol = INFOFunction["tolerance"] 
    m = INFOFunction["m_choice"]
    n = x_i.shape[0]

    stepsize = 1.0
    logs = [x_i.copy()]
    # definition of set S, R, phi
    S = []
    Y = []
    # start for loop
    r = np.zeros(n)
    for k in range(1, max_iter):
        #step 0: compute approximation to the inverse of the Hessian matrix
        if (k==1):
            B_zeroArray = np.ones(n)
        else:
            B_zeroArray = ( S[0].T @ Y[0]) / (Y[0].T @ Y[0]) * np.ones(n)

        #step 1: compute the gradient
        q=grad(x_i)

        T = []
        phi = []
        for i in range(len(S)):
            phi.append( 1.0 / (Y[i].T @ S[i]))
            if(1.0 / (Y[i].T @ S[i]) < 0):
                raise ValueError("phi must be positive")
            T.append(phi[i] * S[i].T @ q)
            q=q-(T[i]*Y[i])
        
        for i in range(len(r)):
            r[i] = B_zeroArray[i] * q[i]

        for i in range(len(S)-1, -1, -1):
            beta = phi[i] * Y[i].T @ r
            r = r + (S[i] * (T[i] - beta))
        direction = -r
        #step 2: compute the step size
        #tk satisfies the Wolfe conditions
        stepsize = line_search_Wolfe(g, grad, x_i, direction)
        x_old= x_i.copy()
        x_i = x_i + stepsize * direction

        if (len(S) >= m):
            #removed oldest element from back of the list
            S.pop()
            Y.pop()
            #phi.pop()
        #insert new elements at the beginning of the list
        S.insert(0, x_i - x_old)
        Y.insert(0, grad(x_i) - grad(x_old))

        logs.append(x_i.copy())
        if StopCriterionFunction(x_i, x_old, tol):
            break
    return x_i, logs


def BFGS(INFOFunction):
    grad = INFOFunction["gradient"]
    g = INFOFunction["g(x)"]
    x_i = INFOFunction["x_0"].copy()
    L = INFOFunction["Lipschitz"]
    StopCriterionFunction = INFOFunction["stop criterion"]
    max_iter = INFOFunction["max_iter"]
    tol = INFOFunction["tolerance"] 
    m = INFOFunction["m_choice"]
    n = x_i.shape[0]

    stepsize = 1.0 
    logs = [x_i.copy()]
    for k in range(1, max_iter):
        #step 0: compute approximation to the inverse of the Hessian matrix

        if (k==1):
            H_inverse= np.eye(n)
        else:
            rho = 1.0/(y.T @ s)
            V   = np.eye(n) - rho * np.outer(s, y)
            H_inverse = V @ H_inverse @ V.T + rho * np.outer(s, s)
        direction = -H_inverse @ grad(x_i)

        #step 2: compute the step size
        stepsize = line_search_Wolfe(g, grad, x_i, direction)
        x_old= x_i.copy()
        x_i = x_i + stepsize * direction
        s = x_i - x_old
        y = grad(x_i) - grad(x_old)
        
        logs.append(x_i.copy())
        if StopCriterionFunction(x_i, x_old, tol):
            break
    return x_i, logs




INFO_Func= {"gradient": None,"g(x)": None, "Lipschitz": None, "prox Operator": None,"subgradient_descent": None,"x_0": None, "stop criterion": None, "tolerance": tolerance, "max_iter": max_iter, "m_choice": m}

def subgradient(x):
    z = np.sign(x)
    zeros = (x == 0)
    z[zeros] = np.random.uniform(-1, 1, size=zeros.sum())
    return z

if CURRENT_MODE == LASSO:
    # LASSO  ||Ax - b||² + λ||x||₁
    objectiveFunction = lambda x: 0.5 * np.linalg.norm(A_train@x-b_train, 2)**2 + lam1 * np.linalg.norm(x, 1)
    #gradient of the smooth part
    gradient = lambda x: A_train.T @ (A_train @ x - b_train)
    INFO_Func["gradient"] = gradient
    INFO_Func["g(x)"] = lambda x: 0.5 * np.linalg.norm(A_train@x-b_train, 2)**2
    INFO_Func["Lipschitz"] = np.linalg.norm(A_train, 2)**2
    INFO_Func["prox Operator"] = lambda x, stepsize: soft_thresholding(x, lam1 * stepsize)
    INFO_Func["subgradient_descent"] = lambda x: subgradient(x) * lam1
elif CURRENT_MODE == RIDGE:
    # RIDGE  ||Ax - b||² + λ||x||₂²
    objectiveFunction = lambda x: 0.5 * np.linalg.norm(A_train@x-b_train, 2)**2 + 0.5*lam2 * np.linalg.norm(x, 2)**2
    gradient = lambda x: A_train.T @ (A_train @ x - b_train) + lam2 * x
    INFO_Func["gradient"] = gradient
    INFO_Func["g(x)"] = lambda x: 0.5 * np.linalg.norm(A_train@x-b_train, 2)**2 + 0.5 * lam2 * np.linalg.norm(x, 2)**2
    INFO_Func["Lipschitz"] = np.linalg.norm(A_train, 2)**2
    INFO_Func["prox Operator"] = lambda x, stepsize: soft_thresholding(x, 0 * stepsize) / (1 + lam2 * stepsize)
    INFO_Func["subgradient_descent"] = lambda x: x * 0
elif CURRENT_MODE == ELASTICNET:
    # ELASTICNET ||Ax - b||² + λ₁||x||₁ + λ₂||x||₂²
    objectiveFunction = lambda x: 0.5 * np.linalg.norm(A_train@x-b_train, 2)**2 + lam1 * np.linalg.norm(x, 1) + 0.5*lam2 * np.linalg.norm(x, 2)**2
    gradient = lambda x: A_train.T @ (A_train @ x - b_train) + lam2 * x
    INFO_Func["gradient"] = gradient
    INFO_Func["g(x)"] = lambda x: 0.5 * np.linalg.norm(A_train@x-b_train, 2)**2 + lam2 * np.linalg.norm(x, 2)**2
    INFO_Func["Lipschitz"] = np.linalg.norm(A_train, 2)**2
    INFO_Func["prox Operator"] = lambda x, stepsize: soft_thresholding(x, lam1 * stepsize) / (1 + lam2 * stepsize)
    INFO_Func["subgradient_descent"] = lambda x: subgradient(x) * lam1

elif CURRENT_MODE == LEASTSQUARES:
    # LEASTSQUARES ||Ax - b||²
    objectiveFunction = lambda x: 0.5 * np.linalg.norm(A_train@x-b_train, 2)**2
    INFO_Func["gradient"] = lambda x: A_train.T @ (A_train @ x - b_train)
    INFO_Func["g(x)"] = lambda x: 0.5 * np.linalg.norm(A_train@x-b_train, 2)**2
    INFO_Func["Lipschitz"] = np.linalg.norm(A_train, 2)**2
    INFO_Func["prox Operator"] = lambda x, lam: x
    INFO_Func["subgradient_descent"] = lambda x: x * 0


def stopCriterion1(x_i, x_old, tol):
    #print(np.abs(objectiveFunction(x_i) - objectiveFunction(x_old))/max(1, np.abs(objectiveFunction(x_i))))
    return np.abs(objectiveFunction(x_i) - objectiveFunction(x_old))/max(1, np.abs(objectiveFunction(x_i))) < tol

def stopCriterion2(x_i, x_old, tol):
    #print(np.linalg.norm(INFO_Func["gradient"](x_i), 2))
    return  np.linalg.norm(INFO_Func["gradient"](x_i), 2) < tol

def stopCriterion3(x_i, x_old, tol):
    return np.max(np.abs(x_i - x_old)) < tol



#INFO_Func["x_0"] = np.random.randn(nb_features)
INFO_Func["x_0"] = np.zeros(nb_features)
INFO_Func["stop criterion"] = stopCriterion1
INFO_Func["tolerance"] = tolerance




# Exécution ISTA

time_ista = time.time()
x_hat_ista, logs_ista = ISTA(INFO_Func)
print("ISTA time:", time.time() - time_ista)
print("ISTA converged in", len(logs_ista), "iterations")

time_fista = time.time()
x_hat_fista, logs_fista = FISTA(INFO_Func)
print("FISTA time:", time.time() - time_fista)
print("FISTA converged in", len(logs_fista), "iterations")







x_hat_sgd, logs_sgd = subgradient_descent(INFO_Func)
print("Subgradient Descent converged in", len(logs_sgd), "iterations")


if CURRENT_MODE== LEASTSQUARES or CURRENT_MODE == RIDGE:
    x_hat_gd, logs_gd = gradient_descent(INFO_Func)
    print("GD converged in", len(logs_gd), "iterations")
    mse_Per_iter_gd = [objectiveFunction(x) for x in logs_gd]

    x_hat_lbfgs, logs_lbfgs = LBFGS(INFO_Func)
    print("LBFGS converged in", len(logs_lbfgs), "iterations")
    mse_Per_iter_lbfgs = [objectiveFunction(x) for x in logs_lbfgs]  


    x_hat_bfgs, logs_bfgs = BFGS(INFO_Func)
    print("BFGS converged in", len(logs_bfgs), "iterations")
    mse_Per_iter_bfgs = [objectiveFunction(x) for x in logs_bfgs]   







mse_Per_iter_ista = [objectiveFunction(x) for x in logs_ista]
mse_Per_iter_fista = [objectiveFunction(x) for x in logs_fista]
mse_Per_iter_sgd = [objectiveFunction(x) for x in logs_sgd]





plt.figure(figsize=(12, 6))
plt.plot(range(1, len(mse_Per_iter_sgd) + 1),mse_Per_iter_sgd, label="Subgradient Descent", color="blue")
plt.plot( len(mse_Per_iter_sgd) + 1, mse_Per_iter_sgd[-1], '|', color="blue", markersize=10)
plt.plot( range(1, len(mse_Per_iter_ista) + 1),mse_Per_iter_ista, label="ISTA", color="orange")
plt.plot( len(mse_Per_iter_ista) + 1, mse_Per_iter_ista[-1], '|', color="orange", markersize=10)

plt.plot( range(1, len(mse_Per_iter_fista) + 1),mse_Per_iter_fista, label="FISTA", color="red")
plt.plot( len(mse_Per_iter_fista) + 1, mse_Per_iter_fista[-1], '|', color="red", markersize=10)

if( CURRENT_MODE == LEASTSQUARES) or (CURRENT_MODE == RIDGE):
    plt.plot( range(1, len(mse_Per_iter_lbfgs) + 1),mse_Per_iter_lbfgs, label="LBFGS", color="green")
    plt.plot( len(mse_Per_iter_lbfgs) + 1, mse_Per_iter_lbfgs[-1], '|', color="green", markersize=10)
    plt.plot( range(1, len(mse_Per_iter_gd) + 1),mse_Per_iter_gd, label="GD", color="purple")
    plt.plot( len(mse_Per_iter_gd) + 1, mse_Per_iter_gd[-1], '|', color="purple", markersize=10)
    plt.plot( range(1, len(mse_Per_iter_bfgs) + 1),mse_Per_iter_bfgs, label="BFGS", color="brown")
    plt.plot( len(mse_Per_iter_bfgs) + 1, mse_Per_iter_bfgs[-1], '|', color="brown", markersize=10)

plt.legend()
plt.title("MSE vs Iterations")
plt.xlabel("Iterations")
plt.ylabel("MSE")
plt.yscale('log')
plt.xscale('log',base=10)
plt.grid()
if CURRENT_MODE == LASSO:
    plt.savefig(f"lasso_{NAME_DATASET}_{lambdaString}_{tolerance}.png")
elif CURRENT_MODE == ELASTICNET:
    plt.savefig(f"elasticnet_{NAME_DATASET}_{lambdaString}_{tolerance}.png")
elif CURRENT_MODE == RIDGE:
    plt.savefig(f"ridge_{NAME_DATASET}_{lambdaString}_{tolerance}.png")
elif CURRENT_MODE == LEASTSQUARES:
    plt.savefig(f"leastSquares_{NAME_DATASET}_{lambdaString}_{tolerance}.png")
plt.show()





# Évaluation



mse_test_ista = mean_squared_error(b_train, A_train @ x_hat_ista)
print("==> Results of ISTA")
print(f"MSE : {mse_test_ista:.4f}")

mse_test_fista = mean_squared_error(b_train, A_train @ x_hat_fista)
print("==> Results of FISTA" )
print(f"MSE : {mse_test_fista:.4f}")


mse_subgradient = mean_squared_error(b_train, A_train @ x_hat_sgd)
print("==> Results of Subgradient Descent")
print(f"MSE : {mse_subgradient:.4f}")


if( CURRENT_MODE == LEASTSQUARES) or (CURRENT_MODE == RIDGE):
    mse_test_gd = mean_squared_error(b_train, A_train @ x_hat_gd)
    print("==> Results of Gradient Descent")
    print(f"MSE : {mse_test_gd:.4f}")

    mse_test_bfgs = mean_squared_error(b_train, A_train @ x_hat_bfgs)
    print("==> Results of BFGS")
    print(f"MSE : {mse_test_bfgs:.4f}")

    mse_test_lbfgs = mean_squared_error(b_train, A_train @ x_hat_lbfgs)
    print("==> Results of LBFGS")
    print(f"MSE : {mse_test_lbfgs:.4f}")



if CURRENT_MODE == LASSO:
    # Lasso sklearn : alpha = lam / n_samples
    alpha = lam1 / A_train.shape[0]
    model = Lasso(alpha=alpha, fit_intercept=False, max_iter=max_iter, tol=tolerance)
    model.fit(A_train, b_train)
    print("converged in", model.n_iter_, "iterations")    
elif CURRENT_MODE == RIDGE:
    # Ridge sklearn : alpha = lam / n_samples
    alpha = lam2 / A_train.shape[0]
    model = Ridge(alpha=alpha, fit_intercept=False, max_iter=max_iter, tol=tolerance)
    model.fit(A_train, b_train)
    print("converged in", model.n_iter_, "iterations")    
elif CURRENT_MODE == ELASTICNET:    
    # ElasticNet sklearn : alpha = lam1 / n_samples, l1_ratio = lam1 / (lam1 + lam2)
    alpha = lam1 / A_train.shape[0]
    l1_ratio = lam1 / (lam1 + lam2)
    model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, fit_intercept=False, max_iter=max_iter, tol=tolerance)
    model.fit(A_train, b_train)
    print("converged in", model.n_iter_, "iterations")    
elif CURRENT_MODE == LEASTSQUARES:
    # Least Squares sklearn
    model = LinearRegression(fit_intercept=False, tol=tolerance)
    model.fit(A_train, b_train)
    print("converged in", model.n_iter_, "iterations")
    mse_test_lbfgs = mean_squared_error(b_train, A_train @ x_hat_lbfgs)
    



# Comparaison des résultats
mse_sklearn = mean_squared_error(b_train, model.predict(A_train))



print("==> Results comparison")
print(f"MSE sklearn : {mse_sklearn:.4f}")
print(f"MSE ISTA : {mse_test_ista:.4f}")
print(f"MSE FISTA : {mse_test_fista:.4f}")
if CURRENT_MODE == LEASTSQUARES or CURRENT_MODE == RIDGE:
    print(f"MSE BFGS : {mse_test_gd:.4f}")
    print(f"MSE LBFGS : {mse_test_bfgs:.4f}")
    print(f"MSE LBFGS : {mse_test_lbfgs:.4f}")
print(f"Difference ISTA vs sklearn : {mse_test_ista - mse_sklearn}")
print(f"Difference FISTA vs sklearn : {mse_test_fista - mse_sklearn}")
print(f"Difference Subgradient vs sklearn : {mse_subgradient - mse_sklearn}")
print(f"Difference ISTA vs Subgradient : {mse_test_ista - mse_subgradient}")
print(f"Difference ISTA vs FISTA : {mse_test_ista - mse_test_fista}")
print(f"Difference amount of iteration ISTA vs FISTA : {len(logs_ista) - len(logs_fista)}")


