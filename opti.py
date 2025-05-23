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
#proximal operator for the L1 norm
def soft_thresholding(x, tau):
    return np.sign(x) * np.maximum(np.abs(x) - tau, 0.0)



def subgradient(x):
    z = np.sign(x)
    zeros = (x == 0)
    z[zeros] = np.random.uniform(-1, 1, size=zeros.sum())
    return z

def backtracking_line_search(x_i,gradX_i,gFunc,sizeStep,eta=0.5,c=0.0001):
    g0 = gFunc(x_i)
    while True:
        x_trial = x_i - gradX_i * sizeStep
        g_trial = gFunc(x_trial)
        # Armijo condition
        if g_trial <= g0 - c *sizeStep * np.dot(gradX_i, gradX_i):
            break
        sizeStep *= eta #update the step size
    return sizeStep

def line_search_Wolfe(f, grad, x, d, alpha0=1.0, c1=1e-4, c2=0.9):
        alpha = alpha0
        fx = f(x); gdx = grad(x).T @ d
        while True:
            # if Armijo condition or curvature condition is not satisfied
            if f(x + alpha*d) > fx + c1*alpha*gdx or grad(x + alpha*d).T @ d < c2*gdx:
                alpha *= 0.5 # reduce step size
            else:
                return alpha



def ISTA(functionInfo):
    grad = functionInfo["gradient"]          # gradient ∇f(x) for smooth part
    g = functionInfo["g(x)"]                 # smooth part of f(x)
    x_i = functionInfo["x_0"].copy()         # initial point x₀
    L = functionInfo["Lipschitz"]            # Lipschitz constant for ∇f
    prox = functionInfo["prox Operator"]     # proximal operator for non-smooth term
    StopCriterionFunction = functionInfo["stop criterion"]
    max_iter = functionInfo["max_iter"]      # max iterations
    tol = functionInfo["tolerance"]          # tolerance for stopping

    sizeStep = 1                                # initial step size (will adjust via line search)
    logs = [x_i.copy()]                         # record iterates

    for _ in range(max_iter):
        x_old = x_i.copy()                     # store previous iterate
        # Step size selection via backtracking line search on smooth part
        sizeStep = backtracking_line_search(x_i, grad(x_i), g, sizeStep)
        # Proximal gradient update: x_{k+1} = prox_{tk h()}(x_k - t ∇f(x_k))
        x_i = prox(x_i - grad(x_i) * sizeStep, sizeStep)
        logs.append(x_i.copy())                # log new iterate

        if StopCriterionFunction(x_i, x_old, tol):
            break

    return x_i, logs                            # return solution and iterate history


def FISTA(functionInfo):
    grad = functionInfo["gradient"]           # gradient ∇f(x) for smooth part
    g = functionInfo["g(x)"]                 # smooth part of f(x)
    x_i = functionInfo["x_0"].copy()         # initial point x₀
    L = functionInfo["Lipschitz"]            # Lipschitz constant for ∇f
    prox = functionInfo["prox Operator"]     # proximal operator
    StopCriterionFunction = functionInfo["stop criterion"]
    max_iter = functionInfo["max_iter"]      # max iterations
    tol = functionInfo["tolerance"]          # tolerance

    # Initialize momentum term
    y = x_i.copy()                              # initialize y
    sizeStep = 1                                # initial step size
    t = 1                                       # initial t
    logs = [x_i.copy()]                         # record iterates

    for k in range(max_iter):
        x_old = x_i.copy()                     # store previous iterate
        # Line search to choose step size
        sizeStep = backtracking_line_search(y, grad(y), g, sizeStep)
        # Proximal update at y_k: x_{k+1} = prox_{t g}(y_k - t ∇f(y_k))
        x_i = prox(y - grad(y) * sizeStep, sizeStep)
        # Update momentum parameter t_{k+1} = (1 + √(1+4 t_k²)) / 2
        t_new = (1 + np.sqrt(1 + 4 * t**2)) / 2
        # Update auxiliary sequence y_{k+1} = x_{k+1} + ((t_k-1)/t_{k+1})(x_{k+1}-x_k)
        y = x_i + ((t - 1) / t_new) * (x_i - x_old)
        t = t_new                              # set t = t_{k+1}
        logs.append(x_i.copy())                # log new iterate

        if StopCriterionFunction(x_i, x_old, tol):
            break

    return x_i, logs                            # return solution and iterate history


def subgradient_descent(INFOFunction):
    grad = INFOFunction["gradient"]          # gradient ∇f(x) for differentiable part
    x_i = INFOFunction["x_0"].copy()         # initial point x₀
    L = INFOFunction["Lipschitz"]            # Lipschitz constant for gradient part
    StopCriterionFunction = INFOFunction["stop criterion"]
    max_iter = INFOFunction["max_iter"]      # maximum iterations
    tol = INFOFunction["tolerance"]          # tolerance for stopping
    subgradientX_i = INFOFunction["subgradient_descent"]  # subgradient operator for non-smooth part

    
    stepsize = 1 / L                           # Initial step size: 1/L ensures convergence
    logs = [x_i.copy()]                        # record iterates

    for k in range(1, max_iter + 1):
        x_old = x_i.copy()                     # store old iterate for stopping check

        # Compute update: gradient + subgradient combined
        # x_{k+1} = x_k - α_k (∇f(x_k) + g_sub(x_k))
        x_i = x_i - stepsize * (grad(x_i) + subgradientX_i(x_i))

        # Decrease step size over iterations: α_k = (1/L) / sqrt(k)
        stepsize = (1 / L) / (k ** 0.5)

        logs.append(x_i.copy())               # log new iterate

        if StopCriterionFunction(x_i, x_old, tol):
            break

    return x_i, logs                          # return solution and iterate history




def gradient_descent(INFOFunction):
    grad = INFOFunction["gradient"]          # gradient function ∇f(x)
    g = INFOFunction["g(x)"]                 # smooth function f(x), for line search
    x_i = INFOFunction["x_0"].copy()         # initial iterate x₀
    L = INFOFunction["Lipschitz"]            # Lipschitz constant 
    StopCriterionFunction = INFOFunction["stop criterion"]
    max_iter = INFOFunction["max_iter"]      # maximum number of iterations
    tol = INFOFunction["tolerance"]          # tolerance for stopping criterion

    stepsize = 1                               # initial step length 
    logs = [x_i.copy()]                        # store iterates for logging

    for _ in range(max_iter):
        # Record current point
        x_old = x_i.copy()

        # Step size selection via backtracking line search
        stepsize = backtracking_line_search(x_i, grad(x_i), g, stepsize)

        # Gradient descent update: x_{k+1} = x_k - t_k ∇f(x_k)
        x_i = x_i - stepsize * grad(x_i)

        # Log new iterate
        logs.append(x_i.copy())

        # Stopping criterion
        if StopCriterionFunction(x_i, x_old, tol):
            break

    return x_i, logs  

def LBFGS(INFOFunction):
    grad = INFOFunction["gradient"]  # gradient function ∇f(x)
    g = INFOFunction["g(x)"]        # smooth part of f(x),which is f(x) in the case of L-BFGS
    x_i = INFOFunction["x_0"].copy()  # initial iterate x₀
    L = INFOFunction["Lipschitz"]  # Lipschitz constant (if used)
    StopCriterionFunction = INFOFunction["stop criterion"]
    max_iter = INFOFunction["max_iter"]
    tol = INFOFunction["tolerance"] 
    m = INFOFunction["m_choice"]     # memory parameter m (number of correction pairs)
    n = x_i.shape[0]

    stepsize = 1.0
    logs = [x_i.copy()]
    # S and Y store last m correction pairs (s_k = x_{k+1}-x_k, y_k = ∇f_{k+1}-∇f_k)
    S = []  # list of s vectors
    Y = []  # list of y vectors
    r = np.zeros(n)

    for k in range(1, max_iter):
        # Step 0: Initialize H₀, the initial Hessian approximation
        if k == 1:
            # At first iteration, use identity scaled: H₀ = I
            B_zeroArray = np.ones(n)  # diagonal entries of H₀
        else:
            # Use scaling based on most recent (sᵀy)/(yᵀy)
            # H₀ = γ I, where γ = (s₀ᵀy₀)/(y₀ᵀy₀)
            B_zeroArray = (S[0].T @ Y[0]) / (Y[0].T @ Y[0]) * np.ones(n)

        # Step 1: Compute two-loop recursion to get r ≈ H_k ∇f(x_k)
        q = grad(x_i)  # current gradient ∇f_k

        T = []   # list to store α_i = (s_iᵀ q) / (y_iᵀ s_i) scalars
        phi = [] # list to store ρ_i = 1/(y_iᵀ s_i)
        # First loop: go from most recent to oldest
        for i in range(len(S)):
            # ρ_i = 1/(y_iᵀ s_i)
            phi_i = 1.0 / (Y[i].T @ S[i])
            phi.append(phi_i)
            # α_i = ρ_i (s_iᵀ q)
            T_i = phi_i * (S[i].T @ q)
            T.append(T_i)
            # q ← q - α_i y_i
            q = q - T_i * Y[i]

        # Multiply by initial matrix H₀: r = H₀ q; for diagonal H₀, elementwise
        r = B_zeroArray * q

        # Second loop: go from oldest to most recent
        for i in range(len(S) - 1, -1, -1):
            # β_i = ρ_i (y_iᵀ r)
            beta = phi[i] * (Y[i].T @ r)
            # r ← r + s_i (α_i - β_i)
            r = r + S[i] * (T[i] - beta)

        # Search direction p_k = - H_k ∇f_k ≈ -r
        direction = -r

        # Step 2: Line search to satisfy Wolfe conditions; find t_k
        stepsize = line_search_Wolfe(g, grad, x_i, direction)
        x_old = x_i.copy()
        # Update iterate: x_{k+1} = x_k + t_k p_k
        x_i = x_i + stepsize * direction

        # Update correction pairs
        if len(S) >= m:
            # Remove oldest pair if beyond memory
            S.pop()
            Y.pop()
        # Insert new s and y at front
        S.insert(0, x_i - x_old)                  # s_k
        Y.insert(0, grad(x_i) - grad(x_old))      # y_k

        logs.append(x_i.copy())
        # Check stopping criterion
        if StopCriterionFunction(x_i, x_old, tol):
            break

    return x_i, logs



def BFGS(INFOFunction):
    grad = INFOFunction["gradient"]    # gradient function ∇f(x)
    g = INFOFunction["g(x)"]          # smooth function f(x), which is f(x) in the case of BFGS
    x_i = INFOFunction["x_0"].copy()  # initial point x₀
    L = INFOFunction["Lipschitz"]     # Lipschitz constant (optional)
    StopCriterionFunction = INFOFunction["stop criterion"]
    max_iter = INFOFunction["max_iter"]
    tol = INFOFunction["tolerance"]
    n = x_i.shape[0]                    # features size

    stepsize = 1.0
    logs = [x_i.copy()]                 # store iterates for logging

    for k in range(1, max_iter):
        # Step 0: Build or update inverse Hessian approximation H_k
        if k == 1:
            # At first iteration, initialize H₀ = I (identity matrix)
            H_inverse = np.eye(n)
        else:
            # Compute curvature pair quantities
            # ρ = 1 / (yᵀ s)
            rho = 1.0 / (y.T @ s)
            # V = I - ρ s yᵀ
            V = np.eye(n) - rho * np.outer(s, y)
            # BFGS update: H_{k+1} = V H_k Vᵀ + ρ s sᵀ
            H_inverse = V @ H_inverse @ V.T + rho * np.outer(s, s)

        # Step 1: Compute search direction p_k = -H_k ∇f_k
        direction = -H_inverse @ grad(x_i)

        # Step 2: Perform Wolfe line search to find step length t_k
        stepsize = line_search_Wolfe(g, grad, x_i, direction)
        x_old = x_i.copy()
        # Update iterate: x_{k+1} = x_k + t_k p_k
        x_i = x_i + stepsize * direction

        # Compute displacement and gradient difference for next update
        s = x_i - x_old                 # s_k = x_{k+1} - x_k
        y = grad(x_i) - grad(x_old)     # y_k = ∇f_{k+1} - ∇f_k

        logs.append(x_i.copy())         # record new iterate

        # Check stopping criterion: e.g., ||x_{k+1}-x_k|| < tol or gradient norm small
        if StopCriterionFunction(x_i, x_old, tol):
            break

    return x_i, logs                    # return solution and log of iterates



train_size = 0.9
random_state = 42
select_dataset = 2



if select_dataset == 1:
    A_train,b_train,b_mean, b_std=load_and_preprocess_Insurance_data(train_size=train_size, random_state=random_state)
    NAME_DATASET = "Insurance" 
elif select_dataset == 2:
    print("Loading California Housing dataset...")
    A_train, b_train, b_mean, b_std = load_and_preprocess_Housing_data(train_size=train_size, random_state=random_state)
    NAME_DATASET = "Housing"
elif select_dataset == 3:    
    A_train, b_train, b_mean, b_std = load_and_preprocess_Student_Performance_data(train_size=train_size, random_state=random_state)
    NAME_DATASET = "Student_Performance"
else:
    A_train, b_train, b_mean, b_std = load_and_preprocess_data(csv_path="synthetic_datasets/synthetic_data_1000_features.csv",targetColumn = 'target',train_size=train_size, random_state=random_state)
    NAME_DATASET = "Synthetic_1000_features"






LASSO=0
RIDGE=1
ELASTICNET=2
LEASTSQUARES=3

# Initialization
max_iter = 8000
tolerance = 1e-9

nb_features = A_train.shape[1]
nb_samples = A_train.shape[0]

lam1 = 0.1 
lam2 = 0.01

lambdaString = f"lambda1_{lam1}_lambda2_{lam2}"
m = 10 # Number of past iterations to consider in L-BFGS

CURRENT_MODE = RIDGE# Choose between LASSO, ELASTICNET, RIDGE, LEASTSQUARES







#Dictionary to store all the information needed about the function which is used in the algorithms

INFO_Func= {"gradient": None,"g(x)": None, "Lipschitz": None, "prox Operator": None,"subgradient_descent": None,"x_0": None, "stop criterion": None, "tolerance": tolerance, "max_iter": max_iter, "m_choice": m}


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
    return np.abs(objectiveFunction(x_i) - objectiveFunction(x_old))/max(1, np.abs(objectiveFunction(x_i))) < tol

def stopCriterion2(x_i, x_old, tol):
    return  np.linalg.norm(INFO_Func["gradient"](x_i), 2) < tol

def stopCriterion3(x_i, x_old, tol):
    return np.max(np.abs(x_i - x_old)) < tol


INFO_Func["x_0"] = np.zeros(nb_features)
INFO_Func["stop criterion"] = stopCriterion1
INFO_Func["tolerance"] = tolerance




x_hat_ista, logs_ista = ISTA(INFO_Func)
mse_Per_iter_ista = [objectiveFunction(x) for x in logs_ista]
mse__ista = mean_squared_error(b_train, A_train @ x_hat_ista)

print("ISTA converged in", len(logs_ista), "iterations")

x_hat_fista, logs_fista = FISTA(INFO_Func)
mse_Per_iter_fista = [objectiveFunction(x) for x in logs_fista]
mse__fista = mean_squared_error(b_train, A_train @ x_hat_fista)
print("FISTA converged in", len(logs_fista), "iterations")

x_hat_sgd, logs_sgd = subgradient_descent(INFO_Func)
mse_Per_iter_sgd = [objectiveFunction(x) for x in logs_sgd]
mse__sgd = mean_squared_error(b_train, A_train @ x_hat_sgd)
print("Subgradient Descent converged in", len(logs_sgd), "iterations")


if CURRENT_MODE== LEASTSQUARES or CURRENT_MODE == RIDGE:
    x_hat_gd, logs_gd = gradient_descent(INFO_Func)
    mse_Per_iter_gd = [objectiveFunction(x) for x in logs_gd]
    mse__gd = mean_squared_error(b_train, A_train @ x_hat_gd)
    print("GD converged in", len(logs_gd), "iterations")

    x_hat_lbfgs, logs_lbfgs = LBFGS(INFO_Func)
    mse_Per_iter_lbfgs = [objectiveFunction(x) for x in logs_lbfgs] 
    mse__lbfgs = mean_squared_error(b_train, A_train @ x_hat_lbfgs) 
    print("LBFGS converged in", len(logs_lbfgs), "iterations")


    x_hat_bfgs, logs_bfgs = BFGS(INFO_Func)
    mse_Per_iter_bfgs = [objectiveFunction(x) for x in logs_bfgs]   
    mse__bfgs = mean_squared_error(b_train, A_train @ x_hat_bfgs)
    print("BFGS converged in", len(logs_bfgs), "iterations")












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
    plt.savefig(f"plots/lasso_{NAME_DATASET}_{lambdaString}_{tolerance}.png")
elif CURRENT_MODE == ELASTICNET:
    plt.savefig(f"plots/elasticnet_{NAME_DATASET}_{lambdaString}_{tolerance}.png")
elif CURRENT_MODE == RIDGE:
    plt.savefig(f"plots/ridge_{NAME_DATASET}_{lambdaString}_{tolerance}.png")
elif CURRENT_MODE == LEASTSQUARES:
    plt.savefig(f"plots/leastSquares_{NAME_DATASET}_{lambdaString}_{tolerance}.png")
plt.show()






if CURRENT_MODE == LASSO:
    alpha = lam1 / A_train.shape[0]
    model = Lasso(alpha=alpha, fit_intercept=False, max_iter=max_iter, tol=tolerance)
    model.fit(A_train, b_train)
    print("converged in", model.n_iter_, "iterations")    
elif CURRENT_MODE == RIDGE:
    alpha = lam2 / A_train.shape[0]
    model = Ridge(alpha=alpha, fit_intercept=False, max_iter=max_iter, tol=tolerance)
    model.fit(A_train, b_train)
    print("converged in", model.n_iter_, "iterations")    
elif CURRENT_MODE == ELASTICNET:    
    alpha = lam1 / A_train.shape[0]
    l1_ratio = lam1 / (lam1 + lam2)
    model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, fit_intercept=False, max_iter=max_iter, tol=tolerance)
    model.fit(A_train, b_train)
    print("converged in", model.n_iter_, "iterations")    
elif CURRENT_MODE == LEASTSQUARES:
    model = LinearRegression(fit_intercept=False, tol=tolerance)
    model.fit(A_train, b_train)
    print("converged in", model.n_iter_, "iterations")
    mse__lbfgs = mean_squared_error(b_train, A_train @ x_hat_lbfgs)
    



mse_sklearn = mean_squared_error(b_train, model.predict(A_train))



print("==> Results comparison")
print(f"MSE sklearn : {mse_sklearn}")
print(f"MSE ISTA : {mse__ista}")
print(f"MSE FISTA : {mse__fista}")
if CURRENT_MODE == LEASTSQUARES or CURRENT_MODE == RIDGE:
    print(f"MSE BFGS : {mse__gd}")
    print(f"MSE LBFGS : {mse__bfgs}")
    print(f"MSE LBFGS : {mse__lbfgs}")

print(f"Comparing MSE of each algorithm with sklearn MSE")
print(f"ISTA  MSE - sklearn MSE : {mse__ista - mse_sklearn}")
print(f"FISTA MSE - sklearn MSE : {mse__fista - mse_sklearn}")
print(f"SD    MSE - sklearn MSE : {mse__sgd - mse_sklearn}")
if CURRENT_MODE == LEASTSQUARES or CURRENT_MODE == RIDGE:
    print(f"BFGS  MSE - sklearn MSE : {mse__gd - mse_sklearn}")
    print(f"LBFGS MSE - sklearn MSE : {mse__lbfgs - mse_sklearn}")
    print(f"GD    MSE - sklearn MSE : {mse__gd - mse_sklearn}")

