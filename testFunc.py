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




train_size = 0.5
random_state = 42

#A_train,b_train,b_mean, b_std=load_and_preprocess_Insurance_data(train_size=train_size, random_state=random_state)
A_train, b_train, b_mean, b_std = load_and_preprocess_Housing_data(train_size=train_size, random_state=random_state)
#A_train, b_train, b_mean, b_std = load_and_preprocess_Student_Performance_data(train_size=train_size, random_state=random_state)
#A_train, b_train, b_mean, b_std = load_and_preprocess_car_data(train_size=train_size, random_state=random_state)
#A_train, b_train,b_mean, b_std = load_and_preprocess_data(csv_path="synthetic_datasets/synthetic_data_1000_features.csv",targetColumn = 'target',train_size=train_size, random_state=random_state)


LASSO=0
RIDGE=1
ELASTICNET=2
LEASTSQUARES=3
X_SQUARE=4
# Initialization
max_iter = 2000
tolerance = 1e-9


nb_features = A_train.shape[1]
nb_samples = A_train.shape[0]
lambdaMax = np.max(np.abs(A_train.T @ b_train)) / nb_samples

lam1 = 0.2
lam2 = 0.03

m = 10 # Number of past iterations to consider in L-BFGS

CURRENT_MODE = X_SQUARE # Choose between LASSO, ELASTICNET, RIDGE



def soft_thresholding(x, lam):
    return np.sign(x) * np.maximum(np.abs(x) - lam, 0.0)




def backtracking_line_search(x_i,gradX_i,gFunc,sizeStep,eta=0.5,c=0.0001):
    g0 = gFunc(x_i)
    for i in range(20):
        x_trial = x_i - gradX_i * sizeStep
        g_trial = gFunc(x_trial)
        if g_trial <= g0 - c *sizeStep * np.dot(gradX_i, gradX_i):
            break
        print("sizeStep",sizeStep)
        sizeStep *= eta
    return sizeStep

def line_search_Wolfe(f, grad, x, d, alpha0=1.0, c1=1e-4, c2=0.9):
        alpha = alpha0
        fx = f(x); gdx = grad(x).T @ d
        for i in range(20):
            # if Armijo condition or curvature condition is not satisfied
            if f(x + alpha*d) > fx + c1*alpha*gdx or grad(x + alpha*d).T @ d < c2*gdx:
                alpha *= 0.5
                print("changed",alpha)
            
            else:
                return alpha
        print("Warning: Wolfe line search failed to converge")
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
        sizeStep = 1
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

    stepsize = 1 / (L)
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

def LBGFS(INFOFunction):
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


def BGFS(INFOFunction):
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




INFO= {"gradient": None,"g(x)": None, "Lipschitz": None, "prox Operator": None,"subgradient_descent": None,"x_0": None, "stop criterion": None, "tolerance": tolerance, "max_iter": max_iter, "m_choice": m}

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
    INFO["gradient"] = gradient
    INFO["g(x)"] = lambda x: 0.5 * np.linalg.norm(A_train@x-b_train, 2)**2
    INFO["Lipschitz"] = np.linalg.norm(A_train, 2)**2
    INFO["prox Operator"] = lambda x, stepsize: soft_thresholding(x, lam1 * stepsize)
    INFO["subgradient_descent"] = lambda x: subgradient(x) * lam1
elif CURRENT_MODE == RIDGE:
    # RIDGE  ||Ax - b||² + λ||x||₂²
    objectiveFunction = lambda x: 0.5 * np.linalg.norm(A_train@x-b_train, 2)**2 + 0.5*lam2 * np.linalg.norm(x, 2)**2
    gradient = lambda x: A_train.T @ (A_train @ x - b_train) + lam2 * x
    INFO["gradient"] = gradient
    INFO["g(x)"] = lambda x: 0.5 * np.linalg.norm(A_train@x-b_train, 2)**2 + 0.5 * lam2 * np.linalg.norm(x, 2)**2
    INFO["Lipschitz"] = np.linalg.norm(A_train, 2)**2
    INFO["prox Operator"] = lambda x, stepsize: soft_thresholding(x, 0 * stepsize) / (1 + lam2 * stepsize)
    INFO["subgradient_descent"] = lambda x: x * 0
elif CURRENT_MODE == ELASTICNET:
    # ELASTICNET ||Ax - b||² + λ₁||x||₁ + λ₂||x||₂²
    objectiveFunction = lambda x: 0.5 * np.linalg.norm(A_train@x-b_train, 2)**2 + lam1 * np.linalg.norm(x, 1) + 0.5*lam2 * np.linalg.norm(x, 2)**2
    gradient = lambda x: A_train.T @ (A_train @ x - b_train) + lam2 * x
    INFO["gradient"] = gradient
    INFO["g(x)"] = lambda x: 0.5 * np.linalg.norm(A_train@x-b_train, 2)**2 + lam2 * np.linalg.norm(x, 2)**2
    INFO["Lipschitz"] = np.linalg.norm(A_train, 2)**2
    INFO["prox Operator"] = lambda x, stepsize: soft_thresholding(x, lam1 * stepsize) / (1 + lam2 * stepsize)
    INFO["subgradient_descent"] = lambda x: subgradient(x) * lam1

elif CURRENT_MODE == LEASTSQUARES:
    # LEASTSQUARES ||Ax - b||²
    objectiveFunction = lambda x: 0.5 * np.linalg.norm(A_train@x-b_train, 2)**2
    INFO["gradient"] = lambda x: A_train.T @ (A_train @ x - b_train)
    INFO["g(x)"] = lambda x: 0.5 * np.linalg.norm(A_train@x-b_train, 2)**2
    INFO["Lipschitz"] = np.linalg.norm(A_train, 2)**2
    INFO["prox Operator"] = lambda x, lam: x
    INFO["subgradient_descent"] = lambda x: x * 0
elif CURRENT_MODE == X_SQUARE:
    # X_SQUARE ||Ax - b||² + λ||x||₁
    objectiveFunction = lambda x: np.log(1+np.exp(x+10))
    gradient = lambda x: (np.exp(x+10) / (1 + np.exp(x+10)))
    INFO["gradient"] = gradient
    INFO["g(x)"] = lambda x: np.log(1+np.exp(x+10)) 
    INFO["Lipschitz"] = 1
    INFO["prox Operator"] = lambda x, stepsize: x
    INFO["subgradient_descent"] = lambda x: x * 0






def stopCriterion1(x_i, x_old, tol):
    #print(np.abs(objectiveFunction(x_i) - objectiveFunction(x_old))/max(1, np.abs(objectiveFunction(x_i))))
    return np.abs(objectiveFunction(x_i) - objectiveFunction(x_old))/max(1, np.abs(objectiveFunction(x_i))) < tol

def stopCriterion2(x_i, x_old, tol):
    #print(np.linalg.norm(INFO["gradient"](x_i), 2))
    return  np.linalg.norm(INFO["gradient"](x_i), 2) < tol


#INFO["x_0"] = np.random.randn(nb_features)
INFO["x_0"] = np.array([10000])
INFO["stop criterion"] = stopCriterion1
INFO["tolerance"] = tolerance












# Exécution ISTA

time_ista = time.time()
x_hat_ista, logs_ista = ISTA(INFO)
print("ISTA time:", time.time() - time_ista)
print("ISTA converged in", len(logs_ista), "iterations")

print("x_hat_ista",x_hat_ista)
exit()
time_fista = time.time()
x_hat_fista, logs_fista = FISTA(INFO)
print("x_hat_fista",x_hat_fista)
print("FISTA time:", time.time() - time_fista)
print("FISTA converged in", len(logs_fista), "iterations")

exit()





x_hat_sgd, logs_sgd = subgradient_descent(INFO)
print("Subgradient Descent converged in", len(logs_sgd), "iterations")


if CURRENT_MODE== LEASTSQUARES or CURRENT_MODE == RIDGE:
    x_hat_gd, logs_gd = gradient_descent(INFO)
    print("GD converged in", len(logs_gd), "iterations")
    mse_Per_iter_gd = [objectiveFunction(x) for x in logs_gd]

    x_hat_lbgfs, logs_lbgfs = LBGFS(INFO)
    print("LBGFS converged in", len(logs_lbgfs), "iterations")
    mse_Per_iter_lbgfs = [objectiveFunction(x) for x in logs_lbgfs]  


    x_hat_bgfs, logs_bgfs = BGFS(INFO)
    print("BGFS converged in", len(logs_bgfs), "iterations")
    mse_Per_iter_bgfs = [objectiveFunction(x) for x in logs_bgfs]   







mse_Per_iter_ista = [objectiveFunction(x) for x in logs_ista]
mse_Per_iter_fista = [objectiveFunction(x) for x in logs_fista]
mse_Per_iter_sgd = [objectiveFunction(x) for x in logs_sgd]





plt.figure(figsize=(12, 6))
plt.plot(range(1, len(mse_Per_iter_sgd) + 1),mse_Per_iter_sgd, label="Subgradient Descent")
plt.plot( range(1, len(mse_Per_iter_ista) + 1),mse_Per_iter_ista, label="ISTA")
plt.plot( range(1, len(mse_Per_iter_fista) + 1),mse_Per_iter_fista, label="FISTA")

if( CURRENT_MODE == LEASTSQUARES) or (CURRENT_MODE == RIDGE):
    plt.plot( range(1, len(mse_Per_iter_lbgfs) + 1),mse_Per_iter_lbgfs, label="LBGFS")
    plt.plot( range(1, len(mse_Per_iter_gd) + 1),mse_Per_iter_gd, label="GD")
    plt.plot( range(1, len(mse_Per_iter_bgfs) + 1),mse_Per_iter_bgfs, label="BGFS")

plt.legend()
plt.title("MSE vs Iterations")
plt.xlabel("Iterations")
plt.ylabel("MSE")
plt.yscale('log')
plt.xscale('log',base=10)
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


if( CURRENT_MODE == LEASTSQUARES) or (CURRENT_MODE == RIDGE):
    mse_test_gd = mean_squared_error(b_train, A_train @ x_hat_gd)
    print("==> Results of Gradient Descent")
    print(f"MSE : {mse_test_gd:.4f}")

    mse_test_bgfs = mean_squared_error(b_train, A_train @ x_hat_bgfs)
    print("==> Results of BGFS")
    print(f"MSE : {mse_test_bgfs:.4f}")

    mse_test_lbgfs = mean_squared_error(b_train, A_train @ x_hat_lbgfs)
    print("==> Results of LBGFS")
    print(f"MSE : {mse_test_lbgfs:.4f}")



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
elif CURRENT_MODE == LEASTSQUARES:
    # Least Squares sklearn
    model = LinearRegression(fit_intercept=False)
    model.fit(A_train, b_train)
    mse_test_lbgfs = mean_squared_error(b_train, A_train @ x_hat_lbgfs)
    



# Comparaison des résultats
mse_sklearn = mean_squared_error(b_train, model.predict(A_train))



print("==> Results comparison")
print(f"MSE sklearn : {mse_sklearn:.4f}")
print(f"MSE ISTA : {mse_test_ista:.4f}")
print(f"MSE FISTA : {mse_test_fista:.4f}")
if CURRENT_MODE == LEASTSQUARES or CURRENT_MODE == RIDGE:
    print(f"MSE BGFS : {mse_test_gd:.4f}")
    print(f"MSE LBGFS : {mse_test_bgfs:.4f}")
    print(f"MSE LBGFS : {mse_test_lbgfs:.4f}")
print(f"Difference ISTA vs sklearn : {mse_test_ista - mse_sklearn}")
print(f"Difference FISTA vs sklearn : {mse_test_fista - mse_sklearn}")
print(f"Difference Subgradient vs sklearn : {mse_subgradient - mse_sklearn}")
print(f"Difference ISTA vs Subgradient : {mse_test_ista - mse_subgradient}")
print(f"Difference ISTA vs FISTA : {mse_test_ista - mse_test_fista}")
print(f"Difference amount of iteration ISTA vs FISTA : {len(logs_ista) - len(logs_fista)}")


