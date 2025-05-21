import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Lasso,Ridge,ElasticNet,LinearRegression
from sklearn.datasets import fetch_california_housing
from data_preprocessing import *
import matplotlib.pyplot as plt
from scipy import sparse




train_size = 0.9
random_state = 42

#A_train,b_train,b_mean, b_std=load_and_preprocess_Insurance_data(train_size=train_size, random_state=random_state)
#A_train, b_train, b_mean, b_std = load_and_preprocess_Housing_data(train_size=train_size, random_state=random_state)
# A_train, b_train, b_mean, b_std = load_and_preprocess_Student_Performance_data(train_size=train_size, random_state=random_state)
#A_train, b_train, b_mean, b_std = load_and_preprocess_genes_data(train_size=train_size, random_state=random_state)
A_train, b_train,b_mean, b_std = load_and_preprocess_data(csv_path="synthetic_datasets/synthetic_data_1000_features.csv",targetColumn = 'target',train_size=train_size, random_state=random_state)


LASSO=0
RIDGE=1
ELASTICNET=2
LEASTSQUARES=3





def soft_thresholding(x, lam):
    return np.sign(x) * np.maximum(np.abs(x) - lam, 0.0)




def backtracking_line_search(x_i,gradX_i,gFunc,sizeStep,eta=0.5,c=0.0001):
    g0 = gFunc(x_i)
    while True:
        x_trial = x_i - gradX_i * sizeStep
        g_trial = gFunc(x_trial)
        if g_trial <= g0 - c *sizeStep * np.dot(gradX_i, gradX_i):
            break
        print("sizeStep",sizeStep)
        sizeStep *= eta
    return sizeStep

def line_search_Wolfe(f, grad, x, d, alpha0=1.0, c1=1e-4, c2=0.9):
        α = alpha0
        fx = f(x); gdx = grad(x).T @ d
        while True:
            # if Armijo condition or curvature condition is not satisfied
            if f(x + α*d) > fx + c1*α*gdx or grad(x + α*d).T @ d < c2*gdx:
                α *= 0.5
            
            else:
                return α



def ISTA(A, b,StopCriterionFunction,MODE,lam1,lam2, max_iter, tol, adaptive_step=False):
    if(MODE != LASSO and MODE != ELASTICNET and MODE != RIDGE and MODE != LEASTSQUARES):
        raise ValueError("MODE must be either LASSO or ELASTICNET")
    def grad(x):
        # Gradient de ||Ax - b||²
        return (A.T @ (A @ x - b))
    
    n = A.shape[1]
    x_i = np.zeros(n)
    L = np.linalg.norm(A, 2)**2

    def g(x):
        # 0.5 * ||Ax - b||²
        r = A @ x - b
        return 0.5 * np.linalg.norm(r, 2)**2

    sizeStep = 1 / (L)
    logs =[x_i]
    k=0
    for k in range(max_iter):
        x_old = x_i.copy()
        if(MODE==LASSO):
            # Soft-thresholding
            
            sizeStep = backtracking_line_search(x_i,grad(x_i),g,sizeStep)
            x_i = soft_thresholding(x_i - grad(x_i) *sizeStep, lam1 *sizeStep)

        elif(MODE==RIDGE):
            # Ridge regression
            x_i = x_i -  sizeStep  * (grad(x_i) + lam2 * x_i)
            lam1 = 0
            x_i = soft_thresholding(x_i -  sizeStep  * (grad(x_i) + lam2 * x_i), lam1 *  sizeStep )/ (1+lam2*sizeStep)
        elif(MODE==ELASTICNET):
            # ElasticNet
            sizeStep = backtracking_line_search(x_i,grad(x_i),g,sizeStep)
            x_i = soft_thresholding(x_i -  sizeStep  * (grad(x_i) + lam2 * x_i), lam1 *  sizeStep )/ (1+lam2*sizeStep)
        else:
            x_i = x_i - sizeStep * grad(x_i)
        logs.append(x_i)
        #stop criterion 
        if StopCriterionFunction(x_i, x_old, tol):
            break
    print
    return x_i,logs









def fista(A, b,StopCriterionFunction,MODE,lam1,lam2, max_iter, tol):
    if(MODE != LASSO and MODE != ELASTICNET and MODE != RIDGE and MODE != LEASTSQUARES):
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
    L = np.linalg.norm(A, 2)**2
    
    def g(x):
        # ½||Ax - b||²
        r = A @ x - b
        return 0.5* np.dot(r, r)

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
        elif(MODE==ELASTICNET):
            sizeStep = backtracking_line_search(y,grad(y),g,sizeStep)
            x_i = soft_thresholding(y - sizeStep  * (grad(y)+lam2 *y), lam1 *sizeStep ) / (1+lam2*sizeStep)
        else:
            x_i = y - sizeStep * grad(y)
        t_new = (1 + np.sqrt(1 + 4 * t**2)) / 2
        y = x_i + ((t - 1) / t_new) * (x_i - x_old)
        t = t_new

        logs.append(x_i)
        #stop criterion 
        if StopCriterionFunction(x_i, x_old, tol):
            break
    return x_i,logs
    
def subgradient_descent(A, b,StopCriterionFunction,MODE,lam1,lam2, max_iter, tol):
    if(MODE != LASSO and MODE != ELASTICNET and MODE != RIDGE and MODE != LEASTSQUARES):
        raise ValueError("MODE must be either LASSO or ELASTICNET")
        
    if(MODE == RIDGE):
        def grad(x):
            return(A.T @ (A @ x - b)) + lam2 * x
        def g(x):
            # 0.5 * ||Ax - b||²
            r = A @ x - b
            return 0.5 * np.linalg.norm(r, 2)**2 + 0.5 * lam2 * np.linalg.norm(x, 2)**2
    else:
        def grad(x):
            return(A.T @ (A @ x - b))
        def g(x):
                # 0.5 * ||Ax - b||²
                r = A @ x - b
                return 0.5 * np.linalg.norm(r, 2)**2

    n = A.shape[1]
    x_i = np.zeros(n)
    
    # calculatate Lipschitz
    L = np.linalg.norm(A, 2)**2
    stepsize = 1.0 / L


    logs = [x_i.copy()]
    for k in range(1, max_iter+1):
        x_old = x_i.copy()

        if(MODE==LASSO):
            # LASSO
            z = np.sign(x_i)
            zeros = (x_i == 0)
            z[zeros] = np.random.uniform(-1, 1, size=zeros.sum())
            stepsize = backtracking_line_search(x_i,grad(x_i),g,stepsize)
            # mise à jour
            x_i = x_i - stepsize * (grad(x_i) + lam1 * z)
        elif(MODE==RIDGE):
            # RIDGE
            stepsize = backtracking_line_search(x_i,grad(x_i),g,stepsize)
            x_i = x_i - stepsize * (grad(x_i) + lam2 * x_i)
        elif(MODE==ELASTICNET):
            # ELASTICNET
            z = np.sign(x_i)
            zeros = (x_i == 0)
            z[zeros] = np.random.uniform(-1, 1, size=zeros.sum())
            stepsize = backtracking_line_search(x_i,grad(x_i),g,stepsize)
            x_i = x_i - stepsize * (grad(x_i) + lam2 * x_i + lam1 * z)
        else:
            # LEASTSQUARES
            stepsize = backtracking_line_search(x_i,grad(x_i),g,stepsize)
            x_i = x_i - stepsize * grad(x_i)
        
        logs.append(x_i.copy())
        if StopCriterionFunction(x_i, x_old, tol):
            break

    return x_i, logs

def gradient_descent(A, b,StopCriterionFunction,MODE,lam2,max_iter, tol):
    if(MODE !=LEASTSQUARES and MODE != RIDGE):
        raise ValueError("MODE must be LEASTSQUARES")
    if(MODE == LEASTSQUARES):
        def grad(x):
            return(A.T @ (A @ x - b))
        def g(x):
                # 0.5 * ||Ax - b||²
                r = A @ x - b
                return 0.5 * np.linalg.norm(r, 2)**2
    elif(MODE == RIDGE):
        def grad(x):
            return(A.T @ (A @ x - b)) + lam2 * x
        def g(x):
            # 0.5 * ||Ax - b||²
            r = A @ x - b
            return 0.5 * np.linalg.norm(r, 2)**2 + 0.5 * lam2 * np.linalg.norm(x, 2)**2

    n = A.shape[1]
    x_i = np.zeros(n)
    

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
        else:
            # LEASTSQUARES
            stepsize = backtracking_line_search(x_i,grad(x_i),g,stepsize)
            x_i = x_i - stepsize * grad(x_i)
        
        logs.append(x_i.copy())
        if StopCriterionFunction(x_i, x_old, tol):
            break

    return x_i, logs

def LBGFS (A, b,StopCriterionFunction,MODE,lam1,lam2, max_iter, tol, m_choice):
    if(MODE !=LEASTSQUARES and MODE != RIDGE):
        raise ValueError("MODE must be LEASTSQUARES")
    if(MODE == LEASTSQUARES):
        def grad(x):
            return(A.T @ (A @ x - b))
        def g(x):
                # 0.5 * ||Ax - b||²
                r = A @ x - b
                return 0.5 * np.linalg.norm(r, 2)**2
    elif(MODE == RIDGE):
        def grad(x):
            return(A.T @ (A @ x - b)) + lam2 * x
        def g(x):
            # 0.5 * ||Ax - b||²
            r = A @ x - b
            return 0.5 * np.linalg.norm(r, 2)**2 + 0.5 * lam2 * np.linalg.norm(x, 2)**2
    
    n = A.shape[1]
    x_i = np.zeros(n)
    x_old = np.zeros(n)
    
    L = np.linalg.norm(A, 2)**2
    stepsize = 1.0/L #The init step influences a lot the convergence plot, try with 320 instead of L as example
    logs = [x_i.copy()]
    #definitio m parameter
    m=m_choice
    # definition of set S, R, phi
    S = []
    Y = []

    
    # start for loop
    for k in range(1, max_iter):
        
        #step 0: compute approximation to the inverse of the Hessian matrix
        if (k==1):
            B_zero= np.eye(n)
        else:
            #B_zero=( S[0].T @ Y[0]) / (Y[0].T @ Y[0]) * np.eye(n)
            B_zero=( S[0].T @ Y[0]) / (Y[0].T @ Y[0])*sparse.eye(n, format='csr')

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
        r=B_zero @ q

        for i in range(len(S)-1, -1, -1):
            beta = phi[i] * Y[i].T @ r
            r = r + (S[i] * (T[i] - beta))
        direction = -r
        #step 2: compute the step size
        #tk satisfies the Wolfe conditions
        stepsize = line_search_Wolfe(g, grad, x_i, direction)
        x_old= x_i.copy()
        x_i = x_i + stepsize * direction

        s = x_i - x_old
        y = grad(x_i) - grad(x_old)

        if (len(S) >= m):
            #removed oldest element from back of the list
            S.pop()
            Y.pop()
            #phi.pop()
        #insert new elements at the beginning of the list
        S.insert(0, x_i - x_old)
        Y.insert(0, grad(x_i) - grad(x_old))
        #phi.insert(0, 1.0 / (Y[0].T @ S[0]))
        logs.append(x_i.copy())
        if StopCriterionFunction(x_i, x_old, tol):
            break

    return x_i, logs


def BGFS(A, b,StopCriterionFunction,MODE,lam1,lam2, max_iter, tol, m_choice):
    if(MODE !=LEASTSQUARES and MODE != RIDGE):
        raise ValueError("MODE must be LEASTSQUARES")
    if(MODE == LEASTSQUARES):
        def grad(x):
            return(A.T @ (A @ x - b))
        def g(x):
            # 0.5 * ||Ax - b||²
            r = A @ x - b
            return 0.5 * np.linalg.norm(r, 2)**2
    elif(MODE == RIDGE):
        def grad(x):
            return(A.T @ (A @ x - b)) + lam2 * x
        def g(x):
            # 0.5 * ||Ax - b||²
            r = A @ x - b
            return 0.5 * np.linalg.norm(r, 2)**2 + 0.5 * lam2 * np.linalg.norm(x, 2)**2
    
    def grad(x):
        return(A.T @ (A @ x - b))
    def g(x):
            # 0.5 * ||Ax - b||²
            r = A @ x - b
            return 0.5 * np.linalg.norm(r, 2)**2
    n = A.shape[1]
    x_i = np.zeros(n)
    x_old = np.zeros(n)
    
    L = np.linalg.norm(A, 2)**2
    stepsize = 1.0/L #The init step influences a lot the convergence plot, try with 320 instead of L as example
    logs = [x_i.copy()]
    #definitio m parameter
    m=m_choice
    # definition of set S, R, phi
    #phi = []
    H_zero = np.eye(n)
    for k in range(1, max_iter):
        #step 0: compute approximation to the inverse of the Hessian matrix

        if (k==1):
            H_inverse= np.eye(n)
        else:
            H_inverse= np.eye(n) - ((s @ s.T)/ (y.T @ s) ) * H_inverse @ (np.eye(n) - ((y @ s.T) / (y.T @ s)) + (s @ y.T) / (y.T @ s))
        direction = -H_inverse @ grad(x_i)

        #step 2: compute the step size
        stepsize = line_search_Wolfe(g, grad, x_i, direction)
        x_old= x_i.copy()
        x_i = x_i + stepsize * direction
        s = x_i - x_old
        y = grad(x_i) - grad(x_old)
        
        #H_zero = (H_zero- (H_zero @ np.outer(s, s) @ H_zero) / (s.T @ H_zero @ s)+ np.outer(y, y) / (y.T @ s))
        logs.append(x_i.copy())
        if StopCriterionFunction(x_i, x_old, tol):
            break
    return x_i, logs



# Initialization
max_iter = 4_000
tolerance = 1e-12

nb_features = A_train.shape[1]
nb_samples = A_train.shape[0]
lambdaMax = np.max(np.abs(A_train.T @ b_train)) / nb_samples

lam1 = 0.1
lam2 = 0.01
CURRENT_MODE = LEASTSQUARES # Choose between LASSO, ELASTICNET, RIDGE


if CURRENT_MODE == RIDGE:
    # LASSO  ||Ax - b||² + λ||x||₁
    objectiveFunction = lambda x: 0.5 * np.linalg.norm(A_train@x-b_train, 2)**2 + lam1 * np.linalg.norm(x, 1)
    #gradient of the smooth part
    gradient = lambda x: A_train.T @ (A_train @ x - b_train)
elif CURRENT_MODE == RIDGE:
    # RIDGE  ||Ax - b||² + λ||x||₂²
    objectiveFunction = lambda x: 0.5 * np.linalg.norm(A_train@x-b_train, 2)**2 + 0.5*lam2 * np.linalg.norm(x, 2)**2
elif CURRENT_MODE == ELASTICNET:
    # ELASTICNET ||Ax - b||² + λ₁||x||₁ + λ₂||x||₂²
    objectiveFunction = lambda x: 0.5 * np.linalg.norm(A_train@x-b_train, 2)**2 + lam1 * np.linalg.norm(x, 1) + 0.5*lam2 * np.linalg.norm(x, 2)**2
    #gradient of the smooth part

elif CURRENT_MODE == LEASTSQUARES:
    # LEASTSQUARES ||Ax - b||²
    objectiveFunction = lambda x: 0.5 * np.linalg.norm(A_train@x-b_train, 2)**2


def stopCriterion(x_i, x_old, tol):
    return np.abs(objectiveFunction(x_i) - objectiveFunction(x_old)) < tol
# Exécution ISTA



x_hat_ista, logs_ista = ISTA(A_train, b_train,stopCriterion,CURRENT_MODE ,lam1,lam2, max_iter,tolerance, adaptive_step=False)
print("ISTA converged in", len(logs_ista), "iterations")

x_hat_fista, logs_fista = fista(A_train, b_train,stopCriterion,CURRENT_MODE ,lam1,lam2, max_iter,tolerance)
print("FISTA converged in", len(logs_fista), "iterations")

x_hat_sgd, logs_sgd = subgradient_descent(A_train, b_train,stopCriterion,CURRENT_MODE ,lam1,lam2, max_iter,tolerance)
print("Subgradient Descent converged in", len(logs_sgd), "iterations")



if CURRENT_MODE== LEASTSQUARES or CURRENT_MODE == RIDGE:

    x_hat_gd, logs_gd = gradient_descent(A_train, b_train,stopCriterion,CURRENT_MODE ,lam2, max_iter,tolerance)
    print("GD converged in", len(logs_gd), "iterations")
    mse_Per_iter_gd = [objectiveFunction(x) for x in logs_gd]

    x_hat_bgfs, logs_bgfs = BGFS(A_train, b_train,stopCriterion,CURRENT_MODE ,lam1,lam2, max_iter,tolerance, m_choice=10)
    print("BGFS converged in", len(logs_bgfs), "iterations")
    mse_Per_iter_bgfs = [objectiveFunction(x) for x in logs_bgfs]   

    x_hat_lbgfs, logs_lbgfs = LBGFS(A_train, b_train,stopCriterion,CURRENT_MODE ,lam1,lam2, max_iter,tolerance, m_choice=10)
    print("LBGFS converged in", len(logs_lbgfs), "iterations")
    mse_Per_iter_lbgfs = [objectiveFunction(x) for x in logs_lbgfs]  


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


