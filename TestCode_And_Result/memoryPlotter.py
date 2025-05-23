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
import tracemalloc



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
    k = 0
    for k in range(max_iter):
        x_old = x_i.copy()
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
        elif(MODE==ELASTICNET):
            # ElasticNet
            sizeStep = backtracking_line_search(x_i,grad(x_i),g,sizeStep)
            x_i = soft_thresholding(x_i -  sizeStep  * (grad(x_i) + lam2 * x_i), lam1 *  sizeStep )/ (1+lam2*sizeStep)
        else:
            x_i = x_i - sizeStep * grad(x_i)
        #stop criterion 
        if StopCriterionFunction(x_i, x_old, tol):
            break
        
    return x_i,k









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
    k = 0
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

        #stop criterion 
        if StopCriterionFunction(x_i, x_old, tol):
            break
    return x_i,k
    
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
    
    #eta0 = 0.001
    # calcul Lipschitz
    L = np.linalg.norm(A, 2)**2
    stepsize = 1.0 / L

    k= 0
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
        
        if StopCriterionFunction(x_i, x_old, tol):
            break

    return x_i,k

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
    k= 0
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
        
        if StopCriterionFunction(x_i, x_old, tol):
            break

    return x_i,k
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
    #definitio m parameter
    m=5
    # definition of set S, R, phi
    S = []
    Y = []
    
    T = [0] * m
    phi = [0] * m
    # start for loop
    k=0
    r = np.zeros(n)
    #B_zeroArray = np.ones(n)
    for k in range(1, max_iter):
        
        #step 0: compute approximation to the inverse of the Hessian matrix
        if (k==1):
            B_zeroArray = np.ones(n)
            #B_zero= np.eye(n)

        else:
            B_zeroArray = ( S[0].T @ Y[0]) / (Y[0].T @ Y[0]) * np.ones(n)
            #B_zero=( S[0].T @ Y[0]) / (Y[0].T @ Y[0]) * np.eye(n)
            #B_zero=( S[0].T @ Y[0]) / (Y[0].T @ Y[0])*sparse.eye(n, format='csr')

        #step 1: compute the gradient
        q=grad(x_i)

        for i in range(len(S)):
            #phi.append( 1.0 / (Y[i].T @ S[i]))
            phi[i] = 1.0 / (Y[i].T @ S[i])
            if(1.0 / (Y[i].T @ S[i]) < 0):
                raise ValueError("phi must be positive")
            #T.append(phi[i] * S[i].T @ q)
            T[i] = phi[i] * S[i].T @ q
            q=q-(T[i]*Y[i])

        for i in range(len(r)):
            r[i] = B_zeroArray[i] * q[i]

        #r=B_zero @ q
        for i in range(len(S)-1, -1, -1):
            beta = phi[i] * Y[i].T @ r
            r = r + (S[i] * (T[i] - beta))
        direction = -r
        #step 2: compute the step size
        #tk satisfies the Wolfe conditions
        #stepsize= backtrackingWolfe(x_i,grad(x_i),grad(x_i + direction*stepsize),g,stepsize)
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
        if StopCriterionFunction(x_i, x_old, tol):
            break

    return x_i,k

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
    #definitio m parameter
    m=m_choice
    # definition of set S, R, phi
    #phi = []
    H_zero = np.eye(n)
    k=0
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
        if StopCriterionFunction(x_i, x_old, tol):
            break
    return x_i,k



# Initialization
x0 = np.zeros(A_train.shape[1])
max_iter = 4_000
tolerance = 1e-6

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





memoryUsage_FISTA=[]
memoryUsage_ISTA=[]
memoryUsage_SGD=[]
memoryUsage_GD=[]
memoryUsage_LBGFS=[]
memoryUsage_BGFS=[]

iterations_FISTA=[]
iterations_ISTA=[]
iterations_SGD=[]
iterations_GD=[]
iterations_LBGFS=[]
iterations_BGFS=[]






max_iter = 500
tolerance = 1e-6


feature_ranges = range(50, 1051, 50)  # de 5 à 1000, pas de 50

for n_features in feature_ranges:
    A_train, b_train, b_mean, b_std = load_and_preprocess_data(csv_path=f"synthetic_datasets/synthetic_data_{n_features}_features.csv",targetColumn = 'target',train_size=train_size, random_state=random_state)
    #FISTA
    tracemalloc.start()
    x_hat_fista, nb_iteration_fista = fista(A_train, b_train,stopCriterion,CURRENT_MODE ,lam1,lam2, max_iter,tolerance)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    memoryUsage_FISTA.append(peak / 1024**2)
    iterations_FISTA.append(nb_iteration_fista)
    #ISTA
    tracemalloc.start()
    x_hat_ista, nb_iteration_ista = ISTA(A_train, b_train,stopCriterion,CURRENT_MODE ,lam1,lam2, max_iter,tolerance)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    memoryUsage_ISTA.append(peak / 1024**2)
    iterations_ISTA.append(nb_iteration_ista)
    
    #Subgradient Descent
    tracemalloc.start()
    x_hat_sgd, nb_iteration_sgd = subgradient_descent(A_train, b_train,stopCriterion,CURRENT_MODE ,lam1,lam2, max_iter,tolerance)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    memoryUsage_SGD.append(peak / 1024**2)
    iterations_SGD.append(nb_iteration_sgd)
    
    #Gradient Descent
    tracemalloc.start()
    x_hat_gd, nb_iteration_gd = gradient_descent(A_train, b_train,stopCriterion,CURRENT_MODE ,lam2, max_iter,tolerance)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    memoryUsage_GD.append(peak / 1024**2)
    iterations_GD.append(nb_iteration_gd)


    #LBGFS
    tracemalloc.start()
    x_hat_lbgfs, nb_iteration_lbgfs = LBGFS(A_train, b_train,stopCriterion,CURRENT_MODE ,lam1,lam2, max_iter,tolerance, m_choice=5)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    memoryUsage_LBGFS.append(peak / 1024**2)
    iterations_LBGFS.append(nb_iteration_lbgfs)
    #BGFS
    tracemalloc.start()
    x_hat_bgfs, nb_iteration_bgfs = BGFS(A_train, b_train,stopCriterion,CURRENT_MODE ,lam1,lam2, max_iter,tolerance, m_choice=10)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    memoryUsage_BGFS.append(peak / 1024**2)
    iterations_BGFS.append(nb_iteration_bgfs)
    print("finished",n_features)



print("Memory usage FISTA:", memoryUsage_FISTA)
print("Memory usage ISTA:", memoryUsage_ISTA)
print("Memory usage Subgradient Descent:", memoryUsage_SGD)
print("Memory usage Gradient Descent:", memoryUsage_GD)
print("Memory usage LBGFS:", memoryUsage_LBGFS)
print("Memory usage BGFS:", memoryUsage_BGFS)


# Plotting memory usage
plt.figure(figsize=(10, 6))
plt.plot(feature_ranges, memoryUsage_FISTA, label='FISTA', marker='o')
plt.plot(feature_ranges, memoryUsage_ISTA, label='ISTA', marker='o')
plt.plot(feature_ranges, memoryUsage_SGD, label='Subgradient Descent', marker='o')
plt.plot(feature_ranges, memoryUsage_GD, label='Gradient Descent', marker='o')
plt.plot(feature_ranges, memoryUsage_LBGFS, label='LBGFS', marker='o')
plt.plot(feature_ranges, memoryUsage_BGFS, label='BGFS', marker='o')
plt.title('Memory Usage vs Number of Features')
plt.xlabel('Number of Features')
plt.ylabel('Memory Usage (MiB)')
plt.legend()
plt.grid()
plt.savefig("memory_usage.png")
plt.show()
print("Memory usage plot saved as 'memory_usage.png'")

#make same plot but with log scale
plt.figure(figsize=(10, 6))
plt.plot(feature_ranges, memoryUsage_FISTA, label='FISTA', marker='o')
plt.plot(feature_ranges, memoryUsage_FISTA, label='FISTA', marker='o')
plt.plot(feature_ranges, memoryUsage_ISTA, label='ISTA', marker='o')
plt.plot(feature_ranges, memoryUsage_SGD, label='Subgradient Descent', marker='o')
plt.plot(feature_ranges, memoryUsage_GD, label='Gradient Descent', marker='o')
plt.plot(feature_ranges, memoryUsage_LBGFS, label='LBGFS', marker='o')
plt.plot(feature_ranges, memoryUsage_BGFS, label='BGFS', marker='o')
plt.title('Memory Usage vs Number of Features (Log Scale)')
plt.xlabel('Number of Features')
plt.ylabel('Memory Usage (MiB)')
plt.yscale('log', base=2)
plt.legend()
plt.grid()
plt.savefig("memory_usage_log.png")
plt.show()
print("Memory usage log plot saved as 'memory_usage_log.png'")

plt.figure(figsize=(10, 6))
plt.plot(feature_ranges, memoryUsage_FISTA, label='FISTA', marker='o')
plt.plot(feature_ranges, memoryUsage_ISTA, label='ISTA', marker='o')
plt.plot(feature_ranges, memoryUsage_SGD, label='Subgradient Descent', marker='o')
plt.plot(feature_ranges, memoryUsage_GD, label='Gradient Descent', marker='o')
plt.plot(feature_ranges, memoryUsage_LBGFS, label='LBGFS', marker='o')
#plt.plot(feature_ranges, memoryUsage_BGFS, label='BGFS', marker='o')
plt.title('Memory Usage vs Number of Features')
plt.xlabel('Number of Features')
plt.ylabel('Memory Usage (MiB)')
plt.legend()
plt.grid()
plt.savefig("memory_usage_whithoutBFGS.png")
plt.show()
print("Memory usage plot saved as 'memory_usage.png'")




print("length of iterations FISTA", len(iterations_FISTA))
print("length of iterations ISTA", len(iterations_ISTA))
print("length of iterations Subgradient Descent", len(iterations_SGD))
print("length of iterations Gradient Descent", len(iterations_GD))
print("length of iterations LBGFS", len(iterations_LBGFS))
print("length of iterations BGFS", len(iterations_BGFS))




plt.figure(figsize=(10, 6))
plt.plot(feature_ranges, iterations_FISTA, label='FISTA', marker='o')
plt.plot(feature_ranges, iterations_ISTA, label='ISTA', marker='o')
plt.plot(feature_ranges, iterations_SGD, label='Subgradient Descent', marker='o')
plt.plot(feature_ranges, iterations_GD, label='Gradient Descent', marker='o')
plt.plot(feature_ranges, iterations_LBGFS, label='LBGFS', marker='o')
plt.plot(feature_ranges, iterations_BGFS, label='BGFS', marker='o')
plt.title('Iterations vs Number of Features')
plt.xlabel('Number of Features')
plt.ylabel('Iterations')
plt.legend()
plt.grid()
plt.savefig("iterations.png")
plt.show()