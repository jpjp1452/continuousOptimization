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

now = time.perf_counter


train_size = 0.9
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

# Initialization
max_iter = 5000
tolerance = 1e-9


nb_features = A_train.shape[1]
nb_samples = A_train.shape[0]
lambdaMax = np.max(np.abs(A_train.T @ b_train)) / nb_samples

lam1 = 0.2
lam2 = 0.03

m = 10 # Number of past iterations to consider in L-BFGS

CURRENT_MODE = RIDGE # Choose between LASSO, ELASTICNET, RIDGE



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

def line_search_Wolfe(f, grad, x, d, alpha0=1.0, c1=1e-4, c2=0.9):
        alpha = alpha0
        fx = f(x); gdx = grad(x).T @ d
        for i in range(20):
            # if Armijo condition or curvature condition is not satisfied
            if f(x + alpha*d) > fx + c1*alpha*gdx or grad(x + alpha*d).T @ d < c2*gdx:
                alpha *= 0.5
            else:
                return alpha
        return alpha








def ISTA(functionInfo):
    start = now()
    timeSpentBacktracking = 0
    timeSpentProx = 0
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
        currentTime = now()
        sizeStep = backtracking_line_search(x_i,grad(x_i),g,sizeStep)
        timeSpentBacktracking += now() - currentTime

        currentTime = now()
        x_i = prox(x_i - grad(x_i) *sizeStep,sizeStep)
        timeSpentProx += now() - currentTime
        logs.append(x_i)
        #stop criterion 
        if StopCriterionFunction(x_i, x_old, tol):
            break
    end = now()
    timeSpent = end - start
    return x_i,logs, timeSpent, timeSpentBacktracking, timeSpentProx

def FISTA(functionInfo):
    start = now()
    timeSpentBacktracking = 0
    timeSpentProx = 0



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
    logs =[x_i]
    t=1
    for k in range(max_iter):
        x_old = x_i.copy()
        currentTime = now()
        sizeStep = backtracking_line_search(y,grad(y),g,sizeStep)
        timeSpentBacktracking += now() - currentTime
        currentTime = now()
        x_i = prox(y - ( grad(y)) *sizeStep,sizeStep)
        timeSpentProx += now() - currentTime
        t_new = (1 + np.sqrt(1 + 4 * t**2)) / 2
        y = x_i + ((t - 1) / t_new) * (x_i - x_old)
        t = t_new
        logs.append(x_i)
        #stop criterion 
        if StopCriterionFunction(x_i, x_old, tol):
            break
    end = now()
    timeSpent = end - start
    return x_i,logs, timeSpent, timeSpentBacktracking, timeSpentProx



def subgradient_descent(INFOFunction):
    start = now()
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
    end = now()
    timeSpent = end - start
    return x_i,logs, timeSpent




def gradient_descent(INFOFunction):
    start = now()
    timeSpentBacktracking = 0
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
        currentTime = now()
        stepsize = backtracking_line_search(x_i,grad(x_i),g,stepsize)
        timeSpentBacktracking += now() - currentTime
        x_i = x_i - stepsize * grad(x_i)
        logs.append(x_i)
        #stop criterion 
        if StopCriterionFunction(x_i, x_old, tol):
            break
    end = now()
    timeSpent = end - start
    return x_i,logs, timeSpent,timeSpentBacktracking

def LBGFS(INFOFunction):
    start = now()
    timeSpentBacktracking = 0
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
        current_time = now()
        stepsize = line_search_Wolfe(g, grad, x_i, direction)
        timeSpentBacktracking += now() - current_time
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
    end = now()
    timeSpent = end - start
    return x_i, logs, timeSpent, timeSpentBacktracking


def BGFS(INFOFunction):
    start = now()
    timeSpentBacktracking = 0
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
        current_time = now()
        stepsize = line_search_Wolfe(g, grad, x_i, direction)
        timeSpentBacktracking += now() - current_time
        x_old= x_i.copy()
        x_i = x_i + stepsize * direction
        s = x_i - x_old
        y = grad(x_i) - grad(x_old)
        
        logs.append(x_i.copy())
        if StopCriterionFunction(x_i, x_old, tol):
            break
    end = now()
    timeSpent = end - start
    return x_i, logs, timeSpent, timeSpentBacktracking




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


def stopCriterion1(x_i, x_old, tol):
    #print(np.abs(objectiveFunction(x_i) - objectiveFunction(x_old))/max(1, np.abs(objectiveFunction(x_i))))
    return np.abs(objectiveFunction(x_i) - objectiveFunction(x_old))/max(1, np.abs(objectiveFunction(x_i))) < tol

def stopCriterion2(x_i, x_old, tol):
    #print(np.linalg.norm(INFO["gradient"](x_i), 2))
    return  np.linalg.norm(INFO["gradient"](x_i), 2) < tol


#INFO["x_0"] = np.random.randn(nb_features)
INFO["x_0"] = np.zeros(nb_features)
INFO["stop criterion"] = stopCriterion1
INFO["tolerance"] = tolerance



CURRENT_MODE = LASSO
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



NB_TRIES = 5

"""
nb_iterations_ISTA = []
mse_ISTA = []
time_ISTA = []
timeSpentBacktracking_ISTA = []
timeSpentProx_ISTA = []


nb_iterations_FISTA = []
mse_FISTA = []
time_FISTA = []
timeSpentBacktracking_FISTA = []
timeSpentProx_FISTA = []

nb_iterations_SGD = []
mse_SGD = []
time_SGD = []

for i in range(NB_TRIES):
    print("x_0", INFO["x_0"])
    x_ISTA, logs_ISTA, timeSpent, timeSpentBacktracking, timeSpentProx = ISTA(INFO)
    nb_iterations_ISTA.append(len(logs_ISTA))
    mse_ISTA.append(mean_squared_error(b_train, A_train @ x_ISTA))
    time_ISTA.append(timeSpent)
    timeSpentBacktracking_ISTA.append(timeSpentBacktracking)
    timeSpentProx_ISTA.append(timeSpentProx)
    print("ISTA converged in", len(logs_ISTA), "iterations")
    #######################################################
    x_FISTA, logs_FISTA, timeSpent, timeSpentBacktracking, timeSpentProx = FISTA(INFO)
    nb_iterations_FISTA.append(len(logs_FISTA))
    mse_FISTA.append(mean_squared_error(b_train, A_train @ x_FISTA))
    time_FISTA.append(timeSpent)
    timeSpentBacktracking_FISTA.append(timeSpentBacktracking)
    timeSpentProx_FISTA.append(timeSpentProx)
    print("FISTA converged in", len(logs_FISTA), "iterations")
    #######################################################
    x_SGD, logs_SGD, timeSpent = subgradient_descent(INFO)
    nb_iterations_SGD.append(len(logs_SGD))
    mse_SGD.append(mean_squared_error(b_train, A_train @ x_SGD))
    time_SGD.append(timeSpent)
    #######################################################
    print("iteration", i)
    print("")



#open csv file to store min max average time store in A1 ISTA, A2 FISTA, A3 SGD   in B put min time , in C put max time, in D put average time
log_base = 2
df = pd.DataFrame({
    'Method': ['ISTA', 'FISTA', 'SGD'],
    'Min Time (s)': [np.min(time_ISTA), np.min(time_FISTA), np.min(time_SGD)],
    'Max Time (s)': [np.max(time_ISTA), np.max(time_FISTA), np.max(time_SGD)],
    'Average Time (s)': [np.mean(time_ISTA), np.mean(time_FISTA), np.mean(time_SGD)],
    'standard deviation Time (s)': [np.std(time_ISTA), np.std(time_FISTA), np.std(time_SGD)],
    'Min Iterations': [np.min(nb_iterations_ISTA), np.min(nb_iterations_FISTA), np.min(nb_iterations_SGD)],
    'Max Iterations': [np.max(nb_iterations_ISTA), np.max(nb_iterations_FISTA), np.max(nb_iterations_SGD)],
    'Average Iterations': [np.mean(nb_iterations_ISTA), np.mean(nb_iterations_FISTA), np.mean(nb_iterations_SGD)],
    'Standard Deviation Iterations': [np.std(nb_iterations_ISTA), np.std(nb_iterations_FISTA), np.std(nb_iterations_SGD)],
})

# Save the DataFrame to a CSV file
# this creates an .xlsx that will open cleanly in Excel
df.to_excel('performance_resultsLASSO.xlsx', index=False)

exit(0)
#ratio subroutines ista fista


df= pd.DataFrame({
    'Method': ['ISTA', 'FISTA'],
    'Time Ratio linesearch': [np.mean(timeSpentBacktracking_ISTA)/np.mean(time_ISTA), np.mean(timeSpentBacktracking_FISTA)/np.mean(time_FISTA)],
    'Time Ratio Prox': [np.mean(timeSpentProx_ISTA)/np.mean(time_ISTA), np.mean(timeSpentProx_FISTA)/np.mean(time_FISTA)],
})
# Save the DataFrame to a CSV file
# this creates an .xlsx that will open cleanly in Excel
df.to_excel('performance_results_ratio_subroutine.xlsx', index=False)
"""
CURRENT_MODE = RIDGE
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


nb_iterations_ISTA = []
mse_ISTA = []
time_ISTA = []
timeSpentBacktracking_ISTA = []
timeSpentProx_ISTA = []


nb_iterations_FISTA = []
mse_FISTA = []
time_FISTA = []
timeSpentBacktracking_FISTA = []
timeSpentProx_FISTA = []

nb_iterations_SGD = []
mse_SGD = []
time_SGD = []

nb_iterations_GD = []
mse_GD = []
time_GD = []
timeSpentBacktracking_GD = []

nb_iterations_LBGFS = []
mse_LBGFS = []
time_LBGFS = []
timeSpentBacktracking_LBGFS = []

nb_iterations_BGFS = []
mse_BGFS = []
time_BGFS = []
timeSpentBacktracking_BGFS = []

for i in range(NB_TRIES):

    x_ISTA, logs_ISTA, timeSpent, timeSpentBacktracking, timeSpentProx = ISTA(INFO)
    nb_iterations_ISTA.append(len(logs_ISTA))
    mse_ISTA.append(mean_squared_error(b_train, A_train @ x_ISTA))
    time_ISTA.append(timeSpent)
    timeSpentBacktracking_ISTA.append(timeSpentBacktracking)
    timeSpentProx_ISTA.append(timeSpentProx)
    #######################################################
    x_FISTA, logs_FISTA, timeSpent, timeSpentBacktracking, timeSpentProx = FISTA(INFO)
    nb_iterations_FISTA.append(len(logs_FISTA))
    mse_FISTA.append(mean_squared_error(b_train, A_train @ x_FISTA))
    time_FISTA.append(timeSpent)
    timeSpentBacktracking_FISTA.append(timeSpentBacktracking)
    timeSpentProx_FISTA.append(timeSpentProx)
    #######################################################
    x_SGD, logs_SGD, timeSpent = subgradient_descent(INFO)
    nb_iterations_SGD.append(len(logs_SGD))
    mse_SGD.append(mean_squared_error(b_train, A_train @ x_SGD))
    time_SGD.append(timeSpent)
    #######################################################

    x_GD, logs_GD, timeSpent, timeSpentBacktracking = gradient_descent(INFO)
    nb_iterations_GD.append(len(logs_GD))
    mse_GD.append(mean_squared_error(b_train, A_train @ x_GD))
    time_GD.append(timeSpent)
    timeSpentBacktracking_GD.append(timeSpentBacktracking)
    #######################################################

    x_LBGFS, logs_LBGFS, timeSpent, timeSpentBacktracking = LBGFS(INFO)
    nb_iterations_LBGFS.append(len(logs_LBGFS))
    mse_LBGFS.append(mean_squared_error(b_train, A_train @ x_LBGFS))
    time_LBGFS.append(timeSpent)
    timeSpentBacktracking_LBGFS.append(timeSpentBacktracking)

    
    #######################################################
    x_BGFS, logs_BGFS, timeSpent, timeSpentBacktracking = BGFS(INFO)
    nb_iterations_BGFS.append(len(logs_BGFS))
    mse_BGFS.append(mean_squared_error(b_train, A_train @ x_BGFS))
    time_BGFS.append(timeSpent)
    timeSpentBacktracking_BGFS.append(timeSpentBacktracking)
    print("iteration", i)
    print("")

#open csv file to store min max average time store in A1 ISTA, A2 FISTA, A3 SGD   in B put min time , in C put max time, in D put average time
log_base = 2
df = pd.DataFrame({
    'Method': ['ISTA', 'FISTA', 'SGD', 'GD', 'LBGFS', 'BGFS'],
    'Min Time (s)': [np.min(time_ISTA), np.min(time_FISTA), np.min(time_SGD), np.min(time_GD), np.min(time_LBGFS), np.min(time_BGFS)],
    'Max Time (s)': [np.max(time_ISTA), np.max(time_FISTA), np.max(time_SGD), np.max(time_GD), np.max(time_LBGFS), np.max(time_BGFS)],
    'Average Time (s)': [np.mean(time_ISTA), np.mean(time_FISTA), np.mean(time_SGD), np.mean(time_GD), np.mean(time_LBGFS), np.mean(time_BGFS)],
    'Standard Deviation Time (s)': [np.std(time_ISTA), np.std(time_FISTA), np.std(time_SGD), np.std(time_GD), np.std(time_LBGFS), np.std(time_BGFS)],
    'Min Iterations': [np.min(nb_iterations_ISTA), np.min(nb_iterations_FISTA), np.min(nb_iterations_SGD), np.min(nb_iterations_GD), np.min(nb_iterations_LBGFS), np.min(nb_iterations_BGFS)],
    'Max Iterations': [np.max(nb_iterations_ISTA), np.max(nb_iterations_FISTA), np.max(nb_iterations_SGD), np.max(nb_iterations_GD), np.max(nb_iterations_LBGFS), np.max(nb_iterations_BGFS)],
    'Average Iterations': [np.mean(nb_iterations_ISTA), np.mean(nb_iterations_FISTA), np.mean(nb_iterations_SGD), np.mean(nb_iterations_GD), np.mean(nb_iterations_LBGFS), np.mean(nb_iterations_BGFS)],
    'Standard Deviation Iterations': [np.std(nb_iterations_ISTA), np.std(nb_iterations_FISTA), np.std(nb_iterations_SGD), np.std(nb_iterations_GD), np.std(nb_iterations_LBGFS), np.std(nb_iterations_BGFS)],
})

# Save the DataFrame to a CSV file
# this creates an .xlsx that will open cleanly in Excel
df.to_excel('performance_results_RIDGE.xlsx', index=False)
#ratio subroutines ista fista

df= pd.DataFrame({
    'Method': ['ISTA', 'FISTA', 'SGD', 'GD', 'LBGFS', 'BGFS'],
    'Time Ratio linesearch': [np.mean(timeSpentBacktracking_ISTA)/np.mean(time_ISTA), np.mean(timeSpentBacktracking_FISTA)/np.mean(time_FISTA), "/", np.mean(timeSpentBacktracking_GD)/np.mean(time_GD), np.mean(timeSpentBacktracking_LBGFS)/np.mean(time_LBGFS), np.mean(timeSpentBacktracking_BGFS)/np.mean(time_BGFS)],
    'Time Ratio Prox': [np.mean(timeSpentProx_ISTA)/np.mean(time_ISTA), np.mean(timeSpentProx_FISTA)/np.mean(time_FISTA), "/","/", "/", "/"],
})

# Save the DataFrame to a CSV file
# this creates an .xlsx that will open cleanly in Excel
df.to_excel('performance_results_ratio_subroutine_LEASTSQUARE.xlsx', index=False)


   




CURRENT_MODE = RIDGE
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

exit(0)
A_train, b_train,b_mean, b_std = load_and_preprocess_data(csv_path="synthetic_datasets/synthetic_data_1000_features.csv",targetColumn = 'target',train_size=train_size, random_state=random_state)
nb_features = A_train.shape[1]
x_0 = np.zeros(nb_features)
INFO["x_0"] = x_0
nb_iterations_LBGFS = []
time_LBGFS = []

nb_iterations_BGFS = []
time_BGFS = []
for i in range(NB_TRIES):

    x_LBGFS, logs_LBGFS, timeSpent, timeSpentBacktracking = LBGFS(INFO)
    nb_iterations_LBGFS.append(len(logs_LBGFS))
    time_LBGFS.append(timeSpent)

    
    #######################################################
    x_BGFS, logs_BGFS, timeSpent, timeSpentBacktracking = BGFS(INFO)
    nb_iterations_BGFS.append(len(logs_BGFS))
    time_BGFS.append(timeSpent)
    print("iteration", i)
    print("")

#open csv file to store min max average time store in A1 ISTA, A2 FISTA, A3 SGD   in B put min time , in C put max time, in D put average time
log_base = 2
df = pd.DataFrame({
    'Method': ['LBGFS', 'BGFS'],
    'Min Time (s)': [np.min(time_LBGFS), np.min(time_BGFS)],
    'Max Time (s)': [np.max(time_LBGFS), np.max(time_BGFS)],
    'Average Time (s)': [np.mean(time_LBGFS), np.mean(time_BGFS)],
    'Standard Deviation Time (s)': [np.std(time_LBGFS), np.std(time_BGFS)],
    'Min Iterations': [np.min(nb_iterations_LBGFS), np.min(nb_iterations_BGFS)],
    'Max Iterations': [np.max(nb_iterations_LBGFS), np.max(nb_iterations_BGFS)],
    'Average Iterations': [np.mean(nb_iterations_LBGFS), np.mean(nb_iterations_BGFS)],
    'Standard Deviation Iterations': [np.std(nb_iterations_LBGFS), np.std(nb_iterations_BGFS)],
})
# Save the DataFrame to a CSV file
# this creates an .xlsx that will open cleanly in Excel
df.to_excel('performance_results_synthetic_data_1000_features.xlsx', index=False)