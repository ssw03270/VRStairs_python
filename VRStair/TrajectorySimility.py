import math
import matplotlib.pyplot as plt
import numpy as np

h = 0.125
d = 0.3

def sigmoid(x):
    x = x/d * 10
    k = np.exp(x)
    return k/(1.0 + k)

def H_VE(H_real,i,transAmount):
    return H_real + (i-1) * h + 2*h *sigmoid(transAmount-i*d)

def dtw(s,t):
    n,m = len(s),len(t)
    dtw_matrix = np.zeros((n+1,m+1))
    for i in range(n+1):
        for j in range(m+1):
            dtw_matrix[i,j] = np.inf
    dtw_matrix[0,0] = 0

    for i in range(1,n+1):
        for j in range(1,m+1):
            cost = abs(s[i-1]-t[j-1])
            last_min = np.min([dtw_matrix[i-1,j],dtw_matrix[i,j-1],dtw_matrix[i-1,j-1]])
            dtw_matrix[i,j] = cost + last_min
    return dtw_matrix[n,m]

def test():
    x = []
    y = []
    allStep = 10

    for k in range(0,allStep,2):
        for j in np.arange((k-1) * d, k*d + d,0.01):
            x.append(j)
            y.append(H_VE(0,k,j))

    plt.plot(x,y)
    x1 = []
    y1 = []
    for k in range(1,allStep,2):
        for j in np.arange((k-1) * d, k*d + d,0.01):
            x1.append(j)
            y1.append(H_VE(0,k,j))

    plt.plot(x1, y1)

    a = np.array([1, 2, 3, 3, 7])
    b = np.array([1, 2, 2, 2, 2, 2, 2, 4])

    print(dtw(y, y1))

    plt.show()



