import numpy as np

n = 100
d = 1
h = 1e-8
k = 2

np.random.seed(42)
X = np.random.randn(n, d)
W = np.random.randn(d, k)
B = np.random.randn(n, k)
Y = np.random.rand(n, k)

def softmax(P):
    Qsm = np.empty_like(P)
    for i in range(P.shape[0]):
        for j in range(P.shape[1]):
            Qsm[i, j] = np.exp(P[i, j]) / np.sum(np.exp(P[i:]))
    return Qsm

def Loss(X, W, B, Y):
    P = X @ W + B
    Qsm = softmax(P)
    L = 0
    for i in range(n):
        for j in range(k):
            L += np.sum(Y[i, j]*np.log(Qsm[i, j]))
    return L

def delLX(L, X):
    delOp = np.empty_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            X_h = X.copy()
            X_h[i, j] += h
            L_h = Loss(X_h, W, B, Y)
            delOp[i, j] = (L_h - L) / h
    print(f"The derivative 'del L/del X' matrix from finite differences is:\n{delOp}")

def delLW(L, W):
    delOp = np.empty_like(W)
    for i in range(W.shape[0]):
        for j in range(W.shape[1]):
            W_h = W.copy()
            W_h[i, j] += h
            L_h = Loss(X, W_h, B, Y)
            delOp[i, j] = (L_h - L) / h
    print(f"The derivative 'del L/del W' matrix from finite differences is:\n{delOp}")

def delLB(L, B):
    delOp = np.empty_like(B)
    for i in range(B.shape[0]):
        for j in range(B.shape[1]):
            B_h = B.copy()
            B_h[i, j] += h
            L_h = Loss(X, W, B_h, Y)
            delOp[i, j] = (L_h - L) / h
    print(f"The derivative 'del L/del B' matrix from finite differences is:\n{delOp}")

def printDelLX():
    delLX_f = -(softmax(X @ W + B) - Y) @ W.T
    print(f"The derivative 'del L/del X' matrix from the formula is:\n{delLX_f}")

def printDelLW():
    delLW_f = -100*X.T @ (softmax(X @ W + B) - Y)
    print(f"The derivative 'del L/del W' matrix from the formula is:\n{delLW_f}")

def printDelLB():
    delLB_f = -10*(softmax(X @ W + B) - Y)
    print(f"The derivative 'del L/del B' vector from the formula is:\n{delLB_f}")

L = Loss(X, W, B, Y)
delLX(L, X)
print("\n")
printDelLX()
print("\n\n\n")
delLW(L, W)
print("\n")
printDelLW()
print("\n\n\n")
delLB(L, B)
print("\n")
printDelLB()