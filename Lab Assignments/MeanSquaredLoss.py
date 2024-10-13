import numpy as np
n=100
d=3
h = 1e-8
np.random.seed(42)
X = np.random.randn(n,d)
W = np.random.randn(d,1)
B = np.random.randn(n,1)
Y = np.random.randn(n,1)

def Loss (X,W,B,Y):
  P = X@W + B
  Q = P-Y
  L = (Q.T@Q)/2
  return L

L = Loss(X,W,B,Y)

# def bayesian_loss(X, W, B, Y, alpha):
#     P = X @ W + B
#     Q = P - Y
#     L = (Q.T @ Q) / 2 + (alpha / 2) * (W.T @ W)  # Add regularization term
#     return L

# # Set a regularization parameter (alpha) for Bayesian regression
# alpha = 1.0
# L = bayesian_loss(X, W, B, Y, alpha)

def delLX (L,X,n,d):
  delOp = np.empty((n,d))
  for i in range(n):
    for j in range(d):
      X_h = np.copy(X)
      X_h[i,j]  = X_h[i,j] + h
      L_h = Loss(X_h,W,B,Y)
      delOp[i,j] = (L_h - L)/h
  print(f"The derivative 'del L/del X' matrix from first principal is: {delOp}")

def delLW (L,W,d):
  delOp = np.empty(d)
  for i in range(d):
    W_h = np.copy(W)
    W_h[i]  += h
    L_h = Loss(X,W_h,B,Y)
    delOp[i] = (L_h - L)/h
  print(f"The derivative 'del L/del W' vector from first principal is: {delOp}")

def delLB (L,B,n):
  delOp = np.empty(n)
  for i in range(n):
    B_h = np.copy(B)
    B_h[i]  += h
    L_h = Loss(X,W,B_h,Y)
    delOp[i] = (L_h - L)/h
  print(f"The derivative 'del L/del B' vector from first principal is: {delOp}")

def printDelLX ():
  delLX_f =((X@W) +B - Y)@W.T
  print(f"The derivative 'del L/del W' vector from formula is: {delLX_f}")

def printDelLW ():
  delLW_f = X.T@((X@W) + B - Y)
  print(f"The derivative 'del L/del W' vector from formula is: {delLW_f}")

def printDelLB ():
  delLB_f = ((X@W) +B  - Y)
  print(f"The derivative 'del L/del B' vector from formula is: {delLB_f}")

delLX(L,X,n,d)
print("\n")
printDelLX()
print("\n\n\n")
delLW(L,W,d)
print("\n")
printDelLW()
print("\n\n\n")
delLB(L,B,n)
print("\n")
printDelLB()