from re import L
import matplotlib.pyplot as plt

def _factorial(n):
    x = 1
    for i in range(1, n+1):
        x *= i
    return x

def _combination(n, k):
    return _factorial(n)/(_factorial(n-k)*_factorial(k))

def _pmf_binomial(n, k, p):
    return _combination(n, k)*(p**k)*((1-p)**(n-k))

def pmf_hist(n, p):
    r= list(range(n + 1))
    dist = [_pmf_binomial(n, k, p) for k in r]
    plt.bar(r, dist)
    plt.show()

def cdf_hist(n, k, p):
    pVal = 0
    r= list(range(k + 1))
    dist = []
    for i in range(k+1):
        pmf_i = _pmf_binomial(n, i, p)
        pVal += pmf_i
        dist.append(pVal)
    plt.bar(r, dist)
    plt.show()

def __init__():
    n = int(input("Enter number of trials (n): "))
    k = int(input("Enter number whose cdf you want to find (k): "))
    p = float(input("Enter probability (p): "))
    pmf_hist(n, p)
    cdf_hist(n, k, p)
__init__()