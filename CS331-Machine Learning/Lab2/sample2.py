import sample2
# import time
from random import randint

def generaterandommatrix(row,col):
    M = []
    for i in range (row):
        t = []
        for j in range(col):
            t.append(randint(0,10))
        M.append(t)
    return M

A = generaterandommatrix(500,500)
B = generaterandommatrix(500,500)

assert 'multiply' in dir(sample2)
assert callable(sample2.multiply)

print(sample2.multiply(A,B))
#C = new.multiply(A,B)
