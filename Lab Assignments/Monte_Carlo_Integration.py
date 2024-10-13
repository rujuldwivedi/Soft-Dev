import random
import matplotlib.pyplot as plt
import math as mth

from unittest import result
a = int(input("Enter lower bound: "))
b = int(input("Enter upper bound: "))
N = int(input("Enter number of trials: "))

def f(x): 
    return mth.sqrt(mth.sin(x))

def monty_carlo_integration(): 
    list = [0 for i in range(N)]
    for i in range (len(list)):
        list[i] = random.uniform(a,b)
    integral = 0.00
    for i in list:
        integral += f(i)
        result = (b-a)/float(N)*integral
    return result

def histogram(user):
    if (user == "yes"):
        values = []
        for i in range(N):
            result = monty_carlo_integration()
            values.append(result)
        plt.title("Distribution")
        plt.hist (values, bins=30, ec="blue") 
        plt.xlabel("Regions")
        plt.show()
    else:
        return

def plot():
    action = monty_carlo_integration()
    print(f"The monty carlo integration of \u221Asin(x) is {action}")
    user = input('Type "yes" to see the histogram\n')
    histogram(user)
plot()