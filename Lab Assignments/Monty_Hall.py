import random
import matplotlib.pyplot as plt
import math as mth
from unittest import result
import numpy as np

N=int(input("Enter number of trials: "))

def monty_hall(n, door, switch):
    success = 0
    door = door.lower()
    switch = switch.lower()
    door_options = ["1", "2", "3", "4"]
    for n in range(0,n):
        universal = ["1", "2", "3", "4"] 
        unchosen = ["1", "2", "3", "4"] 
        unchosen.remove(door) 
        win = np.random.choice(door_options, 1)
        if door == win: 
            universal.remove(win)
        else:
            universal.remove(win)
            universal.remove(door)
        opened_door = np.random.choice(universal, 1) 
        unchosen.remove(opened_door)
        if switch == "no": 
            if door == win: 
                success += 1 
        if switch == "yes": 
            if unchosen[0] == win: 
                success += 1
    return float(success)/float(n)

def histogram(n,door,switch):
    values = []
    for i in range(N):
            result = monty_hall(n,door,switch)
            values.append(result)
            plt.title("Distribution")
            plt.hist (values, bins=30, ec="blue") 
            plt.xlabel("Regions")
            plt.show()

def plot():
    print("Choose one door from the following:\n")
    chosen_door = input("1\t2\t3\t4\n")
    decision = input("Do you want to switch (yes) or keep (no)? ")
    probability = monty_hall(10000, chosen_door, decision)
    print("Probability of winning is:", probability)
    histogram(N,chosen_door,decision)
plot()