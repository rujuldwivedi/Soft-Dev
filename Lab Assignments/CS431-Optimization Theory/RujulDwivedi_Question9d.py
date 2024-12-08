import gurobipy as gp
from gurobipy import GRB

# Data
chores = ['Shopping', 'Cooking', 'Cleaning', 'Laundry']
people = ['Rujul', 'Priyanshu'] #I am Rujul and Priyanshu is my room-mate
time_taken = {
    ('Rujul', 'Shopping'): 4.5,
    ('Rujul', 'Cooking'): 7.8,
    ('Rujul', 'Cleaning'): 3.6,
    ('Rujul', 'Laundry'): 2.9,
    ('Priyanshu', 'Shopping'): 4.9,
    ('Priyanshu', 'Cooking'): 7.2,
    ('Priyanshu', 'Cleaning'): 4.3,
    ('Priyanshu', 'Laundry'): 3.1
}

# Create model
model = gp.Model('ChoresAssignment_Difference_Minimization')

# Decision variables
x = {}
for person in people:
    for chore in chores:
        x[(person, chore)] = model.addVar(vtype=GRB.BINARY, name=f'x_{person}_{chore}')

# Define absolute difference variable
delta = model.addVar(name='delta')

# Define auxiliary variables
T_Y = model.addVar(name='T_Y')
T_R = model.addVar(name='T_R')

# Objective function: minimize absolute difference
model.setObjective(delta, GRB.MINIMIZE)

# Constraints: each chore is assigned to exactly one person
for chore in chores:
    model.addConstr(gp.quicksum(x[(person, chore)] for person in people) == 1)

# Constraints: each person performs at most two chores
for person in people:
    model.addConstr(gp.quicksum(x[(person, chore)] for chore in chores) <= 2)

# Constraints: all chores must be completed
model.addConstrs((gp.quicksum(x[(person, chore)] for person in people) == 1 for chore in chores))

# Constraint: define T_Y and T_R
model.addConstr(T_Y == gp.quicksum(x[('Rujul', chore)] * time_taken[('Rujul', chore)] for chore in chores))
model.addConstr(T_R == gp.quicksum(x[('Priyanshu', chore)] * time_taken[('Priyanshu', chore)] for chore in chores))

# Constraint: define absolute difference
model.addConstr(delta >= T_Y - T_R)
model.addConstr(delta >= T_R - T_Y)

# Optimize model
model.optimize()

# Print optimal assignment
print('\nOptimal assignment:')
for person in people:
    for chore in chores:
        if x[(person, chore)].x > 0.5:
            print(f'{person} performs {chore}')

# Calculate total time spent collectively on chores
total_time_collectively = sum(x[(person, chore)].x * time_taken[(person, chore)] for person in people for chore in chores)

# Calculate time spent by each person
time_spent = {person: sum(x[(person, chore)].x * time_taken[(person, chore)] for chore in chores) for person in people}

# Print total time spent and time spent by each person
print(f'\nTotal time spent collectively on chores in a week: {total_time_collectively} hours')
for person in people:
    print(f'Time spent by {person}: {time_spent[person]} hours')

# Print absolute difference between times spent by both of Rujul and Priyanshu
print(f'\nAbsolute difference between times spent by both of Rujul and Priyanshu: {delta.x} hours')