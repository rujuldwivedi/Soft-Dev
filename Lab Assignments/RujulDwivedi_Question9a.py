import gurobipy as gp
from gurobipy import GRB

# Data
chores = ['Shopping', 'Cooking', 'Cleaning', 'Laundry']
people = ['Rujul', 'Priyanshu'] # I am Rujul and my room-mate is Priyanshu
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
model = gp.Model('ChoresAssignment')

# Decision variables
x = {}
for person in people:
    for chore in chores:
        x[(person, chore)] = model.addVar(vtype=GRB.BINARY, name=f'x_{person}_{chore}')

# Objective function: minimize total time
model.setObjective(gp.quicksum(x[(person, chore)] * time_taken[(person, chore)] for person in people for chore in chores), GRB.MINIMIZE)

# Constraints: each chore is assigned to exactly one person
for chore in chores:
    model.addConstr(gp.quicksum(x[(person, chore)] for person in people) == 1)

# Constraints: each person performs at most two chores
for person in people:
    model.addConstr(gp.quicksum(x[(person, chore)] for chore in chores) <= 2)

# Constraints: all chores must be completed
model.addConstrs((gp.quicksum(x[(person, chore)] for person in people) == 1 for chore in chores))

# Optimize model
model.optimize()

# Print optimal solution
print('\nOptimal assignment:')
for person in people:
    for chore in chores:
        if x[(person, chore)].x > 0.5:
            print(f'{person} performs {chore}')