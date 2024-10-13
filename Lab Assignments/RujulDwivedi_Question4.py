import gurobipy as gp  # Importing gurobipy library for optimization

# (b) Defining pij based on the graph given in the assignment
p14 = 2
p24 = 3
p26 = 2
p34 = 1
p35 = 1

model = gp.Model()  # Creating a new optimization model

# Defining variables and constraints
x14 = model.addVar(name="x14", lb=0)  # Adding decision variable x14 with lower bound 0
x24 = model.addVar(name="x24", lb=0)  # Adding decision variable x24 with lower bound 0
x26 = model.addVar(name="x26", lb=0)  # Adding decision variable x26 with lower bound 0
x34 = model.addVar(name="x34", lb=0)  # Adding decision variable x34 with lower bound 0
x35 = model.addVar(name="x35", lb=0)  # Adding decision variable x35 with lower bound 0

model.addConstr(x14 <= 1)  # Adding constraint: x14 <= 1
model.addConstr(x24 + x26 <= 1)  # Adding constraint: x24 + x26 <= 1
model.addConstr(x34 + x35 <= 1)  # Adding constraint: x34 + x35 <= 1
model.addConstr(x14 + x24 + x34 <= 1)  # Adding constraint: x14 + x24 + x34 <= 1
model.addConstr(x35 <= 1)  # Adding constraint: x35 <= 1
model.addConstr(x26 <= 1)  # Adding constraint: x26 <= 1

# Setting objective function
model.setObjective(x14 * p14 + x24 * p24 + x26 * p26 + x34 * p34 + x35 * p35, gp.GRB.MAXIMIZE)

# Solving the optimization problem
model.optimize()

# Printing results
print("Max possible reward to the company:", model.ObjVal)
print("Optimal Candidate Job Assignments:")
if x14.X == 1:
    print("Candidate 1 gets Job 4")
if x24.X == 1:
    print("Candidate 2 gets Job 4")
if x26.X == 1:
    print("Candidate 2 gets Job 6")
if x34.X == 1:
    print("Candidate 3 gets Job 4")
if x35.X == 1:
    print("Candidate 3 gets Job 5")

# (c) Creating the dual problem
dual = gp.Model()  # Creating a new optimization model for the dual problem

# Defining variables and constraints
y1 = dual.addVar(name="y1", lb=0)  # Adding decision variable y1 with lower bound 0
y2 = dual.addVar(name="y2", lb=0)  # Adding decision variable y2 with lower bound 0
y3 = dual.addVar(name="y3", lb=0)  # Adding decision variable y3 with lower bound 0
z4 = dual.addVar(name="z4", lb=0)  # Adding decision variable z4 with lower bound 0
z5 = dual.addVar(name="z5", lb=0)  # Adding decision variable z5 with lower bound 0
z6 = dual.addVar(name="z6", lb=0)  # Adding decision variable z6 with lower bound 0

# Adding constraints
dual.addConstr(y1 + z4 >= p14)
dual.addConstr(y2 + z4 >= p24)
dual.addConstr(y2 + z6 >= p26)
dual.addConstr(y3 + z4 >= p34)
dual.addConstr(y3 + z5 >= p35)

# Setting objective function
dual.setObjective(y1 + y2 + y3 + z4 + z5 + z6, gp.GRB.MINIMIZE)

# Solving the dual problem
dual.optimize()

# As we can see, the optimal value of both primal and dual is the same, therefore strong duality is verified