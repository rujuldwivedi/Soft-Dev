import gurobipy as gp
from gurobipy import GRB

# Create a new model
model = gp.Model("quadratic_minimization")

# Create variables
x = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name="x")
y = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name="y")

# Set objective: minimize x^2 + y^2
model.setObjective(x*x + y*y, GRB.MINIMIZE)

# Add constraint: x - y = 1
model.addConstr(x + y == 1, "constraint1")

# Optimize the model
model.optimize()

# Print the optimal values of variables
print(f"Optimal value of x: {x.x}")
print(f"Optimal value of y: {y.x}")

# Print the optimal objective value
print(f"Optimal objective value: {model.objVal}")