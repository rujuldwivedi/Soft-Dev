import numpy as np  # Importing numpy library for numerical operations
import gurobipy as gp  # Importing gurobipy library for optimization

# (b) Solving the linear programming problem
A = np.array([[11, 53, 5, 5, 29],
              [3, 6, 5, 1, 34],
              [1, 0, 0, 0, 0],
              [0, 1, 0, 0, 0],
              [0, 0, 1, 0, 0],
              [0, 0, 0, 1, 0],
              [0, 0, 0, 0, 1]])
b = np.array([[40], [20], [1], [1], [1], [1], [1]])
c = np.array([[13], [16], [16], [14], [39]])

lp = gp.Model()  # Creating a new optimization model for linear programming problem

# Setting variables and constraints
X = lp.addMVar((5, 1), lb=np.zeros((5, 1)), vtype=gp.GRB.CONTINUOUS)
lp.addConstr(A @ X <= b)

# Setting objective function
lp.setObjective(c.T @ X, gp.GRB.MAXIMIZE)

# Solving the linear programming problem
lp.optimize()

# Printing results
print("Maximized total profit:", lp.objVal, "m")
investments = ["A", "B", "C", "D", "E"]
print("Optimal portfolio for the firm:")
for i in range(5):
    print(f"{X[i].X[0] * 100}% commitment to {investments[i]}")

# (c) Obtaining dual variable values
lp_dual_values = lp.Pi  # This approximates the values of dual variables

# (d) Solving the dual problem
lp_dual = gp.Model()  # Creating a new optimization model for the dual problem

# Setting variables and constraints
Y = lp_dual.addMVar((7, 1), lb=np.zeros((7, 1)))
lp_dual.addConstr(A.T @ Y >= c)

# Setting objective function
lp_dual.setObjective(b.T @ Y, gp.GRB.MINIMIZE)

# Solving the dual problem
lp_dual.optimize()

# Printing results
print("Optimal solution of the dual:", Y.X.reshape((-1,)))

# As we can see, the value obtained is similar to the value obtained from Pi attribute earlier