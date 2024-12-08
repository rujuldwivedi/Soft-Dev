import numpy as np # Importing numpy library
import gurobipy as gp # Importing gurobipy library

def sendMaxCommodity(V, E): # Function to solve the first problem
    m = gp.Model() # Creating a model

    # Extracting unique vertices
    vertices = set() # Creating a set to store unique vertices
    for edge in E: # Looping through the edges
        vertices.add(edge[0]) # Adding the first vertex of the edge to the set
        vertices.add(edge[1]) # Adding the second vertex of the edge to the set

    # Variables
    x = {} # Creating a dictionary to store the flow through each edge
    for edge in E: # Looping through the edges
        x[edge] = m.addVar(lb=0, ub=edge[2], name=f"x_{edge[0]}_{edge[1]}") # Adding the flow variable for each edge

    v = m.addVar(name="v") # Adding the commodity variable

    # Constraints
    for i, vertex in enumerate(V): # Looping through the vertices
        if vertex == V[0]: # If the vertex is the source
            m.addConstr(gp.quicksum(x[edge] for edge in E if edge[0] == vertex) - gp.quicksum(x[edge] for edge in E if edge[1] == vertex) == v) # Adding the constraint for the source
        elif vertex == V[-1]: # If the vertex is the sink
            m.addConstr(gp.quicksum(x[edge] for edge in E if edge[0] == vertex) - gp.quicksum(x[edge] for edge in E if edge[1] == vertex) == -v) 
        else: # If the vertex is not the source or the sink
            m.addConstr(gp.quicksum(x[edge] for edge in E if edge[0] == vertex) - gp.quicksum(x[edge] for edge in E if edge[1] == vertex) == 0) # Adding the constraint for the other vertices

    # Objective
    m.setObjective(v, gp.GRB.MAXIMIZE) # Setting the objective function

    m.optimize() # Solving the model

    return m, x, v # Returning the model, the flow through each edge and the commodity variable

# Example usage with the provided graph for function 1
V = ['S','A','B','C','D','E','F','G','H','I','J','T']
E = [('S','A',10),('S','D',5),('S','C',2),('S','E',4),('A','B',4),('A','G',2),('A','C',2),('C','G',8),('C','E',5),('E','G',7),('E','F',2),('D','J',2),('D','E',3),('J','F',1),('J','I',5),('F','T', 6),('F','I',10),('G','F',2),('G','H',2),('B','H',3),('I','T',7),('H','T',9),('T','G',6)]

m, x, v = sendMaxCommodity(V, E) # Solving the first problem

print("Objective value:", v.x) # Printing the objective value
print("The flow through each edge is:") # Printing the flow through each edge
for edge in x: # Looping through the edges
    print(f"Edge: {edge[0]} {edge[1]}, Flow: {x[edge].x}") # Printing the flow through each edge

# P.S. I HAVE IMPLEMENTED THE MAX FLOW PROBLEM (SECOND PART) BELOW THIS COMMENT.

# Below is the code for the min flow problem though it is not required for the question
# as I believe it is a mistake in the question. It should be a max flow problem instead of min flow problem.
    
# import numpy as np
# import gurobipy as gp

# def solve_min_flow(graph):
#     model = gp.Model()
#     num_nodes = len(graph)

#     # Variables
#     u_var = model.addMVar(num_nodes, vtype=gp.GRB.CONTINUOUS, name="u")

#     # Flow variables
#     flow_vars = {}
#     for i in range(num_nodes):
#         for j in range(num_nodes):
#             if graph[i][j] > 0:
#                 flow_vars[(i, j)] = model.addVar(vtype=gp.GRB.CONTINUOUS, name=f"x_{i}_{j}")

#     model.update()

#     # Constraints
#     for i in range(num_nodes):
#         for j in range(num_nodes):
#             if graph[i][j] > 0:
#                 model.addConstr(u_var[i] - u_var[j] + flow_vars[(i, j)] >= 0, name=f"constraint_{i}_{j}")

#     model.addConstr(u_var[0] - u_var[-1] == 1, name="source_sink_constraint")

#     # Objective
#     obj_expr = gp.quicksum(graph[i][j] * flow_vars[(i, j)] for i in range(num_nodes) for j in range(num_nodes) if graph[i][j] > 0)
#     model.setObjective(obj_expr, gp.GRB.MINIMIZE)

#     model.optimize()

#     return model, u_var, flow_vars

# graph_input = np.array([
#     [0, 5, 6, 8, 10, 0, 0, 0, 0],
#     [0, 0, 0, 0, 0, 4, 3, 0, 0],
#     [0, 0, 0, 0, 0, 0, 0, 0, 0],
#     [0, 0, 0, 0, 0, 0, 7, 0, 0],
#     [0, 0, 0, 0, 0, 0, 9, 4, 10],
#     [0, 0, 0, 0, 0, 0, 0, 0, 0],
#     [0, 0, 0, 0, 0, 0, 0, 16, 0],
#     [0, 0, 0, 0, 0, 0, 0, 0, 8],
#     [0, 0, 0, 0, 0, 0, 0, 0, 0] 
# ]) # Creating the graph

# model, u_result, flow_result = solve_min_flow(graph_input)

# print("Objective value:", model.objVal)
# print("Optimal solution (u):", u_result.X)
# print("Optimal solution (flow variables):")
# for edge, var in flow_result.items():
#     print(f"Edge {edge}: {var.x}")


def solve_max_flow(graph): # Function to solve the second problem
    model = gp.Model() # Creating a model
    num_nodes = len(graph) # Getting the number of nodes

    # Variables
    zeros_matrix = np.zeros((num_nodes, num_nodes)) # Creating a matrix of zeros

    u_var = model.addMVar((num_nodes,), vtype=gp.GRB.CONTINUOUS) # Adding the u variable
    Y_var = model.addMVar((num_nodes, num_nodes), lb=zeros_matrix, vtype=gp.GRB.CONTINUOUS) # Adding the Y variable

    # Constraints
    for i, row in enumerate(graph): # Looping through the rows of the graph
        for j, elem in enumerate(row): # Looping through the elements of the row
            if elem != 0: # If the element is not zero
                model.addConstr(u_var[i] - u_var[j] + Y_var[i][j] >= 0) # Adding the constraint
    model.addConstr(-u_var[0] + u_var[-1] == 1) # Adding the constraint

    # Objective
    model.setObjective(gp.quicksum((np.array(graph) @ Y_var.T)[i][i] for i in range(num_nodes)), gp.GRB.MINIMIZE) # Setting the objective function

    model.optimize() # Solving the model

    return model, u_var, Y_var # Returning the model, the u variable and the Y variable

graph_input = np.array([
    [0, 5, 6, 8, 10, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 4, 3, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 7, 0, 0],
    [0, 0, 0, 0, 0, 0, 9, 4, 10],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 16, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 8],
    [0, 0, 0, 0, 0, 0, 0, 0, 0] 
]) # Creating the graph

model, u_result, Y_result = solve_max_flow(graph_input) # Solving the second problem

print("Objective value:", model.objVal) # Printing the objective value
print("Optimal solution (u):", u_result.X.tolist()) # Printing the optimal solution for u
print("Optimal solution (Y):", Y_result.X) # Printing the optimal solution for Y
# convert y to integer
Y_result_rounded = np.round(Y_result.X) # Rounding the optimal solution for Y
print("Optimal solution (Y) rounded:", Y_result_rounded) # Printing the rounded optimal solution for Y