import numpy as np

# Given data points
data_points = np.array([[0, 0], [1, 3], [2, 7], [3, -1], [4, 0], [5, 5], [6, 10]])

# Extracting x values
x_values = data_points[:, 0]

# Constructing matrix A
A = np.vstack([np.ones_like(x_values), x_values, x_values**2, x_values**3]).T

# Regularization parameter lambda
lambdas = [0, 1, 10, 1000]

# Perform Newton's method for each lambda
for lam in lambdas:
    # Initial guess for coefficients
    c = np.zeros(4)
    # Convergence threshold
    epsilon = 1e-6
    # Initialize iteration counter
    iterations = 0
    
    # Newton's method loop
    while True:
        # Compute gradient
        grad = 2 * A.T.dot(A.dot(c) - data_points[:, 1]) + 2 * lam * c
        # Compute Hessian
        Hessian = 2 * A.T.dot(A) + 2 * lam * np.eye(4)
        # Solve linear system for step direction
        step = np.linalg.solve(Hessian, -grad)
        # Update coefficients
        c += step
        # Check for convergence
        if np.linalg.norm(step) < epsilon:
            break
        # Increment iteration counter
        iterations += 1
    
    # Print results
    print(f"Lambda = {lam}: Coefficients = {c}, Iterations = {iterations}")