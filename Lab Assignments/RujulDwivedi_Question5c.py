import numpy as np

# Given data points
data_points = np.array([[0, 0], [1, 3], [2, 7], [3, -1], [4, 0], [5, 5], [6, 10]])

# Extracting x values
x_values = data_points[:, 0]

# Constructing matrix A
A = np.vstack([np.ones_like(x_values), x_values, x_values**2, x_values**3]).T

# Regularization parameter lambda
lambdas = [0, 1, 10, 1000]

# Perform gradient descent for each lambda
for lam in lambdas:
    # Initial guess for coefficients
    c = np.zeros(4)
    # Learning rate
    alpha = 0.001
    # Convergence threshold
    epsilon = 1e-6
    # Initialize previous step
    prev_step = np.inf
    # Initialize iteration counter
    iterations = 0
    
    # Gradient descent loop
    while True:
        # Compute gradient
        grad = 2 * A.T.dot(A.dot(c) - data_points[:, 1]) + 2 * lam * c
        # Compute exact line search step size
        step_size = np.linalg.norm(grad)**2 / (2 * np.linalg.norm(A.dot(grad))**2)
        # Update coefficients
        c -= step_size * grad
        # Compute Euclidean norm of gradient
        grad_norm = np.linalg.norm(grad)
        # Check for convergence
        if grad_norm < epsilon or grad_norm >= prev_step:
            break
        # Update previous gradient norm
        prev_step = grad_norm
        # Increment iteration counter
        iterations += 1
    
    # Print results
    print(f"Lambda = {lam}: Coefficients = {c}, Iterations = {iterations}")
