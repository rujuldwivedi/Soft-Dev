import numpy as np
import matplotlib.pyplot as plt

# Load the dataset
data = np.loadtxt('RujulDwivedi_Question6.txt')

# Extract features (X) and labels (y) from the dataset
X = data[:, :-1]
y = data[:, -1]

# Add intercept term to X
X = np.column_stack((np.ones(len(X)), X))

# Define sigmoid function
def sigmoid(a):
    return 1 / (1 + np.exp(-a))

# Define negative log-likelihood function
def negative_log_likelihood(w, X, y):
    z = y * np.dot(X, w)
    return -np.sum(np.log(sigmoid(z)))

# Compute gradient of negative log-likelihood function
def gradient_negative_log_likelihood(w, X, y):
    z = y * np.dot(X, w)
    sigmoid_z = sigmoid(z)
    return -np.dot(X.T, (1 / (sigmoid_z + 1e-12)) * (sigmoid_z - 1) * y)  # Adding small epsilon to avoid division by zero

# Steepest descent algorithm
def steepest_descent(X, y, alpha=0.01, tol=1e-6):
    # Initialize weight vector
    w = np.zeros(X.shape[1])
    iteration = 0
    gradient_norms = []
    
    # Iterate until convergence and finding the max iteration
    while True:
        # Compute gradient
        gradient = gradient_negative_log_likelihood(w, X, y)
        gradient_norm = np.linalg.norm(gradient)
        gradient_norms.append(gradient_norm)
        
        # Check for convergence
        if gradient_norm < tol:
            break
        
        # Update weight vector
        w -= alpha * gradient
        iteration += 1
        if iteration > 1000:
            break
    
    return w, gradient_norms

# Run steepest descent
optimal_w, gradient_norms = steepest_descent(X, y)

#Printing the max iterations and the optimal w
print(f"Max iterations: {len(gradient_norms)}")
print(f"Optimal w: {optimal_w}")

# Plot how l(w) changes in each iteration
plt.plot(range(len(gradient_norms)), gradient_norms)
plt.xlabel('Iteration')
plt.ylabel('Gradient Norm')
plt.title('Convergence of Steepest Descent')
plt.show()