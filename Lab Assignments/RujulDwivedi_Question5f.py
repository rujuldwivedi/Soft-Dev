import matplotlib.pyplot as plt
import numpy as np

# Given data points
data_points = np.array([[0, 0], [1, 3], [2, 7], [3, -1], [4, 0], [5, 5], [6, 10]])

# Extracting x values
x_values = data_points[:, 0]

# Constructing matrix A
A = np.vstack([np.ones_like(x_values), x_values, x_values**2, x_values**3]).T

# Regularization parameter lambda
lambdas = [0, 1, 10, 1000]

# Plot given data points
plt.scatter(data_points[:, 0], data_points[:, 1], label='Data Points')

# Plot polynomial for each lambda
for lam in lambdas:
    # Solve for coefficients using Newton's method
    c = np.linalg.inv(A.T @ A + lam * np.eye(4)) @ A.T @ data_points[:, 1]
    # Generate polynomial values
    x_values_range = np.linspace(0, 6, 100)
    y_values = np.polyval(c[::-1], x_values_range)
    plt.plot(x_values_range, y_values, label=f'Lambda = {lam}')

plt.xlabel('x')
plt.ylabel('y')
plt.title('Polynomial Fitting with Regularization')
plt.legend()
plt.grid(True)
plt.show()