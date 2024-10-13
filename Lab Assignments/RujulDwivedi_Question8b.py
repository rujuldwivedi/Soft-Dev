import numpy as np
import matplotlib.pyplot as plt

# Define the Rosenbrock function
def rosenbrock(x1, x2):
    return (1 - x1)**2 + 100*(x2 - x1**2)**2

# Define the gradient of the Rosenbrock function
def gradient_rosenbrock(x1, x2):
    df_dx1 = -2 * (1 - x1) - 400 * x1 * (x2 - x1**2)
    df_dx2 = 200 * (x2 - x1**2)
    return np.array([df_dx1, df_dx2])

# Generate x and y values for plotting
x_values = np.linspace(-2, 2, 400)
y_values = np.linspace(-2, 2, 400)
X, Y = np.meshgrid(x_values, y_values)

# Generate contours for the Rosenbrock function
alphas = [0.5, 2, 5, 10, 50, 100, 200, 400, 800]
plt.figure(figsize=(10, 8))
for alpha in alphas:
    Z = rosenbrock(X, Y)
    plt.contour(X, Y, Z, levels=[alpha], colors='black')
    plt.annotate(f'α={alpha}', xy=(-1.5, 1.5), fontsize=10, color='black')

# Plot gradient vectors
x_points = [-1.5, -1, 0, 1, 1.5]
y_points = [1.5, 1, 0, -1, -1.5]
for x, y in zip(x_points, y_points):
    gradient = gradient_rosenbrock(x, y)
    plt.quiver(x, y, gradient[0], gradient[1], color='red', angles='xy', scale_units='xy', scale=30)

# Set plot labels and title
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Contours of Rosenbrock Function with Gradient Vectors')
plt.grid(True)
plt.xlim(-2, 2)
plt.ylim(-2, 2)
plt.gca().set_aspect('equal', adjustable='box')
plt.show()
