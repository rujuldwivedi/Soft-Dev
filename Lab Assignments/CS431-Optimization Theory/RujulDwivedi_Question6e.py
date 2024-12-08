import numpy as np
import matplotlib.pyplot as plt

# Load dataset
data = np.genfromtxt("RujulDwivedi_Question6.txt")

# Split dataset into features and labels
X = data[:, :-1]
y = data[:, -1]

# Split dataset into training and test sets
np.random.seed(42)  # for reproducibility
indices = np.random.permutation(X.shape[0])
train_indices, test_indices = indices[:1400], indices[1400:]
X_train, y_train = X[train_indices], y[train_indices]

# Separate points by class
X_pos = X_train[y_train == 1]
X_neg = X_train[y_train == -1]

# Add intercept term to features
X_train = np.concatenate((np.ones((X_train.shape[0], 1)), X_train), axis=1)

# Define sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Initialize coefficients
w = np.zeros(X_train.shape[1])

# Define negative log likelihood function
def negative_log_likelihood(X, y, w):
    z = np.dot(X, w)
    return -np.sum(np.log(sigmoid(y * z)))

# Gradient descent
learning_rate = 0.01
num_iterations = 1000

for i in range(num_iterations):
    # Calculate predictions
    predictions = sigmoid(np.dot(X_train, w))
    # Calculate error
    error = y_train - predictions
    # Update coefficients
    gradient = np.dot(X_train.T, error)
    w += learning_rate * gradient

# Calculate initial decision boundary
x1_values = np.array([np.min(X_train[:, 1]), np.max(X_train[:, 1])])
initial_decision_boundary = -(w[0] + w[1] * x1_values) / w[2]

# Plot training set
plt.figure(figsize=(8, 6))
plt.scatter(X_pos[:, 0], X_pos[:, 1], color='blue', label='Class +1')
plt.scatter(X_neg[:, 0], X_neg[:, 1], color='red', label='Class -1')
plt.plot(x1_values, initial_decision_boundary, linestyle='--', color='green', label='Initial Decision Boundary')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Scatter Plot of Training Set with Decision Boundaries')
plt.legend()
plt.grid(True)
plt.show()