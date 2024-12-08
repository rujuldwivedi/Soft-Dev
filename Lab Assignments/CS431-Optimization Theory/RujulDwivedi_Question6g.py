import numpy as np

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

# Add intercept term to training set features
X_train = np.concatenate((np.ones((X_train.shape[0], 1)), X_train), axis=1)

# Define sigmoid function with clipping
def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

# Define negative log likelihood function
def negative_log_likelihood(X, y, w):
    z = np.dot(X, w)
    return -np.sum(np.log(sigmoid(y * z)))

# Gradient descent function
def gradient_descent(X, y, learning_rate, num_iterations, w_initial):
    w = w_initial.copy()
    losses = []
    for i in range(num_iterations):
        # Calculate predictions
        predictions = sigmoid(np.dot(X, w))
        # Calculate error
        error = y - predictions
        # Update coefficients
        gradient = np.dot(X.T, error)
        w += learning_rate * gradient
        # Calculate and store negative log likelihood
        loss = negative_log_likelihood(X, y, w)
        losses.append(loss)
    return w, losses

# Perform gradient descent with five different starting points
starting_points = [
    np.zeros(X_train.shape[1]),
    np.ones(X_train.shape[1]),
    np.random.normal(size=X_train.shape[1]),
    np.random.uniform(size=X_train.shape[1]),
    np.random.uniform(-1, 1, size=X_train.shape[1])
]

for i, start_point in enumerate(starting_points):
    print(f"Starting Point {i + 1}: {start_point}")
    w_optimal, losses = gradient_descent(X_train, y_train, learning_rate=0.01, num_iterations=1000, w_initial=start_point)
    print(f"Optimal Coefficients: {w_optimal}")
    print(f"Final Negative Log Likelihood: {losses[-1]}")
    print()
