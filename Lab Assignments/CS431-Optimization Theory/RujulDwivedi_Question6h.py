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

# Define sigmoid function with clipping for numerical stability
def sigmoid(z):
    clipped_z = np.clip(z, -500, 500)  # Clip values to avoid overflow
    return 1 / (1 + np.exp(-clipped_z))

# Define negative log likelihood function with epsilon for numerical stability
def negative_log_likelihood(X, y, w):
    z = np.dot(X, w)
    predictions = sigmoid(y * z)
    epsilon = 1e-15  # Small constant to avoid taking the logarithm of zero
    return -np.sum(np.log(predictions.clip(epsilon, 1 - epsilon)))

# Gradient descent function with adaptive learning rate and convergence criterion
def gradient_descent_adaptive(X, y, learning_rate=0.01, tolerance=1e-4, max_iterations=1000):
    w = np.zeros(X.shape[1])
    losses = []
    # Calculate initial loss
    prev_loss = negative_log_likelihood(X, y, w)
    for i in range(max_iterations):
        # Calculate predictions
        predictions = sigmoid(np.dot(X, w))
        # Calculate error
        error = y - predictions
        # Update coefficients with adaptive learning rate
        gradient = np.dot(X.T, error)
        w += learning_rate * gradient
        # Calculate and store negative log likelihood
        loss = negative_log_likelihood(X, y, w)
        losses.append(loss)
        # Check convergence criterion
        if abs(prev_loss - loss) < tolerance:
            break
        prev_loss = loss
    return w, losses

# Test adaptive learning rate and convergence criterion
w_adaptive, losses_adaptive = gradient_descent_adaptive(X_train, y_train)

# Print results
print("Adaptive Learning Rate and Convergence Criterion:")
print("Optimal Coefficients:", w_adaptive)
print("Final Negative Log Likelihood:", losses_adaptive[-1])