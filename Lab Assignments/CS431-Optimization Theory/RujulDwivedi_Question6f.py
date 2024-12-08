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
X_test, y_test = X[test_indices], y[test_indices]

# Add intercept term to training set features
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

# Add intercept term to test set features
X_test = np.concatenate((np.ones((X_test.shape[0], 1)), X_test), axis=1)

# Predictions on test set
predictions_test = np.sign(np.dot(X_test, w))

# Compute misclassification error
misclassification_error = np.mean(predictions_test != y_test)
print("Misclassification error:", misclassification_error)