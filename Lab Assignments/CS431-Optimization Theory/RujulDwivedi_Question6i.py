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

# Define sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Define negative log likelihood function
def negative_log_likelihood(X, y, w):
    z = np.dot(X, w)
    return -np.sum(np.log(sigmoid(y * z)))
i=0

# Newton's method update rule for logistic regression
def newtons_method(X, y, max_iterations=100, tolerance=1e-6):
    w = np.zeros(X.shape[1])  # Initialize weights
    prev_loss = np.inf  # Initialize previous loss
    for i in range(max_iterations):
        # Calculate predictions
        predictions = sigmoid(np.dot(X, w))
        # Calculate error
        error = y - predictions
        # Calculate gradient
        gradient = np.dot(X.T, error)
        # Calculate Hessian
        W = np.diag(predictions * (1 - predictions))
        hessian = np.dot(X.T, np.dot(W, X))
        # Update weights using Newton's method
        w -= np.linalg.solve(hessian, gradient)
        # Calculate current loss
        loss = negative_log_likelihood(X, y, w)
        # Check convergence
        if abs(prev_loss - loss) < tolerance:
            print(f"Converged after {i+1} iterations.")
            break
        prev_loss = loss
    else:
        print("Warning: Did not converge within the maximum number of iterations.")
    return w

# Run Newton's method
w_newton = newtons_method(X_train, y_train)

# Print results
print("Optimal Coefficients:", w_newton)

#Printing the number of iterations
print("Number of iterations:", i+1)