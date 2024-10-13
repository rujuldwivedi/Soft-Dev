import numpy as np

# Load the dataset
data = np.loadtxt('RujulDwivedi_Question6.txt')

# Extract features (x1, x2) and labels (y) from the dataset
X = data[:, :2]  # Features
y = data[:, 2]   # Labels

# Split the dataset into training set (1400 points) and test set (600 points)
X_train, X_test = X[:1400], X[1400:]
y_train, y_test = y[:1400], y[1400:]

# Define sigmoid function and its derivative
def sigmoid(a):
    return 1 / (1 + np.exp(-a))

def sigmoid_prime(a):
    return np.exp(-a) / ((1 + np.exp(-a)) ** 2)

# Define negative log-likelihood function
def negative_log_likelihood(w, X, y):
    z = y * np.dot(X, w)
    return -np.sum(np.log(sigmoid(z)))

# Compute gradient of negative log-likelihood function
def gradient_negative_log_likelihood(w, X, y):
    z = y * np.dot(X, w)
    sigmoid_z = sigmoid(z)
    return -np.dot(X.T, (1 / sigmoid_z) * sigmoid_prime(z) * y)

# Compute Hessian of negative log-likelihood function using second-order finite differences
def hessian_negative_log_likelihood(w, X, y, epsilon=1e-6):
    grad = gradient_negative_log_likelihood(w, X, y)
    hessian = np.zeros((len(w), len(w)))
    for i in range(len(w)):
        for j in range(i, len(w)):
            # Compute second-order partial derivative using central difference
            w1 = np.copy(w)
            w2 = np.copy(w)
            w1[i] += epsilon
            w2[j] += epsilon
            l_w1w2 = negative_log_likelihood(w1, X, y)
            l_w1 = negative_log_likelihood(w1, X, y)
            l_w2 = negative_log_likelihood(w2, X, y)
            l_w = negative_log_likelihood(w, X, y)
            hessian[i, j] = (l_w1w2 - l_w1 - l_w2 + l_w) / (epsilon ** 2)
            hessian[j, i] = hessian[i, j]  # Since Hessian is symmetric
    return hessian

# Test if l(w) is convex
is_convex = True
for i in range(X_train.shape[0]):
    w = np.random.randn(X_train.shape[1])
    gradient = gradient_negative_log_likelihood(w, X_train, y_train)
    hessian = hessian_negative_log_likelihood(w, X_train, y_train)
    print(f"\nSample {i+1}:")
    print("Gradient:")
    print(gradient)
    print("\nHessian:")
    print(hessian)
    if not np.all(np.linalg.eigvals(hessian) >= 0):
        is_convex = False
        break

if is_convex:
    print("\nThe Hessian matrix of the negative log-likelihood function l(w) is positive semi-definite for all w ∈ R^m.")
    print("The negative log-likelihood function l(w) is convex over w ∈ R^m.")
else:
    print("\nThe Hessian matrix of the negative log-likelihood function l(w) is not positive semi-definite for all w ∈ R^m.")
    print("The negative log-likelihood function l(w) is not convex over w ∈ R^m.")
