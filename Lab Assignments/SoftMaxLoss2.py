import numpy as np

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))  # for numerical stability
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

def cross_entropy_loss(y_true, y_pred):
    epsilon = 1e-15  # small value to avoid log(0)
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.sum(y_true * np.log(y_pred)) / len(y_true)

def softmax_with_cross_entropy_loss(X, W, B, y_true):
    # Forward pass
    logits = np.dot(X, W) + B
    probabilities = softmax(logits)
    loss = cross_entropy_loss(y_true, probabilities)

    # Backward pass
    dL_dz = probabilities - y_true
    dL_dW = np.dot(X.T, dL_dz)
    dL_dB = np.sum(dL_dz, axis=0)

    return loss, dL_dW, dL_dB

def calculate_gradients_finite_differences(X, W, B, y_true, epsilon=1e-5):
    loss, dL_dW, dL_dB = softmax_with_cross_entropy_loss(X, W, B, y_true)

    # Calculate gradients using finite differences
    gradients_fd_W = np.zeros_like(W)
    gradients_fd_B = np.zeros_like(B)

    for i in range(W.shape[0]):
        for j in range(W.shape[1]):
            W_plus_epsilon = W.copy()
            W_minus_epsilon = W.copy()

            W_plus_epsilon[i, j] += epsilon
            W_minus_epsilon[i, j] -= epsilon

            loss_plus_epsilon, _, _ = softmax_with_cross_entropy_loss(X, W_plus_epsilon, B, y_true)
            loss_minus_epsilon, _, _ = softmax_with_cross_entropy_loss(X, W_minus_epsilon, B, y_true)

            gradients_fd_W[i, j] = (loss_plus_epsilon - loss_minus_epsilon) / (2 * epsilon)

    for i in range(B.shape[0]):
        B_plus_epsilon = B.copy()
        B_minus_epsilon = B.copy()

        B_plus_epsilon[i] += epsilon
        B_minus_epsilon[i] -= epsilon

        loss_plus_epsilon, _, _ = softmax_with_cross_entropy_loss(X, W, B_plus_epsilon, y_true)
        loss_minus_epsilon, _, _ = softmax_with_cross_entropy_loss(X, W, B_minus_epsilon, y_true)

        gradients_fd_B[i] = (loss_plus_epsilon - loss_minus_epsilon) / (2 * epsilon)

    return gradients_fd_W, gradients_fd_B

# Generate some random data for testing
np.random.seed(42)
X = np.random.rand(100, 3)
W = np.random.rand(3, 4)
B = np.random.rand(4)
y_true = np.eye(4)[np.random.choice(4, 100)]

# Calculate gradients using both methods
_, dL_dW, dL_dB = softmax_with_cross_entropy_loss(X, W, B, y_true)
gradients_fd_W, gradients_fd_B = calculate_gradients_finite_differences(X, W, B, y_true)

# Compare the results
print("Gradient with formula:\n", dL_dW, "\n", dL_dB)
print("\nGradient with finite differences:\n", gradients_fd_W, "\n", gradients_fd_B)
