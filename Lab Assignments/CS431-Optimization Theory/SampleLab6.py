import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.cluster import KMeans

# Load MNIST dataset
mnist = fetch_openml('mnist_784', version=1, as_frame=False)
X, y = mnist.data / 255., mnist.target.astype(int)

# Only using labels 2, 3, and 4
labels = [2, 3, 4]
X_filtered = np.vstack([X[y == label] for label in labels])
y_filtered = np.hstack([y[y == label] for label in labels])

# Binarize function
def binarize(X):
    return 1. * (X >= 0.5)

# Binarize the filtered data
X_binarized = binarize(X_filtered)

# K-means initialization for mu
def kmeans_initialization(X, K):
    kmeans = KMeans(n_clusters=K, n_init=10)  # Suppress the warning by explicitly setting n_init
    kmeans.fit(X)
    return kmeans.cluster_centers_

# EM Algorithm
def EM(X, K, max_iter, threshold=1e-3):
    # Initialize parameters
    mu = kmeans_initialization(X, K)
    pi = np.ones(K) / K

    # Iterate EM steps
    for _ in range(max_iter):
        # E-step
        eps = 1e-10  # for stability in log operations
        gamma = np.exp(np.log(pi + eps) + X @ np.log(mu.T + eps) + (1 - X) @ np.log(1 - mu.T + eps))
        gamma /= gamma.sum(axis=1)[:, np.newaxis]

        # M-step
        N_m = gamma.sum(axis=0)
        pi = N_m / X.shape[0]
        mu = gamma.T @ X / N_m[:, np.newaxis]

    return mu, pi

# Run EM algorithm
K = len(labels)
mu, pi = EM(X_binarized, K=K, max_iter=100)
print("Centroids:")
print(mu)
print("Pi's:")
print(pi)

# Visualize results
fig, ax = plt.subplots(nrows=1, ncols=K, figsize=(15, 15), dpi=100)
for i in range(K):
    ax[i].imshow(mu[i].reshape(28, 28), cmap='gray')
    ax[i].set_title('Parameters class: {}\n pi = {:0.3f}'.format(labels[i], pi[i]), fontsize=K ** (-1) // 0.02)
    ax[i].axis('off')
plt.show()