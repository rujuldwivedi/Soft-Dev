import numpy as np

def my_multivariate_normal(x, mean, covariance):
    d = len(mean)
    constant_term = 1 / ((2 * np.pi) ** (d / 2) * np.linalg.det(covariance) ** 0.5)
    exponent_term = -0.5 * (x - mean).T @ np.linalg.inv(covariance) @ (x - mean)
    return constant_term * np.exp(exponent_term)

def compute_prior(w, prior_mean, prior_covariance):
    return my_multivariate_normal(w, prior_mean, prior_covariance)

def compute_likelihood(X, Y, W, B, noise_variance):
    n = len(Y)
    error = Y - (X @ W + B)
    constant_term = 1 / ((2 * np.pi * noise_variance) ** (n / 2))
    exponent_term = -0.5 * (error.T @ np.linalg.inv(noise_variance * np.eye(n)) @ error)
    return constant_term * np.exp(exponent_term)

def compute_posterior(X, Y, prior_mean, prior_covariance, noise_variance):
    prior_precision = np.linalg.inv(prior_covariance)
    likelihood_precision = np.linalg.inv(noise_variance * np.eye(len(Y)))
    
    posterior_precision = prior_precision + X.T @ likelihood_precision @ X
    posterior_covariance = np.linalg.inv(posterior_precision)
    
    posterior_mean = posterior_covariance @ (prior_precision @ prior_mean + X.T @ likelihood_precision @ Y)
    
    return posterior_mean, posterior_covariance

def bayes_classifier(X, posterior_mean, posterior_covariance, threshold=0.5):
    # Compute the predicted probability for each data point
    predicted_probabilities = 1 / (1 + np.exp(-(X @ posterior_mean)))
    
    # Classify based on the threshold
    predicted_classes = (predicted_probabilities > threshold).astype(int)
    
    return predicted_classes

# Example usage
n = 100
d = 3
np.random.seed(42)

# Generate random data for binary classification
X = np.random.randn(n, d)
B = np.random.randn(n, 1)
Y = (np.random.randn(n, 1) > 0).astype(int)  # Binary labels (0 or 1)

# Prior parameters
prior_mean = np.zeros((d, 1))
prior_covariance = np.eye(d)

# Likelihood parameters
noise_variance = 1.0

# Compute posterior distribution
posterior_mean, posterior_covariance = compute_posterior(X, Y, prior_mean, prior_covariance, noise_variance)

# Apply Bayes classifier
predicted_classes = bayes_classifier(X, posterior_mean, posterior_covariance)

print("Posterior Mean:\n", posterior_mean)
print("Posterior Covariance:\n", posterior_covariance)
print("Predicted Classes:\n", predicted_classes)