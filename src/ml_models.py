"""
"""
import jax
import random
import numpy as np
import jax.numpy as jnp


class PCA:
    """
    A class for performing Principal Component Analysis (PCA) on data.

    Attributes:
        n_components (int): Number of principal components to retain.
        components (ndarray): Principal components learned from the data.
        mean (ndarray): Mean of the data used for centering.

    Methods:
        fit(X):
            Fit the PCA model on the input data.

        transform(X):
            Transform the input data into the PCA space.

        inverse_transform(X_transformed):
            Inverse transform PCA-transformed data back to the original space.

        sample(n_samples=1, key=None):
            Generate synthetic data samples from the learned PCA distribution.
    """

    def __init__(self, n_components):
        """
        Initialize the PCA class.

        Args:
            n_components (int): Number of principal components to retain.
        """
        self.n_components = n_components
        self.components = None
        self.mean = None

    def fit(self, X):
        """
        Fit the PCA model on the input data.

        Args:
            X (ndarray): Input data with shape (num_samples, num_features).

        Returns:
            None
        """
        self.mean = jnp.mean(X, axis=0)
        X_centered = X - self.mean
        cov_matrix = jnp.cov(X_centered, rowvar=False)
        eigvals, eigvecs = jnp.linalg.eigh(cov_matrix)
        sorted_indices = jnp.argsort(eigvals)[::-1]
        sorted_eigvecs = eigvecs[:, sorted_indices]
        self.components = sorted_eigvecs[:, :self.n_components]

    def transform(self, X):
        """
        Transform the input data into the PCA space.

        Args:
            X (ndarray): Input data with shape (num_samples, num_features).

        Returns:
            ndarray: Transformed data with shape (num_samples, n_components).
        """
        X_centered = X - self.mean
        return jnp.dot(X_centered, self.components)

    def inverse_transform(self, X_transformed):
        """
        Inverse transform PCA-transformed data back to the original space.

        Args:
            X_transformed (ndarray): Transformed data with shape (num_samples, n_components).

        Returns:
            ndarray: Inverse transformed data with shape (num_samples, num_features).
        """
        return jnp.dot(X_transformed, self.components.T) + self.mean

    def sample(self, n_samples=1, key=None):
        """
        Generate synthetic data samples from the learned PCA distribution.

        Args:
            n_samples (int, optional): Number of samples to generate.
            key (ndarray, optional): Random key for generating samples.

        Returns:
            ndarray: Generated samples with shape (n_samples, num_features).
        """
        if key is None:
            key = jax.random.PRNGKey(0)

        z = jax.random.normal(key, (n_samples, self.n_components))
        X_sampled = self.inverse_transform(z)
        return X_sampled



class KMeans:
    """
    KMeans clustering algorithm using JAX for efficient computation.
    """
    def __init__(self, k, epochs=1):
        """
        Initialize the KMeans object.

        Args:
            k (int): Number of clusters.
            epochs (int): Number of training epochs. Default is 1.
        """
        self.k = k
        self.epochs = epochs

    def fit(self, data):
        """
        Fit the KMeans model to the given data.

        Args:
            data (list or array): Input data for clustering.

        Returns:
            tuple: A tuple containing the data and assigned cluster labels.
        """
        data = jnp.array(data)
        centroids = jnp.array([random.choice(data) for _ in range(self.k)])
        labels = jnp.zeros(data.shape[0], dtype=jnp.int32)

        for _ in range(self.epochs):
            labels = self.train_step(data, centroids)
            centroids = self.set_centroids(data, labels)

        return data, labels

    @staticmethod
    @jax.jit
    def train_step(data, centroids):
        """
        Perform a single training step of the KMeans algorithm.

        Args:
            data (array): Input data.
            centroids (array): Current cluster centroids.

        Returns:
            array: Cluster labels for each data point.
        """
        distances = jnp.linalg.norm(data[:, None, :] - centroids, axis=2)
        labels = jnp.argmin(distances, axis=1)
        return labels

    @staticmethod
    def set_centroids(data, labels):
        """
        Update cluster centroids based on data and labels.

        Args:
            data (array): Input data.
            labels (array): Cluster labels for each data point.

        Returns:
            array: Updated cluster centroids.
        """
        centroids = jnp.zeros((labels.max() + 1, data.shape[1]))
        counts = jnp.bincount(labels)

        for label in range(centroids.shape[0]):
            if counts[label] == 0:
                continue
            cluster_data = data[labels == label]
            mean = jnp.mean(cluster_data, axis=0)
            centroids = centroids.at[label].set(mean)

        return centroids

class GaussianMixtureModel:
    def __init__(self, n_components, n_features, max_iter=100, tol=1e-4):
        """
        Initialize a Gaussian Mixture Model.

        Args:
            n_components (int): Number of Gaussian components.
            n_features (int): Number of features in the data.
            max_iter (int): Maximum number of iterations for EM algorithm.
            tol (float): Convergence tolerance.

        Attributes:
            n_components (int): Number of Gaussian components.
            n_features (int): Number of features in the data.
            max_iter (int): Maximum number of iterations for EM algorithm.
            tol (float): Convergence tolerance.
            weights (ndarray): Component weights.
            means (ndarray): Component means.
            covariances (ndarray): Component covariances.
        """
        self.n_components = n_components
        self.n_features = n_features
        self.max_iter = max_iter
        self.tol = tol

        self.weights = np.ones(n_components) / n_components
        self.means = np.random.randn(n_components, n_features)
        self.covariances = np.array([np.eye(n_features) for _ in range(n_components)])

    def fit(self, X):
        """
        Fit the Gaussian Mixture Model to the data using Expectation-Maximization.

        Args:
            X (ndarray): Input data with shape (n_samples, n_features).
        """
        for _ in range(self.max_iter):
            prev_log_likelihood = self.log_likelihood(X)

            # Expectation step
            responsibilities = self.expectation(X)

            # Maximization step
            self.weights = responsibilities.sum(axis=0) / X.shape[0]
            self.means = np.dot(responsibilities.T, X) / responsibilities.sum(axis=0)[:, np.newaxis]
            for k in range(self.n_components):
                diff = X - self.means[k]
                self.covariances[k] = np.dot(responsibilities[:, k] * diff.T, diff) / responsibilities[:, k].sum()

            current_log_likelihood = self.log_likelihood(X)

            if np.abs(current_log_likelihood - prev_log_likelihood) < self.tol:
                break

    def expectation(self, X):
        """
        Perform the Expectation step of the EM algorithm.

        Args:
            X (ndarray): Input data with shape (n_samples, n_features).

        Returns:
            ndarray: Responsibilities of data points for each component.
        """
        responsibilities = np.zeros((X.shape[0], self.n_components))
        for k in range(self.n_components):
            responsibilities[:, k] = self.weights[k] * self.pdf(X, self.means[k], self.covariances[k])
        responsibilities /= responsibilities.sum(axis=1, keepdims=True)
        return responsibilities

    def pdf(self, X, mean, cov):
        """
        Compute the probability density function of data points for a Gaussian component.

        Args:
            X (ndarray): Input data with shape (n_samples, n_features).
            mean (ndarray): Mean of the Gaussian component.
            cov (ndarray): Covariance of the Gaussian component.

        Returns:
            ndarray: PDF values for each data point.
        """
        d = X.shape[1]
        norm_factor = np.sqrt((2 * np.pi)**d * np.linalg.det(cov))
        exponent = -0.5 * np.sum(np.dot((X - mean), np.linalg.inv(cov)) * (X - mean), axis=1)
        return np.exp(exponent) / norm_factor

    def log_likelihood(self, X):
        """
        Compute the log-likelihood of the data given the model parameters.

        Args:
            X (ndarray): Input data with shape (n_samples, n_features).

        Returns:
            float: Log-likelihood of the data.
        """
        likelihoods = np.zeros((X.shape[0], self.n_components))
        for k in range(self.n_components):
            likelihoods[:, k] = self.weights[k] * self.pdf(X, self.means[k], self.covariances[k])
        return np.sum(np.log(np.sum(likelihoods, axis=1)))
    

class BayesianInference:
    """
    A class for performing Bayesian inference to estimate the mean and variance of a normal distribution.

    Attributes:
        prior_mean (float): Prior mean for the unknown mean of the distribution.
        prior_var (float): Prior variance for the unknown mean of the distribution.
        likelihood_variance (float): Known likelihood variance of the observed data.

    Methods:
        likelihood(data, mean, var):
            Calculate the likelihood of the data given mean and variance.

        posterior(data):
            Calculate the posterior distribution of the unknown mean after observing data.
    """

    def __init__(self, prior_mean, prior_var, likelihood_variance):
        """
        Initialize the BayesianInference class.

        Args:
            prior_mean (float): Prior mean for the unknown mean of the distribution.
            prior_var (float): Prior variance for the unknown mean of the distribution.
            likelihood_variance (float): Known likelihood variance of the observed data.
        """
        self.prior_mean = prior_mean
        self.prior_var = prior_var
        self.likelihood_variance = likelihood_variance

    def likelihood(self, data, mean, var):
        """
        Calculate the likelihood of the data given mean and variance.

        Args:
            data (ndarray): Observed data.
            mean (float): Mean parameter.
            var (float): Variance parameter.

        Returns:
            float: Likelihood value.
        """
        return jnp.exp(-0.5 * jnp.sum((data - mean)**2) / var) / jnp.sqrt(2 * jnp.pi * var)**len(data)

    def posterior(self, data):
        """
        Calculate the posterior distribution of the unknown mean after observing data.

        Args:
            data (ndarray): Observed data.

        Returns:
            float: Posterior mean estimate.
            float: Posterior variance estimate.
        """
        likelihood_mean = jnp.mean(data)
        posterior_var = 1.0 / (1.0 / self.prior_var + len(data) / self.likelihood_variance)
        posterior_mean = posterior_var * (self.prior_mean / self.prior_var + len(data) * likelihood_mean / self.likelihood_variance)
        return posterior_mean, posterior_var


class NaiveBayesClassifier:
    """
    A simple implementation of the Naive Bayes classifier for classification tasks.

    Attributes:
        class_probabilities (dict): Class probabilities for each unique class label.
        feature_probabilities (dict): Feature probabilities for each class and feature.

    Methods:
        fit(X, y):
            Fits the classifier to the training data by calculating class and feature probabilities.

        predict(X):
            Predicts class labels for input samples using the trained classifier.
    """

    def __init__(self):
        """Initialize the NaiveBayesClassifier."""
        self.class_probabilities = {}
        self.feature_probabilities = {}

    def fit(self, X, y):
        """
        Fit the classifier to the training data.

        Args:
            X (list): List of training samples, where each sample is a list of feature values.
            y (list): List of corresponding class labels.

        Returns:
            None
        """
        unique_classes = list(set(y))
        total_samples = len(y)

        for cls in unique_classes:
            class_samples = [X[i] for i in range(total_samples) if y[i] == cls]
            self.class_probabilities[cls] = len(class_samples) / total_samples

            self.feature_probabilities[cls] = {}
            for feature_idx in range(len(X[0])):
                feature_values = [sample[feature_idx] for sample in class_samples]
                unique_feature_values = list(set(feature_values))
                self.feature_probabilities[cls][feature_idx] = {}
                for value in unique_feature_values:
                    count = sum(1 for v in feature_values if v == value)
                    self.feature_probabilities[cls][feature_idx][value] = count / len(class_samples)

    def predict(self, X):
        """
        Predict class labels for input samples.

        Args:
            X (list): List of input samples, where each sample is a list of feature values.

        Returns:
            list: List of predicted class labels for input samples.
        """
        predictions = []
        for sample in X:
            max_probability = -jnp.inf
            predicted_class = None
            for cls, class_prob in self.class_probabilities.items():
                likelihood = class_prob
                for feature_idx, value in enumerate(sample):
                    if value in self.feature_probabilities[cls][feature_idx]:
                        likelihood *= self.feature_probabilities[cls][feature_idx][value]
                    else:
                        # Handle unseen feature values using a small value
                        likelihood *= 1e-10
                if likelihood > max_probability:
                    max_probability = likelihood
                    predicted_class = cls
            predictions.append(predicted_class)
        return predictions


class GaussianProcess:
    """
    A basic implementation of Gaussian Process regression using JAX.

    Attributes:
        kernel (callable): The kernel function to measure similarity between data points.
        noise (float): Measurement noise added to the diagonal of the kernel matrix.
        X (ndarray): Training inputs.
        y (ndarray): Training outputs.
        K (ndarray): Kernel matrix incorporating training inputs and noise.

    Methods:
        fit(X, y):
            Fits the Gaussian Process model to the training data.

        predict(X_new):
            Makes predictions for new input points using the trained model.
    """

    def __init__(self, kernel, noise=1e-3):
        """
        Initialize the GaussianProcess.

        Args:
            kernel (callable): The kernel function.
            noise (float, optional): Measurement noise added to the diagonal of the kernel matrix.
        """
        self.kernel = kernel
        self.noise = noise
        self.X = None
        self.y = None
        self.K = None

    def fit(self, X, y):
        """
        Fit the Gaussian Process model to the training data.

        Args:
            X (ndarray): Training input data.
            y (ndarray): Training output data.

        Returns:
            None
        """
        self.X = X
        self.y = y
        self.K = self.kernel(self.X, self.X) + jnp.eye(len(X)) * self.noise

    def predict(self, X_new):
        """
        Make predictions for new input points.

        Args:
            X_new (ndarray): New input points.

        Returns:
            ndarray: Predicted mean for the new input points.
            ndarray: Predicted covariance matrix for the new input points.
        """
        K_inv = jnp.linalg.inv(self.K)
        K_s = self.kernel(self.X, X_new)
        K_ss = self.kernel(X_new, X_new)

        mu_s = jnp.dot(K_s.T, jnp.dot(K_inv, self.y))
        cov_s = K_ss - jnp.dot(K_s.T, jnp.dot(K_inv, K_s))
        return mu_s, cov_s
    

def rbf_kernel(x1, x2, length_scale=1.0):
    """For testing Gaussian Processes"""
    diff = x1[:, None] - x2
    return jnp.exp(-0.5 * jnp.sum(diff**2, axis=-1) / length_scale**2)


class VariationalInference:
    """
    A class for performing Variational Inference using a given model and variational family.

    Attributes:
        model (object): The probabilistic model for which to perform inference.
        variational_family (object): The variational family used for approximation.
        params (ndarray): Parameters of the variational distribution.

    Methods:
        fit(data, num_epochs=100, learning_rate=0.01):
            Fit the variational distribution using stochastic gradient descent on ELBO.

        sample(num_samples=10):
            Sample from the fitted variational distribution.

    """
    def __init__(self, model, variational_family):
        """
        Initialize the VariationalInference class.

        Args:
            model (object): The probabilistic model for which to perform inference.
            variational_family (object): The variational family used for approximation.
        """
        self.model = model
        self.variational_family = variational_family
        self.params = None

    def fit(self, data, num_epochs=100, learning_rate=0.01):
        """
        Fit the variational distribution using stochastic gradient descent on ELBO.

        Args:
            data (ndarray): Observed data used for fitting.
            num_epochs (int, optional): Number of training epochs.
            learning_rate (float, optional): Learning rate for optimization.

        Returns:
            None
        """
        self.params = jax.random.normal(jax.random.PRNGKey(0), self.variational_family.param_shape)

        def elbo(params, data):
            q_params = self.variational_family.unpack(params)
            q_samples = self.variational_family.sample(q_params, num_samples=10)
            q_log_probs = self.variational_family.log_prob(q_params, q_samples)
            p_log_probs = self.model.log_prob(data, q_samples)
            return jnp.mean(p_log_probs - q_log_probs)

        grad_elbo = jax.grad(elbo)

        for _ in range(num_epochs):
            gradient = grad_elbo(self.params, data)
            self.params -= learning_rate * gradient

    def sample(self, num_samples=10):
        """
        Sample from the fitted variational distribution.

        Args:
            num_samples (int, optional): Number of samples to generate.

        Returns:
            ndarray: Samples from the variational distribution.
        """
        q_params = self.variational_family.unpack(self.params)
        return self.variational_family.sample(q_params, num_samples)


def convolution(matrix, kernel, stride=1, padding='same'):
    """
    Perform convolution operation on an matrix using a given kernel.

    Args:
        matrix (jax.numpy.ndarray): Input matrix as a 2D array.
        kernel (jax.numpy.ndarray): Convolution kernel as a 2D array.
        stride (int, optional): Stride value for convolution. Defaults to 1.
        padding (str, optional): Padding mode ('same' or 'valid'). Defaults to 'same'.

    Returns:
        jax.numpy.ndarray: Output matrix after convolution.
    """
    matrix_height, matrix_width = matrix.shape
    kernel_height, kernel_width = kernel.shape

    if padding == 'same':
        padding_height = (kernel_height - 1) // 2
        padding_width = (kernel_width - 1) // 2
        height = (matrix_height - kernel_height + 2 * padding_height) // stride + 1
        width = (matrix_width - kernel_width + 2 * padding_width) // stride + 1
        matrix = jnp.pad(matrix, ((padding_height, padding_height), (padding_width, padding_width)))
    else:
        height = matrix_height - kernel_height + 1
        width = matrix_width - kernel_width + 1

    output = jnp.zeros((height, width))

    for row in range(output.shape[0]):
        for col in range(output.shape[1]):
            patch = matrix[row:row+kernel_height, col:col+kernel_width]
            output = output.at[row, col].set(jnp.sum(patch * kernel))

    return output



class LinearRegression:
    """
    Linear Regression model implemented using JAX.

    Parameters:
    - input_dim (int): Dimension of the input feature.
    - output_dim (int): Dimension of the output target.

    Methods:
    - linear_regression(params, x): Linear regression prediction function.
    - loss(params, x, y): Mean squared error loss function.
    - train(x_data, y_data, learning_rate=0.1, num_epochs=100): Training function.
    - get_params(): Get the learned weights and bias.

    Example usage:
    ```
    num_samples = 100
    input_dim = 1
    output_dim = 1

    x_data = jax.random.normal(random.PRNGKey(0), (num_samples, input_dim))
    y_data = jnp.dot(x_data, jnp.array([[2.0]])) - jnp.array([[-1.0]])

    lr_model = LinearRegression(input_dim, output_dim)
    lr_model.train(x_data, y_data)

    learned_weights, learned_bias = lr_model.get_params()
    print("Learned Weights:", learned_weights)
    print("Learned Bias:", learned_bias)
    ```
    """
    def __init__(self, input_dim, output_dim):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.key = random.PRNGKey(0)
        self.params = (jnp.zeros((input_dim, output_dim)), jnp.zeros((output_dim,)))

    def linear_regression(self, params, x):
        """
        Compute the linear regression prediction.

        Args:
        - params (tuple): Model parameters (weights, bias).
        - x (jax.numpy.ndarray): Input data.

        Returns:
        - jax.numpy.ndarray: Predicted output.
        """
        weights, bias = params
        return jnp.dot(x, weights) + bias

    def loss(self, params, x, y):
        """
        Calculate the mean squared error loss.

        Args:
        - params (tuple): Model parameters (weights, bias).
        - x (jax.numpy.ndarray): Input data.
        - y (jax.numpy.ndarray): Target data.

        Returns:
        - float: Mean squared error loss.
        """
        predictions = self.linear_regression(params, x)
        return jnp.mean((predictions - y) ** 2)

    def train(self, x_data, y_data, learning_rate=0.1, num_epochs=100):
        """
        Train the linear regression model.

        Args:
        - x_data (jax.numpy.ndarray): Input data.
        - y_data (jax.numpy.ndarray): Target data.
        - learning_rate (float): Learning rate for gradient descent.
        - num_epochs (int): Number of training epochs.

        Returns:
        - None
        """
        grad_loss = jax.grad(self.loss)
        for epoch in range(num_epochs):
            grads = grad_loss(self.params, x_data, y_data)
            weights_grad, bias_grad = grads
            weights, bias = self.params
            weights -= learning_rate * weights_grad
            bias -= learning_rate * bias_grad
            self.params = (weights, bias)
            epoch_loss = self.loss(self.params, x_data, y_data)
            print(f"Epoch {epoch+1}, Loss: {epoch_loss:.4f}")

        print("Training completed.")

    def get_params(self):
        """
        Get the learned weights and bias.

        Returns:
        - tuple: Learned weights and bias.
        """
        return self.params
    


class LogisticRegression:
    """
    Logistic Regression model implemented using JAX.

    Parameters:
    - input_dim (int): Dimension of the input feature.

    Methods:
    - sigmoid(x): Sigmoid activation function.
    - logistic_regression(params, x): Logistic regression prediction function.
    - loss(params, x, y): Binary cross-entropy loss function.
    - train(x_data, y_data, learning_rate=0.1, num_epochs=100): Training function.
    - predict(x_data): Predict probabilities using the trained model.

    Example usage:
    ```
    num_samples = 100
    input_dim = 2

    x_data = jax.random.normal(random.PRNGKey(0), (num_samples, input_dim))
    logits = jnp.dot(x_data, jnp.array([0.5, -0.5])) - 0.1
    y_data = (logits > 0).astype(jnp.float32)

    lr_model = LogisticRegression(input_dim)
    lr_model.train(x_data, y_data)

    test_data = jax.random.normal(random.PRNGKey(0), (num_samples, input_dim))
    predictions = lr_model.predict(test_data)
    print("Predictions:", predictions)
    ```
    """
    def __init__(self, input_dim):
        self.input_dim = input_dim
        self.key = random.PRNGKey(0)
        self.params = (jnp.zeros((input_dim,)), jnp.zeros(()))

    def sigmoid(self, x):
        """
        Sigmoid activation function.

        Args:
        - x (jax.numpy.ndarray): Input values.

        Returns:
        - jax.numpy.ndarray: Sigmoid activations.
        """
        return 1.0 / (1.0 + jnp.exp(-x))

    def logistic_regression(self, params, x):
        """
        Compute the logistic regression prediction.

        Args:
        - params (tuple): Model parameters (weights, bias).
        - x (jax.numpy.ndarray): Input data.

        Returns:
        - jax.numpy.ndarray: Predicted probabilities.
        """
        weights, bias = params
        return self.sigmoid(jnp.dot(x, weights) + bias)

    def loss(self, params, x, y):
        """
        Calculate the binary cross-entropy loss.

        Args:
        - params (tuple): Model parameters (weights, bias).
        - x (jax.numpy.ndarray): Input data.
        - y (jax.numpy.ndarray): Binary target data (0 or 1).

        Returns:
        - float: Binary cross-entropy loss.
        """
        predictions = self.logistic_regression(params, x)
        return -jnp.mean(y * jnp.log(predictions) + (1 - y) * jnp.log(1 - predictions))

    def train(self, x_data, y_data, learning_rate=0.1, num_epochs=100):
        """
        Train the logistic regression model.

        Args:
        - x_data (jax.numpy.ndarray): Input data.
        - y_data (jax.numpy.ndarray): Binary target data (0 or 1).
        - learning_rate (float): Learning rate for gradient descent.
        - num_epochs (int): Number of training epochs.

        Returns:
        - None
        """
        grad_loss = jax.grad(self.loss)
        for epoch in range(num_epochs):
            grads = grad_loss(self.params, x_data, y_data)
            weights_grad, bias_grad = grads
            weights, bias = self.params
            weights -= learning_rate * weights_grad
            bias -= learning_rate * bias_grad
            self.params = (weights, bias)
            epoch_loss = self.loss(self.params, x_data, y_data)
            print(f"Epoch {epoch+1}, Loss: {epoch_loss:.4f}")

        print("Training completed.")

    def predict(self, x_data):
        """
        Predict probabilities using the trained model.

        Args:
        - x_data (jax.numpy.ndarray): Input data.

        Returns:
        - jax.numpy.ndarray: Predicted probabilities.
        """
        return self.logistic_regression(self.params, x_data)