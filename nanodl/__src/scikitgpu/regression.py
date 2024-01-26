import jax
import jax.numpy as jnp
from typing import Callable, Tuple

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
        self.key = jax.random.PRNGKey(0)
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
        self.key = jax.random.PRNGKey(0)
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
    

class GaussianProcess:
    """
    A basic implementation of Gaussian Process regression using JAX.

    Attributes:
        kernel (Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]): The kernel function to measure similarity between data points.
        noise (float): Measurement noise added to the diagonal of the kernel matrix.
        X (jnp.ndarray): Training inputs.
        y (jnp.ndarray): Training outputs.
        K (jnp.ndarray): Kernel matrix incorporating training inputs and noise.

    Methods:
        fit(X, y):
            Fits the Gaussian Process model to the training data.

        predict(X_new):
            Makes predictions for new input points using the trained model.

    Example Usage:
        # Define a kernel function, e.g., Radial Basis Function (RBF) kernel
        def rbf_kernel(x1, x2, length_scale=1.0):
            diff = x1[:, None] - x2
            return jnp.exp(-0.5 * jnp.sum(diff**2, axis=-1) / length_scale**2)

        # Create an instance of the GaussianProcess class
        gp = GaussianProcess(kernel=rbf_kernel, noise=1e-3)

        # Fit the model on the training data
        gp.fit(X_train, y_train)

        # Make predictions on new data
        mean, covariance = gp.predict(X_new)
    """

    def __init__(self, 
                 kernel: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray], 
                 noise: float = 1e-3):
        """
        Initialize the GaussianProcess.

        Args:
            kernel (Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]): The kernel function.
            noise (float, optional): Measurement noise added to the diagonal of the kernel matrix.
        """
        self.kernel = kernel
        self.noise = noise
        self.X = None
        self.y = None
        self.K = None

    def fit(self, 
            X: jnp.ndarray, 
            y: jnp.ndarray) -> None:
        """
        Fit the Gaussian Process model to the training data.

        Args:
            X (jnp.ndarray): Training input data.
            y (jnp.ndarray): Training output data.

        Returns:
            None
        """
        self.X = X
        self.y = y
        self.K = self.kernel(self.X, self.X) + jnp.eye(len(X)) * self.noise

    def predict(self, 
                X_new: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Make predictions for new input points.

        Args:
            X_new (jnp.ndarray): New input points.

        Returns:
            Tuple[jnp.ndarray, jnp.ndarray]: Predicted mean and covariance matrix for the new input points.
        """
        K_inv = jnp.linalg.inv(self.K)
        K_s = self.kernel(self.X, X_new)
        K_ss = self.kernel(X_new, X_new)

        mu_s = jnp.dot(K_s.T, jnp.dot(K_inv, self.y))
        cov_s = K_ss - jnp.dot(K_s.T, jnp.dot(K_inv, K_s))
        return mu_s, cov_s