from typing import Callable, Tuple
import jax.numpy as jnp

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

    def __init__(self, kernel: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray], noise: float = 1e-3):
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

    def fit(self, X: jnp.ndarray, y: jnp.ndarray) -> None:
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

    def predict(self, X_new: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
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
