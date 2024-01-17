import jax
import jax.numpy as jnp
from typing import Any, Callable

class VariationalInference:
    """
    A class for performing Variational Inference using a given model and variational family.

    Attributes:
        model (object): The probabilistic model for which to perform inference.
        variational_family (object): The variational family used for approximation.
        params (jnp.ndarray): Parameters of the variational distribution.

    Methods:
        fit(data, num_epochs=100, learning_rate=0.01):
            Fit the variational distribution using stochastic gradient descent on ELBO.

        sample(num_samples=10):
            Sample from the fitted variational distribution.

    Example Usage:
        # Define a probabilistic model (e.g., a Bayesian neural network)
        model = BayesianNeuralNetwork(...)

        # Define a variational family (e.g., mean-field Gaussian)
        variational_family = MeanFieldGaussian(...)

        # Create an instance of the VariationalInference class
        vi = VariationalInference(model, variational_family)

        # Fit the variational distribution to the data
        vi.fit(data, num_epochs=100, learning_rate=0.01)

        # Sample from the fitted variational distribution
        samples = vi.sample(num_samples=10)
    """

    def __init__(self, model: Any, variational_family: Any):
        """
        Initialize the VariationalInference class.

        Args:
            model (object): The probabilistic model for which to perform inference.
            variational_family (object): The variational family used for approximation.
        """
        self.model = model
        self.variational_family = variational_family
        self.params = None

    def fit(self, data: jnp.ndarray, num_epochs: int = 100, learning_rate: float = 0.01) -> None:
        """
        Fit the variational distribution using stochastic gradient descent on ELBO.

        Args:
            data (jnp.ndarray): Observed data used for fitting.
            num_epochs (int, optional): Number of training epochs.
            learning_rate (float, optional): Learning rate for optimization.
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

    def sample(self, num_samples: int = 10) -> jnp.ndarray:
        """
        Sample from the fitted variational distribution.

        Args:
            num_samples (int, optional): Number of samples to generate.

        Returns:
            jnp.ndarray: Samples from the variational distribution.
        """
        q_params = self.variational_family.unpack(self.params)
        return self.variational_family.sample(q_params, num_samples)
