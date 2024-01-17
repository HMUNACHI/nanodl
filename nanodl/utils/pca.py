import jax
import jax.numpy as jnp
from typing import Optional

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

    Example Usage:
        # Create an instance of the PCA class
        pca = PCA(n_components=2)

        # Fit the PCA model on the dataset
        pca.fit(data)

        # Transform the dataset into the PCA space
        transformed_data = pca.transform(data)

        # Optionally, inverse transform the data back to the original space
        original_data = pca.inverse_transform(transformed_data)
    """

    def __init__(self,
                 n_components: int):
        """
        Initialize the PCA class.

        Args:
            n_components (int): Number of principal components to retain.
        """
        self.n_components = n_components
        self.components = None
        self.mean = None

    def fit(self, 
            X: jnp.ndarray) -> None:
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

    def transform(self, 
                  X: jnp.ndarray) -> jnp.ndarray:
        """
        Transform the input data into the PCA space.

        Args:
            X (ndarray): Input data with shape (num_samples, num_features).

        Returns:
            ndarray: Transformed data with shape (num_samples, n_components).
        """
        X_centered = X - self.mean
        return jnp.dot(X_centered, self.components)

    def inverse_transform(self, 
                          X_transformed: jnp.ndarray) -> jnp.ndarray:
        """
        Inverse transform PCA-transformed data back to the original space.

        Args:
            X_transformed (ndarray): Transformed data with shape (num_samples, n_components).

        Returns:
            ndarray: Inverse transformed data with shape (num_samples, num_features).
        """
        return jnp.dot(X_transformed, self.components.T) + self.mean

    def sample(self, 
               n_samples:int=1, 
               key: Optional[jnp.ndarray] = None) -> jnp.ndarray:
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