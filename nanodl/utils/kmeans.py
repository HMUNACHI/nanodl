import jax
import random
import jax.numpy as jnp
from typing import Tuple, Union

class KMeans:
    """
    KMeans clustering algorithm using JAX for efficient computation.

    This class implements the KMeans clustering algorithm, which groups data into 'k' number of clusters.

    Attributes:
        k (int): Number of clusters.
        epochs (int): Number of training epochs. Default is 1.

    Methods:
        fit(data):
            Fit the KMeans model to the given data and return the data with assigned cluster labels.

    Example Usage:
        # Create an instance of the KMeans class
        kmeans = KMeans(k=3, epochs=10)

        # Fit the model on the dataset
        data, labels = kmeans.fit(dataset)

        # `data` is the original dataset, `labels` contains the cluster assignments for each data point
    """
    def __init__(self, k: int, epochs: int = 1):
        """
        Initialize the KMeans object.

        Args:
            k (int): Number of clusters.
            epochs (int): Number of training epochs. Default is 1.
        """
        self.k = k
        self.epochs = epochs

    def fit(self, data: Union[list, jnp.ndarray]) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Fit the KMeans model to the given data.

        Args:
            data (list or jnp.ndarray): Input data for clustering.

        Returns:
            Tuple[jnp.ndarray, jnp.ndarray]: A tuple containing the data and assigned cluster labels.
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
    def train_step(data: jnp.ndarray, centroids: jnp.ndarray) -> jnp.ndarray:
        """
        Perform a single training step of the KMeans algorithm.

        Args:
            data (jnp.ndarray): Input data.
            centroids (jnp.ndarray): Current cluster centroids.

        Returns:
            jnp.ndarray: Cluster labels for each data point.
        """
        distances = jnp.linalg.norm(data[:, None, :] - centroids, axis=2)
        labels = jnp.argmin(distances, axis=1)
        return labels

    @staticmethod
    def set_centroids(data: jnp.ndarray, labels: jnp.ndarray) -> jnp.ndarray:
        """
        Update cluster centroids based on data and labels.

        Args:
            data (jnp.ndarray): Input data.
            labels (jnp.ndarray): Cluster labels for each data point.

        Returns:
            jnp.ndarray: Updated cluster centroids.
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