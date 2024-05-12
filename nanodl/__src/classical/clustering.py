import jax
import jax.numpy as jnp


class KMeans:
    """
    KMeans clustering using JAX for GPU/TPU acceleration.

    Attributes:
        k (int): Number of clusters.
        num_iters (int): Maximum number of iterations.
        random_seed (int): Random seed for initialization.
        centroids (Optional[jnp.ndarray]): Centroids of the clusters.
        clusters (Optional[jnp.ndarray]): Cluster assignments of the data points.

    Example usage:
    ```
        kmeans = KMeans(k=4)
        X, _ = make_blobs(n_samples=300, centers=4, n_features=2, random_state=0)
        kmeans.fit(X)
        clusters = kmeans.predict(X)
        print("Centroids:", kmeans.centroids)
        print("Cluster assignments:", clusters)
    ```
    """

    def __init__(self, k: int, num_iters: int = 100, random_seed: int = 0) -> None:
        self.k = k
        self.num_iters = num_iters
        self.random_seed = random_seed
        self.centroids = None
        self.clusters = None

    def initialize_centroids(self, X: jnp.ndarray) -> jnp.ndarray:

        indices = jnp.arange(X.shape[0])
        selected = jax.random.choice(
            jax.random.PRNGKey(self.random_seed),
            indices,
            shape=(self.k,),
            replace=False,
        )
        return X[selected]

    def assign_clusters(self, X: jnp.ndarray, centroids: jnp.ndarray) -> jnp.ndarray:

        distances = jnp.sqrt(
            ((X[:, jnp.newaxis, :] - centroids[jnp.newaxis, :, :]) ** 2).sum(axis=2)
        )
        return jnp.argmin(distances, axis=1)

    def update_centroids(self, X: jnp.ndarray, clusters: jnp.ndarray) -> jnp.ndarray:

        return jnp.array([X[clusters == i].mean(axis=0) for i in range(self.k)])

    def fit(self, X: jnp.ndarray) -> None:

        self.centroids = self.initialize_centroids(X)
        for _ in range(self.num_iters):
            self.clusters = self.assign_clusters(X, self.centroids)
            new_centroids = self.update_centroids(X, self.clusters)
            if jnp.allclose(self.centroids, new_centroids):
                break
            self.centroids = new_centroids

    def predict(self, X: jnp.ndarray) -> jnp.ndarray:
        if self.centroids is None:
            raise ValueError("Model not yet trained. Call 'fit' with training data.")
        return self.assign_clusters(X, self.centroids)


class GaussianMixtureModel:
    """
    Gaussian Mixture Model implemented in JAX.

    This class represents a Gaussian Mixture Model (GMM) for clustering and density estimation.
    It uses the Expectation-Maximization (EM) algorithm for fitting the model to data.

    Attributes:
        n_components (int): Number of mixture components.
        tol (float): Tolerance for convergence.
        max_iter (int): Maximum number of iterations for the EM algorithm.
        means (jnp.ndarray): Means of the Gaussian components.
        covariances (jnp.ndarray): Covariances of the Gaussian components.
        weights (jnp.ndarray): Weights of the Gaussian components.
        seed (int): Random seed for initialization.

    Example:
    ```
        >>> import jax.numpy as jnp
        >>> from gaussian_mixture_model_jax import GaussianMixtureModelJAX
        >>> X = jnp.array([[1, 2], [1, 4], [1, 0],
        ...                [10, 2], [10, 4], [10, 0]])
        >>> gmm = GaussianMixtureModelJAX(n_components=2, seed=42)
        >>> gmm.fit(X)
        >>> print(gmm.means)
        >>> labels = gmm.predict(X)
        >>> print(labels)
    ```
    """

    def __init__(
        self, n_components: int, tol: float = 1e-3, max_iter: int = 100, seed: int = 0
    ) -> None:
        self.n_components = n_components
        self.tol = tol
        self.max_iter = max_iter
        self.means = None
        self.covariances = None
        self.weights = None
        self.seed = seed

    def fit(self, X: jnp.ndarray) -> None:
        _, n_features = X.shape
        rng = jax.random.PRNGKey(self.seed)

        self.means = jax.random.normal(rng, (self.n_components, n_features))
        self.covariances = jnp.array([jnp.eye(n_features)] * self.n_components)
        self.weights = jnp.ones(self.n_components) / self.n_components

        log_likelihood = 0
        for _ in range(self.max_iter):
            responsibilities = self._e_step(X)
            self._m_step(X, responsibilities)

            new_log_likelihood = self._compute_log_likelihood(X)
            if jnp.abs(new_log_likelihood - log_likelihood) < self.tol:
                break
            log_likelihood = new_log_likelihood

    def _e_step(self, X: jnp.ndarray) -> jnp.ndarray:
        responsibilities = jnp.zeros((X.shape[0], self.n_components))
        for k in range(self.n_components):
            responsibilities = responsibilities.at[:, k].set(
                self.weights[k]
                * self._multivariate_gaussian(X, self.means[k], self.covariances[k])
            )
        responsibilities /= responsibilities.sum(axis=1, keepdims=True)
        return responsibilities

    def _m_step(self, X: jnp.ndarray, responsibilities: jnp.ndarray) -> None:
        n_samples = X.shape[0]
        for k in range(self.n_components):
            Nk = responsibilities[:, k].sum()
            self.means = self.means.at[k].set(
                (1 / Nk) * jnp.dot(responsibilities[:, k], X)
            )
            diff = X - self.means[k]
            self.covariances = self.covariances.at[k].set(
                (1 / Nk) * jnp.dot(responsibilities[:, k] * diff.T, diff)
            )
            self.weights = self.weights.at[k].set(Nk / n_samples)

    def _multivariate_gaussian(
        self, X: jnp.ndarray, mean: jnp.ndarray, cov: jnp.ndarray
    ) -> jnp.ndarray:
        n = X.shape[1]
        diff = X - mean
        return jnp.exp(
            -0.5 * jnp.sum(jnp.dot(diff, jnp.linalg.inv(cov)) * diff, axis=1)
        ) / (jnp.sqrt((2 * jnp.pi) ** n * jnp.linalg.det(cov)))

    def _compute_log_likelihood(self, X: jnp.ndarray) -> float:
        log_likelihood = 0
        for k in range(self.n_components):
            log_likelihood += jnp.sum(
                jnp.log(
                    self.weights[k]
                    * self._multivariate_gaussian(X, self.means[k], self.covariances[k])
                )
            )
        return log_likelihood

    def predict(self, X: jnp.ndarray) -> jnp.ndarray:
        responsibilities = self._e_step(X)
        return jnp.argmax(responsibilities, axis=1)
