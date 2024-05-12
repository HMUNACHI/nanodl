import unittest

import jax
import jax.numpy as jnp

from nanodl import *


class TestNaiveBayesFunctions(unittest.TestCase):
    def setUp(self):
        self.num_samples = 3
        self.num_features = 2
        self.num_classes = 2
        self.X = jnp.array([[0, 1], [1, 0], [1, 1]])
        self.y = jnp.array([0, 1, 0])

    def test_naive_bayes_classifier(self):
        classifier = NaiveBayesClassifier(num_classes=self.num_classes)
        classifier.fit(self.X, self.y)
        predictions = classifier.predict(self.X)
        self.assertEqual(predictions.shape, (self.num_samples,))
        self.assertTrue(
            jnp.all(predictions >= 0) and jnp.all(predictions < self.num_classes)
        )


class TestKClustering(unittest.TestCase):
    def setUp(self):
        self.k = 3
        self.num_samples = 300
        self.num_features = 2
        self.X = jax.random.normal(
            jax.random.PRNGKey(0), (self.num_samples, self.num_features)
        )

    def test_kmeans_fit_predict(self):
        kmeans = KMeans(k=self.k)
        kmeans.fit(self.X)
        clusters = kmeans.predict(self.X)
        self.assertEqual(len(set(clusters.tolist())), self.k)

    def test_gmm_fit_predict(self):
        gmm = GaussianMixtureModel(n_components=self.k)
        gmm.fit(self.X)
        labels = gmm.predict(self.X)
        self.assertEqual(len(set(labels.tolist())), self.k)


class TestPCA(unittest.TestCase):
    def test_pca_fit_transform(self):
        data = jax.random.normal(jax.random.PRNGKey(0), (1000, 10))
        pca = PCA(n_components=2)
        pca.fit(data)
        transformed_data = pca.transform(data)
        self.assertEqual(transformed_data.shape, (1000, 2))

    def test_pca_inverse_transform(self):
        data = jax.random.normal(jax.random.PRNGKey(0), (1000, 10))
        pca = PCA(n_components=2)
        pca.fit(data)
        transformed_data = pca.transform(data)
        inverse_data = pca.inverse_transform(transformed_data)
        self.assertEqual(inverse_data.shape, data.shape)

    def test_pca_sample(self):
        data = jax.random.normal(jax.random.PRNGKey(0), (1000, 10))
        pca = PCA(n_components=2)
        pca.fit(data)
        synthetic_samples = pca.sample(n_samples=100)
        self.assertEqual(synthetic_samples.shape, (100, 10))


class TestRegression(unittest.TestCase):
    def test_linear_regression(self):
        num_samples = 100
        input_dim = 1
        output_dim = 1
        x_data = jax.random.normal(jax.random.PRNGKey(0), (num_samples, input_dim))
        y_data = jnp.dot(x_data, jnp.array([[2.0]])) - jnp.array([[-1.0]])
        lr_model = LinearRegression(input_dim, output_dim)
        lr_model.fit(x_data, y_data)
        learned_weights, learned_bias = lr_model.get_params()
        self.assertTrue(jnp.allclose(learned_weights, jnp.array([[2.0]]), atol=1e-1))
        self.assertTrue(jnp.allclose(learned_bias, jnp.array([[1.0]]), atol=1e-1))

    def test_logistic_regression(self):
        num_samples = 100
        input_dim = 2
        x_data = jax.random.normal(jax.random.PRNGKey(0), (num_samples, input_dim))
        logits = jnp.dot(x_data, jnp.array([0.5, -0.5])) - 0.1
        y_data = (logits > 0).astype(jnp.float32)
        lr_model = LogisticRegression(input_dim)
        lr_model.fit(x_data, y_data)
        test_data = jax.random.normal(jax.random.PRNGKey(0), (num_samples, input_dim))
        predictions = lr_model.predict(test_data)
        self.assertTrue(jnp.all(predictions >= 0) and jnp.all(predictions <= 1))

    def test_gaussian_process(self):
        def rbf_kernel(x1, x2, length_scale=1.0):
            diff = x1[:, None] - x2
            return jnp.exp(-0.5 * jnp.sum(diff**2, axis=-1) / length_scale**2)

        num_samples = 100
        input_dim = 1
        X_train = jax.random.normal(jax.random.PRNGKey(0), (num_samples, input_dim))
        y_train = (
            jnp.sin(X_train)
            + jax.random.normal(jax.random.PRNGKey(0), (num_samples, 1)) * 0.1
        )
        gp = GaussianProcess(kernel=rbf_kernel, noise=1e-3)
        gp.fit(X_train, y_train)
        X_new = jax.random.normal(jax.random.PRNGKey(0), (num_samples, input_dim))
        mean, covariance = gp.predict(X_new)
        self.assertEqual(mean.shape, (num_samples, 1))
        self.assertEqual(covariance.shape, (num_samples, num_samples))


if __name__ == "__main__":
    unittest.main()
