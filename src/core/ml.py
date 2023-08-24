"""
"""
import jax
import jax.numpy as jnp
import numpy as np


class PCA:
    """
    """
    def __init__(self, dim):
        self.dim = dim

    def fit(self, data):
        data = data.T
        N_dim, N_samples = data.shape
        assert self.dim < N_samples and self.dim < N_dim

        self.μ = jnp.mean(data, axis=1, keepdims=True)
        self.σ = jnp.ones((N_dim, 1))
        data = (data - self.μ) / self.σ

        if N_dim < N_samples:
            C = jnp.einsum("ik,jk->ij", data, data) / (N_samples - 1)
            self.eigenvalues, self.U = jnp.linalg.eigh(C)
            self.eigenvalues = self.eigenvalues[::-1]
            self.U = self.U[:, ::-1]
            self.λ = jnp.sqrt(self.eigenvalues)
            self.U = self.U[:, : self.dim]
        else:
            D = (jnp.einsum("ki,kj->ij", data, data)/ N_dim)
            self.eigenvalues, V = jnp.linalg.eigh(D)
            self.eigenvalues = self.eigenvalues[::-1]
            V = V[:, ::-1]
            self.eigenvalues = self.eigenvalues[: self.dim] * (N_dim / (N_samples - 1))
            self.λ = jnp.sqrt(self.eigenvalues)
            S_inv = (1 / jnp.sqrt(self.eigenvalues * (N_samples - 1)))[jnp.newaxis, :]
            VS_inv = V[:, : self.dim] * S_inv
            self.U = jnp.einsum("ij,jk->ik",data, VS_inv)

        return self

    def transform(self, X):
        X = jnp.asarray(X).T
        return jnp.einsum("ji,jk->ik", self.U, (X - self.μ) / self.σ).T

    def inverse_transform(self, X):
        X = jnp.asarray(X).T
        return (jnp.einsum("ij,jk->ik", self.U, X) * self.σ + self.μ).T
    
    def sample(self, n=1):
        return jnp.array(np.random.normal(size=(self.dim, n)) * np.array(self.λ)[:, np.newaxis]).T



def kMeans(data, k=3, epochs=1):
    centroids = {key:random.choice(data) for key in range(k)}
    labels = [0 for _ in data]

    for i in range(epochs):
        labels = train_step(data, centroids, labels)
        centroids = set_centroids(k, data, labels)

    return data, labels


def train_step(data, centroids, labels):
    for idx in range(len(data)):
        min_distance = float("inf")
        for label, centroid in centroids.items():
            current_distance = distance(data[idx], centroid)
            if current_distance < min_distance:
                labels[idx] = label
                min_distance = current_distance
    return labels


def distance(a,b):
    return (sum([(one - two)**2 for one, two in zip(a, b)]))**0.5


def set_centroids(k, data, labels):
    centroids = {key:[0,0] for key in range(k)}
    counts = [0.005 for _ in range(k)]

    for point, label in zip(data, labels):
        centroids[label] = [centroids[label][0] + point[0], centroids[label][1] + point[1]]
        counts[label] += 1

    for key in centroids.keys():
        centroids[key] = [centroids[key][0]/counts[key], centroids[key][1]/counts[key]]

    return centroids



def convolution(image, kernel, stride=1, padding='same'):
    image_height, image_width = image.shape
    kernel_height, kernel_width = kernel.shape

    if padding == 'same':
        padding_height = int((kernel_height - 1)/2)
        padding_width = int((kernel_width - 1)/2)
        height = int((image_height - kernel_height + 2 * padding_height) / stride+1)
        width = int((image_width - kernel_width + 2 * padding_width) / stride+1)
        image = np.pad(image,((padding_height,padding_height), (padding_width,padding_width)))
    else:
        height = image_height - kernel_height + 1
        width = image_width - kernel_width + 1

    output = np.zeros((height, width))

    for row in range(output.shape[0]):
        for col in range(output.shape[1]):
            output[row, col] = (image[row:row+kernel_height, col:col+kernel_height] * kernel).sum()

    return output