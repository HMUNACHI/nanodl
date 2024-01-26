import jax
import jax.numpy as jnp
from typing import Tuple

def fit_naive_bayes(X: jnp.ndarray, y: jnp.ndarray, num_classes: int) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Fit a Naive Bayes classifier to the training data.

    Args:
        X (jnp.ndarray): Training data, shape (num_samples, num_features).
        y (jnp.ndarray): Class labels, shape (num_samples,).
        num_classes (int): Number of classes.

    Returns:
        Tuple[jnp.ndarray, jnp.ndarray]: Tuple containing class priors and feature probabilities.
        
    Example usage:
        X = jnp.array([[0, 1], [1, 0], [1, 1]])
        y = jnp.array([0, 1, 0])
        num_classes = 2
        class_priors, feature_probs = fit_naive_bayes(X, y, num_classes)
    """
    class_priors = jnp.zeros(num_classes)
    feature_probs = jnp.zeros((num_classes, X.shape[1]))

    for i in range(num_classes):
        class_mask = y == i
        class_count = jnp.sum(class_mask)
        class_priors = class_priors.at[i].set(class_count / X.shape[0])
        feature_count = jnp.sum(X[class_mask], axis=0)
        feature_probs = feature_probs.at[i].set(feature_count / class_count)

    return class_priors, feature_probs

@jax.jit
def predict_naive_bayes(X: jnp.ndarray, class_priors: jnp.ndarray, feature_probs: jnp.ndarray) -> jnp.ndarray:
    """
    Predict class labels for the given data using a Naive Bayes classifier.

    Args:
        X (jnp.ndarray): Data to predict, shape (num_samples, num_features).
        class_priors (jnp.ndarray): Class priors, shape (num_classes,).
        feature_probs (jnp.ndarray): Feature probabilities, shape (num_classes, num_features).

    Returns:
        jnp.ndarray: Predicted class labels.
    
    Example usage:
        X = jnp.array([[0, 1], [1, 0], [1, 1]])
        class_priors = jnp.array([0.5, 0.5])
        feature_probs = jnp.array([[0.5, 0.5], [0.5, 0.5]])
        predictions = predict_naive_bayes(X, class_priors, feature_probs)
    """
    # Calculate log probabilities for features
    log_feature_probs = jnp.log(feature_probs)
    log_feature_probs_neg = jnp.log(1 - feature_probs)

    # Expand dimensions for broadcasting
    expanded_log_feature_probs = log_feature_probs[:, None, :]
    expanded_log_feature_probs_neg = log_feature_probs_neg[:, None, :]

    # Calculate log probabilities for each sample and class
    log_probs = jnp.sum(expanded_log_feature_probs * X + expanded_log_feature_probs_neg * (1 - X), axis=2)
    log_probs += jnp.log(class_priors)[:, None]
    return jnp.argmax(log_probs, axis=0)

@jax.jit
def accuracy(y_true: jnp.ndarray, y_pred: jnp.ndarray) -> float:
    """
    Calculate the accuracy of predictions.

    Args:
        y_true (jnp.ndarray): True labels, shape (num_samples,).
        y_pred (jnp.ndarray): Predicted labels, shape (num_samples,).

    Returns:
        float: Accuracy of the predictions.

    Example usage:
        y_true = jnp.array([0, 1, 0])
        y_pred = jnp.array([0, 0, 1])
        acc = accuracy(y_true, y_pred)
    """
    return jnp.mean(y_true == y_pred)

class NaiveBayesClassifier:
    """
    Naive Bayes classifier using JAX.

    Example usage:
        classifier = NaiveBayesClassifier(num_classes=2)
        X = jnp.array([[0, 1], [1, 0], [1, 1]])
        y = jnp.array([0, 1, 0])
        classifier.fit(X, y)
        predictions = classifier.predict(X)
        acc = accuracy(y, predictions)
        print(f"Accuracy: {acc}")

    Attributes:
        num_classes (int): Number of classes.
        class_priors (jnp.ndarray): Class priors, shape (num_classes,).
        feature_probs (jnp.ndarray): Feature probabilities, shape (num_classes, num_features).
    """

    def __init__(self, num_classes: int):
        self.num_classes = num_classes
        self.class_priors = None
        self.feature_probs = None

    def fit(self, X: jnp.ndarray, y: jnp.ndarray) -> None:
        """
        Fit the Naive Bayes classifier to the training data.

        Args:
            X (jnp.ndarray): Training data, shape (num_samples, num_features).
            y (jnp.ndarray): Class labels, shape (num_samples,).
        """
        self.class_priors, self.feature_probs = fit_naive_bayes(X, y, self.num_classes)

    def predict(self, X: jnp.ndarray) -> jnp.ndarray:
        """
        Predict class labels for the given data.

        Args:
            X (jnp.ndarray): Data to predict, shape (num_samples, num_features).

        Returns:
            jnp.ndarray: Predicted class labels.
        """
        return predict_naive_bayes(X, self.class_priors, self.feature_probs)