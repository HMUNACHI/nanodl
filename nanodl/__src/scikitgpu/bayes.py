import jax.numpy as jnp

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