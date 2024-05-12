from typing import Any, List

import jax
import jax.numpy as jnp


@jax.jit
def batch_cosine_similarities(
    source: jnp.ndarray, candidates: jnp.ndarray
) -> jnp.ndarray:
    """
    Calculate cosine similarities between a source vector and a batch of candidate vectors.

    Args:
        source (jnp.ndarray): Source vector of shape (D,).
        candidates (jnp.ndarray): Batch of candidate vectors of shape (N, D), where N is the number of candidates.

    Returns:
        jnp.ndarray: Array of cosine similarity scores of shape (N,).

    Example usage:
        ```
        >>> source = jnp.array([1, 0, 0])
        >>> candidates = jnp.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        >>> similarities = batch_cosine_similarities(source, candidates)
        >>> print(similarities)
        ```
    """
    dot_products = jnp.einsum("ij,j->i", candidates, source)
    norm_source = jnp.sqrt(jnp.einsum("i,i->", source, source))
    norm_candidates = jnp.sqrt(jnp.einsum("ij,ij->i", candidates, candidates))
    return dot_products / (norm_source * norm_candidates)


@jax.jit
def batch_pearsonr(x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    """
    Calculate batch Pearson correlation coefficient between two sets of vectors.

    Args:
        x (jnp.ndarray): First set of vectors of shape (N, D), where N is the number of vectors.
        y (jnp.ndarray): Second set of vectors of shape (N, D).

    Returns:
        jnp.ndarray: Array of Pearson correlation coefficients of shape (N,).

    Example usage:
        ```
        >>> x = jnp.array([[1, 2, 3], [4, 5, 6]])
        >>> y = jnp.array([[1, 5, 7], [2, 6, 8]])
        >>> correlations = batch_pearsonr(x, y)
        >>> print(correlations)
        ```
    """
    x = jnp.asarray(x).T
    y = jnp.asarray(y).T
    x = x - jnp.expand_dims(x.mean(axis=1), axis=-1)
    y = y - jnp.expand_dims(y.mean(axis=1), axis=-1)
    numerator = jnp.sum(x * y, axis=-1)
    sum_of_squares_x = jnp.einsum("ij,ij -> i", x, x)
    sum_of_squares_y = jnp.einsum("ij,ij -> i", y, y)
    denominator = jnp.sqrt(sum_of_squares_x * sum_of_squares_y)
    return numerator / denominator


@jax.jit
def classification_scores(labels: jnp.ndarray, preds: jnp.ndarray) -> jnp.ndarray:
    """
    Calculate classification evaluation scores using JAX.

    Args:
        labels (jnp.ndarray): Array of true labels.
        preds (jnp.ndarray): Array of predicted labels.

    Returns:
        jnp.ndarray: Array containing accuracy, precision, recall, and F1-score.

    Example usage:
        ```
        >>> labels = jnp.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0])
        >>> preds = jnp.array([1, 1, 1, 0, 1, 0, 1, 0, 0, 0])
        >>> print(classification_scores(labels, preds))
        ```
    """
    true_positives = jnp.sum(jnp.logical_and(preds == 1, labels == 1))
    true_negatives = jnp.sum(jnp.logical_and(preds == 0, labels == 0))
    false_positives = jnp.sum(jnp.logical_and(preds == 1, labels == 0))
    false_negatives = jnp.sum(jnp.logical_and(preds == 0, labels == 1))

    accuracy = (true_positives + true_negatives) / len(preds)
    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)
    f1 = 2 * (precision * recall) / (precision + recall)
    return jnp.array([accuracy, precision, recall, f1])


@jax.jit
def mean_reciprocal_rank(predictions: jnp.ndarray) -> float:
    """
    Calculate the Mean Reciprocal Rank (MRR) for a list of ranked predictions using JAX.

    Example usage:
        ```
        predictions = jnp.array([
            [0, 1, 2],  # "correct" prediction at index 0
            [1, 0, 2],  # "correct" prediction at index 1
            [2, 1, 0]   # "correct" prediction at index 2
        ])
        mrr_score = mean_reciprocal_rank(predictions)
        ```

    Args:
        predictions (jnp.ndarray): 2D array where each row contains ranked predictions
                                   and the "correct" prediction is indicated by a specific index.

    Returns:
        float: Mean Reciprocal Rank (MRR) score.
    """
    correct_indices = jnp.argmin(predictions, axis=1)
    ranks = correct_indices + 1
    reciprocal_ranks = 1.0 / ranks
    mean_mrr = jnp.mean(reciprocal_ranks)
    return mean_mrr


def jaccard(sequence1: List, sequence2: List) -> float:
    """
    Calculate Jaccard similarity between two sequences.

    Args:
        sequence1 (List): First input sequence.
        sequence2 (List): Second input sequence.

    Returns:
        float: Jaccard similarity score.

    Example usage:
        ```py
        >>> sequence1 = [1, 2, 3]
        >>> sequence2 = [2, 3, 4]
        >>> similarity = jaccard(sequence1, sequence2)
        >>> print(similarity)
        ```
    """
    numerator = len(set(sequence1).intersection(sequence2))
    denominator = len(set(sequence1).union(sequence2))
    return numerator / denominator


@jax.jit
def hamming(sequence1: jnp.ndarray, sequence2: jnp.ndarray) -> int:
    """
    Calculate Hamming similarity between two sequences using JAX.

    Args:
        sequence1 (jnp.ndarray): First input sequence.
        sequence2 (jnp.ndarray): Second input sequence.

    Returns:
        int: Hamming similarity score.

    Example usage:
        ```py
        >>> sequence1 = jnp.array([1, 2, 3, 4])
        >>> sequence2 = jnp.array([1, 2, 4, 4])
        >>> similarity = hamming_jax(sequence1, sequence2)
        >>> print(similarity)
        ```
    """
    return jnp.sum(sequence1 == sequence2)


def zero_pad_sequences(arr: jnp.array, max_length: int) -> jnp.array:
    """
    Zero-pad the given array to the specified maximum length along axis=1.

    This function pads the input array with zeros along the second dimension (axis=1)
    until it reaches the specified maximum length. If the array is already longer
    than the maximum length, it is returned as is.

    Args:
        arr (jax.numpy.ndarray): The array to be padded. Must be 2-dimensional.
        max_length (int): The maximum length to pad the array to along axis=1.

    Returns:
        jax.numpy.ndarray: The zero-padded array.

    Example usage:
        ```py
        >>> arr = jnp.array([[1, 2, 3], [4, 5, 6]])
        >>> max_length = 5
        >>> padded_arr = zero_pad_sequences(arr, max_length)
        >>> print(padded_arr)
        [[1 2 3 0 0]
         [4 5 6 0 0]]
        ```
    """
    current_length = arr.shape[1]
    num_zeros = max_length - current_length

    if num_zeros > 0:
        zeros = jnp.zeros((arr.shape[0], num_zeros), dtype=arr.dtype)
        padded_array = jnp.concatenate([arr, zeros], axis=1)
    else:
        padded_array = arr

    return padded_array


@jax.jit
def entropy(probabilities: jnp.ndarray) -> float:
    """
    Calculate the entropy of a probability distribution using JAX.

    Example usage:
        ```
        probabilities = jnp.array([0.25, 0.75])
        entropy_value = entropy(probabilities)
        ```

    Args:
        probabilities (jnp.ndarray): Array of probability values.

    Returns:
        float: Entropy value.
    """
    log_probs = jnp.log2(probabilities)
    entropy_value = -jnp.sum(probabilities * log_probs)
    return entropy_value


@jax.jit
def gini_impurity(probabilities: jnp.ndarray) -> float:
    """
    Calculate the Gini impurity of a probability distribution using JAX.

    Example usage:
        ```
        probabilities = jnp.array([0.25, 0.75])
        gini_value = gini_impurity(probabilities)
        ```

    Args:
        probabilities (jnp.ndarray): Array of probability values.

    Returns:
        float: Gini impurity value.
    """
    gini_value = 1 - jnp.sum(probabilities**2)
    return gini_value


@jax.jit
def kl_divergence(p: jnp.ndarray, q: jnp.ndarray) -> float:
    """
    Calculate the Kullback-Leibler (KL) divergence between two probability distributions using JAX.

    Example usage:
        ```
        p = jnp.array([0.25, 0.75])
        q = jnp.array([0.5, 0.5])
        kl_value = kl_divergence(p, q)
        ```

    Args:
        p (jnp.ndarray): Array of probability values for distribution p.
        q (jnp.ndarray): Array of probability values for distribution q.

    Returns:
        float: KL divergence value.
    """
    kl_value = jnp.sum(p * jnp.log2(p / q))
    return kl_value


@jax.jit
def count_parameters(params: Any) -> int:
    """
    Count the total number of parameters in a model's parameter dictionary using JAX.

    Example usage:
        ```
        model = MyModel()
        params = model.init(jax.random.PRNGKey(0), jnp.ones(input_shape))
        total_params = count_parameters(params)
        ```

    Args:
        params (Any): Model's parameter dictionary.

    Returns:
        int: Total number of parameters.
    """
    return sum(x.size for x in jax.tree_leaves(params))
