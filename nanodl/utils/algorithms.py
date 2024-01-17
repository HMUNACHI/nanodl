import jax
import numpy as np
import jax.numpy as jnp
from typing import List


def batch_cosine_similarities(source: jnp.ndarray, candidates: jnp.ndarray) -> jnp.ndarray:
    """
    Calculate cosine similarities between a source vector and a batch of candidate vectors.
    
    Args:
        source (jnp.ndarray): Source vector.
        candidates (jnp.ndarray): Batch of candidate vectors.
        
    Returns:
        jnp.ndarray: Array of cosine similarity scores.
    """
    dot_products = jnp.einsum("ij,j->i", candidates, source)
    norm_source = jnp.sqrt(jnp.einsum('i,i->', source, source))
    norm_candidates = jnp.sqrt(jnp.einsum('ij,ij->i', candidates, candidates))
    return dot_products / (norm_source * norm_candidates)

@jax.jit
def batch_pearsonr(x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    """
    Calculate batch Pearson correlation coefficient between two sets of vectors.
    
    Args:
        x (jnp.ndarray): First set of vectors.
        y (jnp.ndarray): Second set of vectors.
        
    Returns:
        jnp.ndarray: Array of Pearson correlation coefficients.
    """
    x = jnp.asarray(x).T
    y = jnp.asarray(y).T
    x = x - jnp.expand_dims(x.mean(axis=1), axis=-1)
    y = y - jnp.expand_dims(y.mean(axis=1), axis=-1)
    numerator = jnp.sum(x * y, axis=-1)
    sum_of_squares_x = jnp.einsum('ij,ij -> i', x, x)
    sum_of_squares_y = jnp.einsum('ij,ij -> i', y, y)
    denominator = jnp.sqrt(sum_of_squares_x * sum_of_squares_y)
    return numerator / denominator

def jaccard(sequence1: List, sequence2: List) -> float:
    """
    Calculate Jaccard similarity between two sequences.
    
    Args:
        sequence1 (List): First input sequence.
        sequence2 (List): Second input sequence.
        
    Returns:
        float: Jaccard similarity score.
    """
    numerator = len(set(sequence1).intersection(sequence2))
    denominator = len(set(sequence1).union(sequence2))
    return numerator / denominator

def hamming(sequence1: List, sequence2: List) -> int:
    """
    Calculate Hamming similarity between two sequences.
    
    Args:
        sequence1 (List): First input sequence.
        sequence2 (List): Second input sequence.
        
    Returns:
        int: Hamming similarity score.
    """
    score = 0
    idx1, idx2 = 0, 0
    while idx1 < len(sequence1) and idx2 < len(sequence2):
        if sequence1[idx1] == sequence2[idx2]:
            score += 1
        idx1 += 1
        idx2 += 1
    return score

def classification_scores(labels: List[int], preds: List[int]) -> tuple:
    """
    Calculate classification evaluation scores.
    
    Args:
        labels (List[int]): List of true labels.
        preds (List[int]): List of predicted labels.
        
    Returns:
        tuple: Accuracy, precision, recall, and F1-score.
    """
    true_positives = 0
    true_negatives = 0
    false_positives = 0
    false_negatives = 0
    total = len(preds)

    for pred, label in zip(preds, labels):
        if pred == 1:
            if pred == label:
                true_positives += 1
            else:
                false_positives += 1
        else:
            if pred == label:
                true_negatives += 1
            else:
                false_negatives += 1

    accuracy = (true_positives + true_negatives) / total
    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)
    f1 = 2 * (precision * recall) / (precision + recall)
    return accuracy, precision, recall, f1

def mean_average_precision(preds: List[int], labels: List[int], epsilon: float = 1E-4) -> float:
    """
    Calculate mean average precision.
    
    Args:
        preds (List[int]): List of predicted labels.
        labels (List[int]): List of true labels.
        epsilon (float, optional): A small constant to avoid division by zero. Default is 1E-4.
        
    Returns:
        float: Mean average precision score.
    """
    true_positives = np.ones(len(set(labels))) / epsilon
    false_positives = true_positives.copy()

    for pred, label in zip(preds, labels):
        if pred == label:
            true_positives[pred-1] += 1
        else:
            false_positives[pred-1] += 1

    return sum(true_positives / (true_positives + false_positives)) / len(true_positives)


def entropy(probabilities):
    """
    Calculate the entropy of a probability distribution.
    
    Args:
        probabilities (jax.numpy.ndarray): Array of probability values.
        
    Returns:
        float: Entropy value.
    """
    log_probs = jax.numpy.log2(probabilities)
    entropy = -jax.numpy.sum(probabilities * log_probs)
    return entropy


def gini_impurity(probabilities):
    """
    Calculate the Gini impurity of a probability distribution.
    
    Args:
        probabilities (jax.numpy.ndarray): Array of probability values.
        
    Returns:
        float: Gini impurity value.
    """
    gini_impurity = 1 - jax.numpy.sum(probabilities ** 2)
    return gini_impurity


def kl_divergence(p, q):
    """
    Calculate the Kullback-Leibler (KL) divergence between two probability distributions.
    
    Args:
        p (jax.numpy.ndarray): Array of probability values for distribution p.
        q (jax.numpy.ndarray): Array of probability values for distribution q.
        
    Returns:
        float: KL divergence value.
    """
    kl_divergence = jax.numpy.sum(p * jax.numpy.log2(p / q))
    return kl_divergence


def count_parameters(params):
    """
    Count the total number of parameters in a model's parameter dictionary.
    
    Args:
        params (jax.tree_util.PyTreeDef): Model's parameter dictionary.
        
    Returns:
        int: Total number of parameters.
    """
    return sum(x.size for x in jax.tree_leaves(params))

def mean_reciprocal_rank(predictions):
    """
    Calculate the Mean Reciprocal Rank (MRR) for a list of ranked predictions.
    
    Args:
        predictions (List[List[str]]): List of lists containing ranked predictions.
        
    Returns:
        float: Mean Reciprocal Rank (MRR) score.
    """
    reciprocal_ranks = []
    
    for pred_list in predictions:
        correct_rank = None
        for rank, pred in enumerate(pred_list, start=1):
            if pred == "correct":
                correct_rank = rank
                break
        
        if correct_rank is not None:
            reciprocal_rank = 1.0 / correct_rank
            reciprocal_ranks.append(reciprocal_rank)
    
    if len(reciprocal_ranks) > 0:
        mean_mrr = sum(reciprocal_ranks) / len(reciprocal_ranks)
        return mean_mrr
    else:
        return 0.0