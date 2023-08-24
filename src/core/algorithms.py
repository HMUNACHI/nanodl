import jax 
import jax.numpy as jnp


def batch_cosine_similarities(source, candidates):
    dot_products = jnp.einsum("ij,j->i", candidates, source)
    norm_source = jnp.sqrt(np.einsum('i,i->', source, source))
    norm_candidates = jnp.sqrt(np.einsum('ij,ij->i', candidates, candidates))
    return dot_products / (norm_source * norm_candidates)

@jax.jit
def batch_pearsonr(x, y):
    x = jnp.asarray(x).T
    y = jnp.asarray(y).T
    x = x - jnp.expand_dims(x.mean(axis=1), axis=-1)
    y = y - jnp.expand_dims(y.mean(axis=1), axis=-1)
    numerator = jnp.sum(x * y, axis=-1)
    sum_of_squares_x = jnp.einsum('ij,ij -> i', x, x)
    sum_of_squares_y = jnp.einsum('ij,ij -> i', y, y)
    denominator = jnp.sqrt(sum_of_squares_x * sum_of_squares_y)
    return numerator / denominator



def jaccard(sequence1, sequence2):
    """
    """
    numerator = len(set(sequence1).intersection(sequence2))
    denominator = len(set(sequence1).union(sequence2))
    return numerator / denominator


def hamming(sequence1, sequence2):
    """
    """
    score = 0
    idx1, idx2 = 0, 0
    while idx1 < len(sequence1) and idx2 < sequence2:
        if sequence1[idx1] == sequence2[idx2]:
            score += 1
        idx1 += 1
        idx2 += 2
    return score


def classification_scores(labels, preds):
    """
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



def mean_average_precision(preds, labels, epsilon=1E4):
    true_positives = np.ones(len(set(labels))) / epsilon
    false_positives = true_positives.copy()

    for pred, label in zip(preds, labels):
        if pred == label:
            true_positives[pred-1] += 1
        else:
            false_positives[pred-1] += 1

    return sum(true_positives / (true_positives + false_positives)) / len(true_positives)



def entropy():
    pass

def gini_impurity():
    pass

def kl_divergence():
    pass

def count_parameters(params):
    return sum(x.size for x in jax.tree_leaves(params))