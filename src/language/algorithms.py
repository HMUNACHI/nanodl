"""
Implementations of common heuristic algorithms in NLP, mostly written in raw python for readabiity
"""

import jax
import numpy as np
import jax.numpy as jnp


class Trie:
     pass


class KnowledgeGraph:
     pass


def ngrams(sequences, n):
    """
    """
    outputs = []
    for sequence in sequences:
        start, end = 0, n
        new_sequence = []
        while end < len(sequence):
            new_sequence.append(sequence[start:end])
            start += 1
            end += 1
        outputs.append(new_sequence)
    return jnp.array(outputs)


def zero_pad_sequence_3d(array, last_dim):
    """
    """
    new_array = np.zeros((len(array),21,last_dim),dtype=array[0].dtype)
    for idx, x in enumerate(array):
        mid_dim = min(x.shape[0], 21)
        new_array[idx, :mid_dim, :] = x[:mid_dim, :]
    return new_array


def zero_pad_sequence_2d(array, last_dim):
    """
    """
    g = []
    for x in array:
        y = np.zeros((len(x),last_dim),dtype=x.dtype)
        print(y.shape, y[:, :len(x[0])].shape, x.shape)
        y[:, :len(x[0])] = x
        g.append(y)
    return np.concatenate(g, axis=0)


def longestCommonSubsequence(sequence1, sequence2):
    """
    """
    height = len(sequence1) + 1
    width = len(sequence2) + 1
    table = [[0]*width for _ in range(height)]
    for row in range(1, height):
        for col in range(1, width):
            if sequence1[row-1] == sequence2[col-1]:
                table[row][col] = table[row-1][col-1] + 1
            else:
                table[row][col] = max(table[row][col-1], table[row-1][col])
    return table[-1][-1]


def rouge():
    pass

def bleu():
    pass

def perplexity():
    pass

def word_error_rate():
    pass