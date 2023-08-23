"""
Implementations of common heuristic algorithms in NLP
"""

import jax
import jax.numpy as jnp


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


def jaccard(sequence1, sequence2):
    """
    """
    numerator = len(set(sequence1).intersection(sequence2))
    denominator = len(set(sequence1).union(sequence2))
    return numerator / denominator


class Trie:
     pass


class KnowledgeGraph:
     pass

print(jaccard(['henry', 'is', 'good'], ['henry', 'is']))