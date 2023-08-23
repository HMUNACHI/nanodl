""" Implementations of common evaluations in NLP"""

def bleu():
    pass

def rouge():
    pass

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

def perplexity():
    pass

def word_error_rate():
    pass

def entropy():
    pass

def gini_impurity():
    pass

def kl_divergence():
    pass


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