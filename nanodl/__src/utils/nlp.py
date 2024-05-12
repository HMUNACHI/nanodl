import re
from collections import Counter
from typing import List

import numpy as np


def rouge(
    hypotheses: List[str], references: List[str], ngram_sizes: List[int] = [1, 2]
) -> dict:
    """
    Calculate the ROUGE (Recall-Oriented Understudy for Gisting Evaluation) metric.
    ROUGE-F1 = (Precision + Recall) / (2⋅Precision⋅Recall)

    Args:
        hypotheses (List[str]): List of hypothesis sentences.
        references (List[str]): List of reference sentences.
        ngram_sizes (List[int], optional): List of n-gram sizes. Default is [1, 2].

    Returns:
        dict: Dictionary containing precision, recall, and F1-score for each n-gram size.

    Example usage:
        ```
        >>> hypotheses = ["the cat is on the mat", "there is a cat on the mat"]
        >>> references = ["the cat is on the mat", "the cat sits on the mat"]
        >>> rouge_scores = rouge(hypotheses, references, [1, 2])
        >>> print(rouge_scores)
        ```
    """

    def ngrams(sequence: List[str], n: int) -> List[str]:
        return [tuple(sequence[i : i + n]) for i in range(len(sequence) - n + 1)]

    def precision_recall_f1(hypothesis_tokens, reference_tokens, n):
        hypothesis_ngrams = set(ngrams(hypothesis_tokens, n))
        reference_ngrams = set(ngrams(reference_tokens, n))

        common_ngrams = hypothesis_ngrams.intersection(reference_ngrams)

        precision = (
            len(common_ngrams) / len(hypothesis_ngrams)
            if len(hypothesis_ngrams) > 0
            else 0.0
        )
        recall = (
            len(common_ngrams) / len(reference_ngrams)
            if len(reference_ngrams) > 0
            else 0.0
        )

        f1 = 2 * (precision * recall) / (precision + recall + 1e-12)
        return precision, recall, f1

    rouge_scores = {}
    for n in ngram_sizes:
        total_precision = 0.0
        total_recall = 0.0
        total_f1 = 0.0
        for hypothesis, reference in zip(hypotheses, references):
            hypothesis_tokens = hypothesis.split()
            reference_tokens = reference.split()

            precision, recall, f1 = precision_recall_f1(
                hypothesis_tokens, reference_tokens, n
            )
            total_precision += precision
            total_recall += recall
            total_f1 += f1

        average_precision = total_precision / len(hypotheses)
        average_recall = total_recall / len(hypotheses)
        average_f1 = total_f1 / len(hypotheses)

        rouge_scores[f"ROUGE-{n}"] = {
            "precision": average_precision,
            "recall": average_recall,
            "f1": average_f1,
        }

    return rouge_scores


def bleu(hypotheses: List[str], references: List[str], max_ngram: int = 4) -> float:
    """
    Calculate the BLEU (Bilingual Evaluation Understudy) metric.
    BLEU = (BP) * (exp(sum(wn * log(pn))))
    where BP = brevity penalty, wn = weight for n-gram precision, and pn = n-gram precision

    Args:
        hypotheses (List[str]): List of hypothesis sentences.
        references (List[str]): List of reference sentences.
        max_ngram (int, optional): Maximum n-gram size to consider. Default is 4.

    Returns:
        float: BLEU score.

    Example usage:
        ```
        >>> hypotheses = ["the cat is on the mat", "there is a cat on the mat"]
        >>> references = ["the cat is on the mat", "the cat sits on the mat"]
        >>> bleu_score = bleu(hypotheses, references)
        >>> print(bleu_score)
        ```
    """

    def ngrams(sequence: List[str], n: int) -> List[str]:
        return [tuple(sequence[i : i + n]) for i in range(len(sequence) - n + 1)]

    def modified_precision(hypothesis_tokens, reference_tokens, n):
        hypothesis_ngrams = ngrams(hypothesis_tokens, n)
        reference_ngrams = ngrams(reference_tokens, n)

        hypothesis_ngram_counts = Counter(hypothesis_ngrams)
        reference_ngram_counts = Counter(reference_ngrams)

        common_ngrams = hypothesis_ngram_counts & reference_ngram_counts
        common_count = sum(common_ngrams.values())

        if len(hypothesis_ngrams) == 0:
            return 0.0
        else:
            precision = common_count / len(hypothesis_ngrams)
            return precision

    brevity_penalty = np.exp(min(0, 1 - len(hypotheses) / len(references)))
    bleu_scores = []

    for n in range(1, max_ngram + 1):
        ngram_precisions = []
        for hypothesis, reference in zip(hypotheses, references):
            hypothesis_tokens = hypothesis.split()
            reference_tokens = reference.split()

            precision = modified_precision(hypothesis_tokens, reference_tokens, n)
            ngram_precisions.append(precision)

        geometric_mean = np.exp(np.mean(np.log(np.clip(ngram_precisions, 1e-10, None))))
        bleu_scores.append(geometric_mean)

    final_bleu = brevity_penalty * np.exp(
        np.mean(np.log(np.clip(bleu_scores, 1e-10, None)))
    )
    return final_bleu


def meteor(hypothesis: str, reference: str) -> float:
    """
    Calculates the METEOR score between a reference and hypothesis sentence.

    Args:
        reference (str): The reference sentence.
        hypothesis (str): The hypothesis sentence.

    Returns:
        float: METEOR score.

    Example usage:
        ```
        >>> hypothesis = "the cat is on the mat"
        >>> reference = "the cat sits on the mat"
        >>> meteor_score = meteor(hypothesis, reference)
        >>> print(meteor_score)
        ```
    """

    def tokenize(sentence):
        return re.findall(r"\w+", sentence.lower())

    def stemming(token):
        return token.lower()

    def exact_matching(reference_tokens, hypothesis_tokens):
        return sum(1 for token in hypothesis_tokens if token in reference_tokens)

    def stemmed_matching(reference_tokens, hypothesis_tokens):
        stemmed_reference = [stemming(token) for token in reference_tokens]
        stemmed_hypothesis = [stemming(token) for token in hypothesis_tokens]
        return sum(1 for token in stemmed_hypothesis if token in stemmed_reference)

    def precision_recall_f1(match_count, hypothesis_length, reference_length):
        precision = match_count / hypothesis_length if hypothesis_length > 0 else 0
        recall = match_count / reference_length if reference_length > 0 else 0
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall > 0
            else 0
        )
        return precision, recall, f1

    reference_tokens = tokenize(reference)
    hypothesis_tokens = tokenize(hypothesis)

    exact_matches = exact_matching(reference_tokens, hypothesis_tokens)
    stemmed_matches = stemmed_matching(reference_tokens, hypothesis_tokens)

    _, _, f1_exact = precision_recall_f1(
        exact_matches, len(hypothesis_tokens), len(reference_tokens)
    )

    precision_stemmed, recall_stemmed, f1_stemmed = precision_recall_f1(
        stemmed_matches, len(hypothesis_tokens), len(reference_tokens)
    )

    alpha = 0.5
    meteor_score = (1 - alpha) * f1_exact + alpha * precision_stemmed * recall_stemmed
    return meteor_score


def cider_score(hypothesis: str, reference: str) -> float:
    """
    Calculates the CIDEr score between a reference and hypothesis sentence.

    Args:
        reference (str): The reference sentence.
        hypothesis (str): The hypothesis sentence.

    Returns:
        float: CIDEr score.

    Example usage:
        ```
        >>> hypothesis = "the cat is on the mat"
        >>> reference = "the cat sits on the mat"
        >>> score = cider_score(hypothesis, reference)
        >>> print(score)
        ```
    """

    def tokenize(sentence):
        return re.findall(r"\w+", sentence.lower())

    def ngrams(tokens, n):
        return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]

    reference_tokens = tokenize(reference)
    hypothesis_tokens = tokenize(hypothesis)

    max_n = 4  # Maximum n-gram size
    weights = [1.0] * max_n  # Weights for different n-gram sizes

    cider_scores = []
    for n in range(1, max_n + 1):
        ref_ngrams = ngrams(reference_tokens, n)
        hyp_ngrams = ngrams(hypothesis_tokens, n)

        ref_ngram_freq = Counter(ref_ngrams)
        hyp_ngram_freq = Counter(hyp_ngrams)

        common_ngrams = set(ref_ngrams) & set(hyp_ngrams)

        if len(common_ngrams) == 0:
            cider_scores.append(0)
            continue

        precision = sum(
            min(ref_ngram_freq[ngram], hyp_ngram_freq[ngram]) for ngram in common_ngrams
        ) / len(hyp_ngrams)
        ref_ngram_freq_sum = sum(ref_ngram_freq[ngram] for ngram in common_ngrams)
        hyp_ngram_freq_sum = sum(hyp_ngram_freq[ngram] for ngram in common_ngrams)
        recall = ref_ngram_freq_sum / len(ref_ngrams)

        cider_scores.append((precision * recall) / (precision + recall) * 2)

    avg_cider_score = np.average(cider_scores, weights=weights)

    return avg_cider_score


def perplexity(log_probs: List[float]) -> float:
    """
    Calculate the perplexity of a sequence using a list of log probabilities.
    Perplexity = 2^(-average log likelihood)
    where average log likelihood = total log likelihood / total word count

    Args:
        log_probs (List[float]): List of log probabilities for each predicted word.

    Returns:
        float: Perplexity score.

    Example usage:
        ```
        >>> log_probs = [-2.3, -1.7, -0.4]  # Example log probabilities
        >>> perplexity_score = perplexity(log_probs)
        >>> print(perplexity_score)
        ```
    """
    log_likelihood = 0.0
    word_count = 0

    for i in range(len(log_probs) - 1):
        predicted_log_prob = log_probs[
            i
        ]  # Replace this with your language model's log probability
        log_likelihood += predicted_log_prob
        word_count += 1

    average_log_likelihood = log_likelihood / word_count
    perplexity_score = 2 ** (-average_log_likelihood)
    return perplexity_score


def word_error_rate(hypotheses: List[int], references: List[int]) -> float:
    """
    Calculate the Word Error Rate (WER) metric.

    Args:
        hypotheses (List[str]): List of hypothesis words.
        references (List[str]): List of reference words.

    Returns:
        float: Word Error Rate score.

    Example usage:
        ```
        >>> hypotheses = ["the cat is on the mat", "there is a cat on the mat"]
        >>> references = ["the cat is on the mat", "the cat sits on the mat"]
        >>> wer_score = word_error_rate(hypotheses, references)
        >>> print(wer_score)
        ```
    """

    def edit_distance(str1, str2):
        len_str1 = len(str1)
        len_str2 = len(str2)

        dp = [[0] * (len_str2 + 1) for _ in range(len_str1 + 1)]

        for i in range(len_str1 + 1):
            dp[i][0] = i

        for j in range(len_str2 + 1):
            dp[0][j] = j

        for i in range(1, len_str1 + 1):
            for j in range(1, len_str2 + 1):
                cost = 0 if str1[i - 1] == str2[j - 1] else 1
                dp[i][j] = min(
                    dp[i - 1][j] + 1,  # Deletion
                    dp[i][j - 1] + 1,  # Insertion
                    dp[i - 1][j - 1] + cost,  # Substitution or no operation
                )
        return dp[len_str1][len_str2]

    total_edit_distance = 0
    total_reference_length = 0

    for hyp, ref in zip(hypotheses, references):
        edit_dist = edit_distance(hyp.split(), ref.split())
        total_edit_distance += edit_dist
        total_reference_length += len(ref.split())

    wer_score = total_edit_distance / total_reference_length
    return wer_score
