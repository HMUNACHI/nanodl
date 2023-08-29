"""
Implementations of common heuristic algorithms in NLP, mostly written in raw python for readabiity
"""

import re
import json
import numpy as np
import jax.numpy as jnp
from collections import Counter
from typing import Dict, Any, Union, List


class TrieNode:
    """
    TrieNode represents a single node in the Trie data structure.
    
    Attributes:
        children (dict): A dictionary of children nodes.
        is_end_of_word (bool): True if the node marks the end of a word.
    """
    def __init__(self):
        self.children = {}
        self.is_end_of_word = False

class Trie:
    """
    Trie is a tree-like data structure that is used for efficient storage and retrieval
    of a dynamic set of strings.

    Attributes:
        root (TrieNode): The root node of the Trie.
    """
    def __init__(self):
        self.root = TrieNode()
    
    def insert(self, word: str) -> None:
        """
        Inserts a word into the Trie.
        
        Args:
            word (str): The word to be inserted.
        """
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.is_end_of_word = True
    
    def search(self, word: str) -> bool:
        """
        Searches for a word in the Trie.
        
        Args:
            word (str): The word to be searched.
            
        Returns:
            bool: True if the word exists in the Trie, False otherwise.
        """
        node = self.root
        for char in word:
            if char not in node.children:
                return False
            node = node.children[char]
        return node.is_end_of_word
    
    def starts_with(self, prefix: str) -> bool:
        """
        Checks if there is any word in the Trie that starts with a given prefix.
        
        Args:
            prefix (str): The prefix to be checked.
            
        Returns:
            bool: True if there is a word with the given prefix, False otherwise.
        """
        node = self.root
        for char in prefix:
            if char not in node.children:
                return False
            node = node.children[char]
        return True


class KnowledgeGraph:
    def __init__(self):
        self.nodes: Dict[str, Dict[str, Any]] = {}
        self.edges: Dict[str, Dict[str, Union[str, Dict[str, Any]]]] = {}

    def add_node(self, node_id: str, node_data: Dict[str, Any]):
        """
        Add a node to the knowledge graph.

        Args:
            node_id (str): Identifier for the node.
            node_data (dict): Data associated with the node.
        """
        self.nodes[node_id] = node_data

    def add_edge(self, edge_id: str, source_node_id: str, target_node_id: str, edge_data: Dict[str, Any]):
        """
        Add an edge to the knowledge graph.

        Args:
            edge_id (str): Identifier for the edge.
            source_node_id (str): Identifier for the source node.
            target_node_id (str): Identifier for the target node.
            edge_data (dict): Data associated with the edge.
        """
        self.edges[edge_id] = {
            "source_node_id": source_node_id,
            "target_node_id": target_node_id,
            "edge_data": edge_data
        }

    def add_knowledge(self, subject: str, relation: str, obj: str):
        """
        Add knowledge to the graph in the form of a triple.

        Args:
            subject (str): Subject of the triple.
            relation (str): Relation of the triple.
            obj (str): Object of the triple.
        """
        subject_id = self._get_or_create_node(subject)
        obj_id = self._get_or_create_node(obj)
        edge_id = f"{subject_id}_{relation}_{obj_id}"
        self.add_edge(edge_id, subject_id, obj_id, {"relation": relation})

    def _get_or_create_node(self, node_data: Dict[str, Any]) -> str:
        """
        Get or create a node based on node data.

        Args:
            node_data (dict): Data associated with the node.

        Returns:
            str: Identifier of the node.
        """
        for node_id, data in self.nodes.items():
            if data == node_data:
                return node_id
        node_id = str(len(self.nodes) + 1)
        self.add_node(node_id, node_data)
        return node_id

    def get_node(self, node_id: str) -> Dict[str, Any]:
        """
        Get the data associated with a node.

        Args:
            node_id (str): Identifier of the node.

        Returns:
            dict: Data associated with the node.
        """
        return self.nodes.get(node_id)

    def get_edges(self, source_node_id: str) -> List[Dict[str, Any]]:
        """
        Get a list of edges originating from a source node.

        Args:
            source_node_id (str): Identifier of the source node.

        Returns:
            list: List of edge data.
        """
        return [edge for edge_id, edge in self.edges.items() if edge["source_node_id"] == source_node_id]

    def to_dict(self) -> Dict[str, Union[Dict[str, Dict[str, Any]], Dict[str, Dict[str, Union[str, Dict[str, Any]]]]]]:
        """
        Convert the knowledge graph to a dictionary representation.

        Returns:
            dict: Dictionary representing the knowledge graph.
        """
        return {
            "nodes": self.nodes,
            "edges": self.edges
        }

    def to_json(self) -> str:
        """
        Convert the knowledge graph to a JSON string.

        Returns:
            str: JSON string representing the knowledge graph.
        """
        graph_dict = self.to_dict()
        return json.dumps(graph_dict, indent=4)



def extract_ngrams(sequences: List[List[str]], n: int) -> jnp.ndarray:
    """
    Generate n-grams from a list of sequences.
    
    Args:
        sequences (List[List[str]]): List of input sequences, where each sequence is a list of strings.
        n (int): The length of the n-grams.
        
    Returns:
        jnp.ndarray: An array containing n-grams for each input sequence.
    """
    outputs = []
    for sequence in sequences:
        start, end = 0, n
        new_sequence = []
        while end <= len(sequence):
            new_sequence.append(sequence[start:end])
            start += 1
            end += 1
        outputs.append(new_sequence)
    return outputs


def zero_pad_sequences_2d(sequences, max_length):
    """
    Zero-pad a list of 2D sequences along axis 1 to a specified maximum length.
    
    Args:
        sequences (List[np.ndarray]): List of 2D numpy arrays representing sequences.
        max_length (int): Maximum length for padding.
        
    Returns:
        List[np.ndarray]: List of zero-padded 2D sequences.
    """
    padded_sequences = []
    for sequence in sequences:
        pad_length = max_length - sequence.shape[1]
        padded_sequence = np.pad(sequence, ((0, 0), (0, pad_length)), mode='constant')
        padded_sequences.append(padded_sequence)
    return padded_sequences


def rouge(hypotheses: List[str], references: List[str], ngram_sizes: List[int] = [1, 2]) -> dict:
    """
    Calculate the ROUGE (Recall-Oriented Understudy for Gisting Evaluation) metric.
    ROUGE-F1 = (Precision + Recall) / (2⋅Precision⋅Recall)
​    ROUGE-F2 = (1 + beta**2) * (precision * recall) / (beta**2 * precision + recall)

    Args:
        hypotheses (List[str]): List of hypothesis sentences.
        references (List[str]): List of reference sentences.
        ngram_sizes (List[int], optional): List of n-gram sizes. Default is [1, 2].
        
    Returns:
        dict: Dictionary containing precision, recall, and F1-score for each n-gram size.
    """
    def ngrams(sequence: List[str], n: int) -> List[str]:
        """
        Generate n-grams from a sequence.
        """
        return [tuple(sequence[i:i+n]) for i in range(len(sequence) - n + 1)]
    
    def precision_recall_f1(hypothesis_tokens, reference_tokens, n):
        """
        Calculate precision, recall, and F1-score for a specific n-gram size.
        """
        hypothesis_ngrams = set(ngrams(hypothesis_tokens, n))
        reference_ngrams = set(ngrams(reference_tokens, n))
        
        common_ngrams = hypothesis_ngrams.intersection(reference_ngrams)
        
        precision = len(common_ngrams) / len(hypothesis_ngrams) if len(hypothesis_ngrams) > 0 else 0.0
        recall = len(common_ngrams) / len(reference_ngrams) if len(reference_ngrams) > 0 else 0.0
        
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
            
            precision, recall, f1 = precision_recall_f1(hypothesis_tokens, reference_tokens, n)
            total_precision += precision
            total_recall += recall
            total_f1 += f1
        
        average_precision = total_precision / len(hypotheses)
        average_recall = total_recall / len(hypotheses)
        average_f1 = total_f1 / len(hypotheses)
        
        rouge_scores[f'ROUGE-{n}'] = {
            'precision': average_precision,
            'recall': average_recall,
            'f1': average_f1
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
    """
    def ngrams(sequence: List[str], n: int) -> List[str]:
        """
        Generate n-grams from a sequence.
        
        Args:
            sequence (List[str]): Input sequence.
            n (int): Size of n-grams.
            
        Returns:
            List[str]: List of n-grams.
        """
        return [tuple(sequence[i:i+n]) for i in range(len(sequence) - n + 1)]
    
    def modified_precision(hypothesis_tokens, reference_tokens, n):
        """
        Calculate modified precision for a specific n-gram size.
        """
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
    
    final_bleu = brevity_penalty * np.exp(np.mean(np.log(np.clip(bleu_scores, 1e-10, None))))
    return final_bleu



def meteor(reference, hypothesis):
    """
    Calculates the METEOR score between a reference and hypothesis sentence.
    
    Args:
        reference (str): The reference sentence.
        hypothesis (str): The hypothesis sentence.
        
    Returns:
        float: METEOR score.
    """
    
    def tokenize(sentence):
        """
        Tokenizes a sentence into a list of lowercase words.
        
        Args:
            sentence (str): The input sentence.
            
        Returns:
            list: List of lowercase tokens.
        """
        return re.findall(r'\w+', sentence.lower())
    
    def stemming(token):
        """
        Applies simple stemming by lowercasing the token.
        
        Args:
            token (str): The input token.
            
        Returns:
            str: Stemmed token.
        """
        return token.lower()
    
    def exact_matching(reference_tokens, hypothesis_tokens):
        """
        Calculates the count of exact matches between hypothesis and reference tokens.
        
        Args:
            reference_tokens (list): List of reference tokens.
            hypothesis_tokens (list): List of hypothesis tokens.
            
        Returns:
            int: Count of exact matches.
        """
        return sum(1 for token in hypothesis_tokens if token in reference_tokens)
    
    def stemmed_matching(reference_tokens, hypothesis_tokens):
        """
        Calculates the count of stemmed matches between hypothesis and reference tokens.
        
        Args:
            reference_tokens (list): List of reference tokens.
            hypothesis_tokens (list): List of hypothesis tokens.
            
        Returns:
            int: Count of stemmed matches.
        """
        stemmed_reference = [stemming(token) for token in reference_tokens]
        stemmed_hypothesis = [stemming(token) for token in hypothesis_tokens]
        return sum(1 for token in stemmed_hypothesis if token in stemmed_reference)
    
    def precision_recall_f1(match_count, hypothesis_length, reference_length):
        """
        Calculates precision, recall, and F1-score.
        
        Args:
            match_count (int): Count of matched tokens.
            hypothesis_length (int): Length of hypothesis in tokens.
            reference_length (int): Length of reference in tokens.
            
        Returns:
            tuple: Precision, recall, and F1-score.
        """
        precision = match_count / hypothesis_length if hypothesis_length > 0 else 0
        recall = match_count / reference_length if reference_length > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
        return precision, recall, f1
    
    reference_tokens = tokenize(reference)
    hypothesis_tokens = tokenize(hypothesis)

    exact_matches = exact_matching(reference_tokens, hypothesis_tokens)
    stemmed_matches = stemmed_matching(reference_tokens, hypothesis_tokens)

    precision_exact, recall_exact, f1_exact = precision_recall_f1(
        exact_matches, len(hypothesis_tokens), len(reference_tokens)
    )

    precision_stemmed, recall_stemmed, f1_stemmed = precision_recall_f1(
        stemmed_matches, len(hypothesis_tokens), len(reference_tokens)
    )

    alpha = 0.5
    meteor_score = (1 - alpha) * f1_exact + alpha * precision_stemmed * recall_stemmed
    return meteor_score


def cider_score(reference, hypothesis):
    """
    Calculates the CIDEr score between a reference and hypothesis sentence.
    
    Args:
        reference (str): The reference sentence.
        hypothesis (str): The hypothesis sentence.
        
    Returns:
        float: CIDEr score.
    """
    def tokenize(sentence):
        """
        Tokenizes a sentence into a list of lowercase words.
        
        Args:
            sentence (str): The input sentence.
            
        Returns:
            list: List of lowercase tokens.
        """
        return re.findall(r'\w+', sentence.lower())
    
    def ngrams(tokens, n):
        """
        Generates n-grams from a list of tokens.
        
        Args:
            tokens (list): List of tokens.
            n (int): N-gram size.
            
        Returns:
            list: List of n-grams.
        """
        return [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]

    reference_tokens = tokenize(reference)
    hypothesis_tokens = tokenize(hypothesis)

    max_n = 4  # Maximum n-gram size
    weights = [1.0] * max_n  # Weights for different n-gram sizes

    cider_scores = []
    for n in range(1, max_n+1):
        ref_ngrams = ngrams(reference_tokens, n)
        hyp_ngrams = ngrams(hypothesis_tokens, n)
        
        ref_ngram_freq = Counter(ref_ngrams)
        hyp_ngram_freq = Counter(hyp_ngrams)
        
        common_ngrams = set(ref_ngrams) & set(hyp_ngrams)
        
        if len(common_ngrams) == 0:
            cider_scores.append(0)
            continue
        
        precision = sum(min(ref_ngram_freq[ngram], hyp_ngram_freq[ngram]) for ngram in common_ngrams) / len(hyp_ngrams)
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
    """
    log_likelihood = 0.0
    word_count = 0

    for i in range(len(log_probs) - 1):
        predicted_log_prob = log_probs[i]  # Replace this with your language model's log probability
        log_likelihood += predicted_log_prob
        word_count += 1

    average_log_likelihood = log_likelihood / word_count
    perplexity_score = 2 ** (-average_log_likelihood)

    return perplexity_score



def word_error_rate(hypotheses, references):
    """
    Calculate the Word Error Rate (WER) metric.

    Args:
        hypotheses (List[str]): List of hypothesis words.
        references (List[str]): List of reference words.

    Returns:
        float: Word Error Rate score.
    """

    def edit_distance(str1, str2):
        """
        Calculate the edit distance (Levenshtein distance) between two strings.

        Args:
            str1 (List[str]): List of words in the first string.
            str2 (List[str]): List of words in the second string.

        Returns:
            int: Edit distance between the two strings.
        """
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
                    dp[i - 1][j] + 1,     # Deletion
                    dp[i][j - 1] + 1,     # Insertion
                    dp[i - 1][j - 1] + cost  # Substitution or no operation
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


def longestCommonSubsequence(sequence1: List[int], sequence2: List[int]) -> int:
    """
    Compute the length of the longest common subsequence of two sequences.
    
    Args:
        sequence1 (List[int]): First input sequence.
        sequence2 (List[int]): Second input sequence.
        
    Returns:
        int: Length of the longest common subsequence.
    """
    height = len(sequence1) + 1
    width = len(sequence2) + 1
    table = [[0] * width for _ in range(height)]
    for row in range(1, height):
        for col in range(1, width):
            if sequence1[row - 1] == sequence2[col - 1]:
                table[row][col] = table[row - 1][col - 1] + 1
            else:
                table[row][col] = max(table[row][col - 1], table[row - 1][col])
    return table[-1][-1]