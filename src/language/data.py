"""
Implementations of dataloaders, processing and postprocessing algoritms in nlp
Each class is implemented in pure Python for demonstrative process and due to the 
intricacies of using jax for strings, each dataloader returns a jax.numpy array of tokenized samples.
"""

import os
import random
import jax.numpy as jnp
from collections import Counter, defaultdict

class TextDataLoader:
    """
    A data loader class for handling batched text data from a dataset.

    Args:
    - dataset (list or other iterable): The dataset containing text samples.
    - batch_size (int): The batch size for each iteration.
    - shuffle (bool, optional): Whether to shuffle the dataset indices before iterating. Default is True.

    Attributes:
    - dataset (list or other iterable): The dataset containing text samples.
    - batch_size (int): The batch size for each iteration.
    - shuffle (bool): Whether the dataset indices should be shuffled before iterating.
    - indices (list): List of indices corresponding to the samples in the dataset.

    Methods:
    - __iter__(): Generates batches of text data for iteration.

    Example:
    Assuming you have a list of text samples 'text_samples' and want to create a TextDataLoader
    with a batch size of 32:
    ```
    data_loader = TextDataLoader(dataset=text_samples, batch_size=32)
    for X, Y in data_loader:
        # batch will contain a list of 32 text samples in each iteration
        process_batch(batch)
    ```

    Note:
    This class assumes that the dataset is indexable (e.g., list) and each index provides a text sample.

    """

    def __init__(self, dataset, batch_size, shuffle=True):
        """
        Initializes the TextDataLoader with the provided dataset and batch size.

        Args:
        - dataset (list or other iterable): The dataset containing text samples.
        - batch_size (int): The batch size for each iteration.
        - shuffle (bool, optional): Whether to shuffle the dataset indices before iterating. Default is True.
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = list(range(len(dataset)))
        if self.shuffle:
            random.shuffle(self.indices)
    
    def __iter__(self):
        """
        Generates batches of text data for iteration.

        Yields:
        - batch (list): A batch of text samples.
        """
        for batch_start in range(0, len(self.indices), self.batch_size):
            batch_indices = self.indices[batch_start : batch_start + self.batch_size]
            batch = [self.dataset[idx] for idx in batch_indices]
            try:
                batch = jnp.transpose(jnp.array(batch), (1,0,2))
                yield batch 
            except: # classification problems
                X, Y = [], []
                for x,y in batch:
                    X.append(x)
                    Y.append(y)
                yield jnp.array(X), jnp.array(Y) 



class BytePairEncoder:
    def __init__(self, vocab_size, max_length, preprocess_fn=None):
        """
        Initializes the BytePairEncoder class instance.
        
        Args:
            vocab_size (int): The desired size of the vocabulary.
            max_length (int): The maximum length for padding.
            preprocess_fn: A function which accepts List[str] and return same
                        This is where to include concepts like lower, etc.
        """
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.preprocess_fn = preprocess_fn
        self.bpe_vocab = None
        self.token_to_id = {}
        self.id_to_token = {}
        
    def learn_bpe(self, corpus, num_iters=10):
        """
        Learns Byte-Pair Encoding from the given corpus.

        Args:
            corpus (list of str): The text corpus for learning BPE.
            num_iters (int): The number of BPE merging iterations (default: 10).
        """
        # Initialize the vocabulary with characters
        vocab = Counter(' '.join(corpus).split())
        
        for _ in range(num_iters):
            pairs = defaultdict(int)
            for word, freq in vocab.items():
                symbols = word.split()
                for i in range(len(symbols) - 1):
                    pairs[symbols[i], symbols[i+1]] += freq
            
            if not pairs:
                break
            
            best_pair = max(pairs, key=pairs.get)
            new_vocab = {}
            bigram = ' '.join(best_pair)
            replacement = ''.join(best_pair)
            
            for word in vocab:
                new_word = word.replace(bigram, replacement)
                new_vocab[new_word] = vocab[word]
            vocab = new_vocab
        
        # Create BPE vocabulary
        bpe_vocab = set(' '.join(vocab).split())
        self.bpe_vocab = list(bpe_vocab)
        
        # Add special tokens to the vocabulary
        special_tokens = ['[PAD]', '[START]', '[END]', '[UNK]']
        bpe_vocab.update(special_tokens)
        
        self.bpe_vocab = list(bpe_vocab)
        
        # Create token-to-id and id-to-token mappings
        for i, token in enumerate(self.bpe_vocab):
            self.token_to_id[token] = i
            self.id_to_token[i] = token
    
    def tokenize(self, text):
        """
        Tokenizes the given text using the learned BPE vocabulary.

        Args:
            text (str): The input text to be tokenized.

        Returns:
            list of str: The list of subword tokens.
        """
        if self.bpe_vocab is None:
            raise ValueError("BPE vocabulary not learned. Call learn_bpe() first.")
        
        tokens = []
        words = text.split()
        for word in words:
            tokenized_word = self.tokenize_word(word)
            tokens.extend(tokenized_word)
        return tokens
    
    def tokenize_word(self, word):
        """
        Tokenizes a single word using the learned BPE vocabulary.

        Args:
            word (str): The input word to be tokenized.

        Returns:
            list of str: The list of subword tokens for the word.
        """
        if not self.bpe_vocab:
            raise ValueError("BPE vocabulary not learned. Call learn_bpe() first.")
        
        if word in self.bpe_vocab:
            return [word]
        
        tokens = []
        while len(word) > 0:
            found_subword = False
            for i in range(len(word), 0, -1):
                subword = word[:i]
                if subword in self.bpe_vocab:
                    tokens.append(subword)
                    word = word[i:]
                    found_subword = True
                    break
            if not found_subword:
                tokens.append(word[0])
                word = word[1:]
        return tokens

    def encode(self, text):
        """
        Encodes the given text using the learned BPE vocabulary.

        Args:
            text (str): The input text to be encoded.

        Returns:
            list of int: The list of encoded token IDs.
        """
        if self.bpe_vocab is None:
            raise ValueError("BPE vocabulary not learned. Call learn_bpe() first.")
        
        tokens = self.tokenize(text)

        if self.preprocess_fn is not None:
            tokens = self.preprocess_fn(tokens)

        tokens = ['[START]'] + tokens + ['[END]']
        tokens = self.pad_sequence(tokens)
        encoded_tokens = [self.token_to_id.get(token, self.token_to_id['[UNK]']) for token in tokens]
        return encoded_tokens
    
    def decode(self, encoded_tokens, skip_special_tokens=True):
        """
        Decodes the given encoded tokens into text.

        Args:
            encoded_tokens (list of int): The list of encoded token IDs.
            skip_special_tokens (bool): Whether to skip special tokens during decoding (default: True).

        Returns:
            str: The decoded text.
        """
        decoded_tokens = [self.id_to_token[token_id] for token_id in encoded_tokens]

        if skip_special_tokens:
            decoded_tokens = [token for token in decoded_tokens if token not in ['[START]', '[END]']]
        
        return ' '.join(decoded_tokens)
    

    def pad_sequence(self, sequence, padding_token='[PAD]'):
        """
        Pad a sequence with a specified padding token to match the maximum sequence length.

        Args:
            sequence (list): The input sequence to be padded.
            padding_token (str, optional): The padding token to use. Defaults to '[PAD]'.

        Returns:
            list: The padded sequence.
        """
        if len(sequence) < self.max_length:
            padded_sequence = sequence + [padding_token] * (self.max_length - len(sequence))
        else:
            padded_sequence = sequence[:self.max_length]
        return padded_sequence



class CausalTextDataset:
    """
    A dataset class for handling text data in a causal language modeling setup.

    Args:
    - root_dir (str): The root directory containing text files.
    - tokenizer (BytePairEncoder): A tokenizer to encode the text data.

    Attributes:
    - root_dir (str): The root directory containing text files.
    - file_list (list): List of filenames in the root directory ending with '.txt'.
    - tokenizer (BytePairEncoder): A tokenizer for text encoding.

    Methods:
    - __len__(): Returns the number of files in the dataset.
    - __getitem__(idx): Loads and encodes the text data at the given index.

    """

    def __init__(self, root_dir, tokenizer):
        """
        Initializes the CausalTextDataset with the root directory and tokenizer.

        Args:
        - root_dir (str): The root directory containing text files.
        - tokenizer (BytePairEncoder): A tokenizer to encode the text data.
        """
        self.root_dir = root_dir
        self.file_list = [f for f in os.listdir(root_dir) if f.endswith('.txt')]
        self.tokenizer = tokenizer
    
    def __len__(self):
        """
        Returns the number of files in the dataset.
        """
        return len(self.file_list)
    
    def __getitem__(self, idx):
        """
        Loads and encodes the text data at the given index.

        Args:
        - idx (int): Index of the file to load.

        Returns:
        - input_seq (list): Encoded input sequence.
        - target_seq (list): Encoded target sequence (shifted by one token).
        """
        file_path = os.path.join(self.root_dir, self.file_list[idx])
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        text = self.tokenizer.encode(text)
        return text[:-1], text[1:]



class CausalTextData:
    """
    A class for managing causal language modeling text data.

    Args:
    - data_directory (str): Directory containing text files.
    - batch_size (int): Batch size for data loaders.
    - max_length (int): Maximum sequence length for the tokenizer.
    - vocab_size (int, optional): Vocabulary size for the tokenizer. Default is 1000.
    - validation_split (float, optional): Proportion of data for validation. Default is 0.2.
    - shuffle (bool, optional): Whether to shuffle data. Default is True.
    - corpus_save_path (str, optional): Path to save the corpus. Default is None.
    - preprocess_fn: A function which accepts List[str] and return same
                         This is where to include concepts like lower, etc.

    Attributes:
    - batch_size (int): Batch size for data loaders.
    - validation_split (float): Proportion of data for validation.
    - shuffle (bool): Whether to shuffle data.
    - tokenizer (BytePairEncoder): Tokenizer for text encoding.
    - text_dataset (CausalTextDataset): Dataset containing encoded text.
    - train_dataset (list): List of training samples from the dataset.
    - val_dataset (list): List of validation samples from the dataset.
    - train_loader (TextDataLoader): Data loader for training data.
    - val_loader (TextDataLoader): Data loader for validation data.

    Methods:
    - create_corpus(data_directory, corpus_save_path=None): Creates a corpus from text files.
    - create_dataloader(dataset): Creates a data loader for the given dataset.

    Example:
    ```
    causal_text_data = CausalTextData(data_directory='data', batch_size=32, max_length=128)
    for batch in causal_text_data.train_loader:
        # Process training batch
    ```

    """

    def __init__(self, 
                 data_directory, 
                 batch_size, 
                 max_length,
                 vocab_size=1000, 
                 validation_split=0.2, 
                 shuffle=True, 
                 corpus_save_path=None,
                 preprocess_fn=None):
        """
        Initializes the CausalTextData with the specified parameters.

        Args:
        - batch_size (int): Batch size for data loaders.
        - max_length (int): Maximum sequence length for the tokenizer.
        - vocab_size (int, optional): Vocabulary size for the tokenizer. Default is 1000.
        - validation_split (float, optional): Proportion of data for validation. Default is 0.2.
        - shuffle (bool, optional): Whether to shuffle data. Default is True.
        - corpus_save_path (str, optional): Path to save the corpus. Default is None.
        - preprocess_fn: A function which accepts List[str] and return same
                         This is where to include concepts like lower, etc.
        """
        self.batch_size = batch_size
        self.validation_split = validation_split
        self.shuffle = shuffle
        self.tokenizer = BytePairEncoder(vocab_size, max_length+1, preprocess_fn)
        self.tokenizer.learn_bpe(self.create_corpus(data_directory, corpus_save_path))
        self.text_dataset = CausalTextDataset(data_directory, self.tokenizer)
        
        # Split the dataset into training and validation sets
        num_samples = len(self.text_dataset)
        num_validation = int(num_samples * validation_split)
        num_train = num_samples - num_validation
        
        indices = list(range(num_samples))
        if shuffle:
            random.shuffle(indices)
        
        train_indices, val_indices = indices[:num_train], indices[num_train:]
        self.train_dataset = [self.text_dataset[idx] for idx in train_indices]
        self.val_dataset = [self.text_dataset[idx] for idx in val_indices]
        
        self.train_loader = self.create_dataloader(self.train_dataset)
        self.val_loader = self.create_dataloader(self.val_dataset)

    def create_corpus(self, data_directory, corpus_save_path=None):
        """
        Creates a corpus from text files in the given directory.

        Args:
        - data_directory (str): Directory containing text files.
        - corpus_save_path (str, optional): Path to save the corpus. Default is None.

        Returns:
        - corpus (list): List of text samples from the files.
        """
        corpus = []
        file_list = [f for f in os.listdir(data_directory) if f.endswith('.txt')]
        
        for filename in file_list:
            file_path = os.path.join(data_directory, filename)
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
                corpus.append(text)
        
        if corpus_save_path:
            with open(corpus_save_path, 'w', encoding='utf-8') as f:
                for text in corpus:
                    f.write(text + '\n')
            print(f"Corpus saved to {corpus_save_path}")
            
        return corpus
    
    def create_dataloader(self, dataset):
        """
        Creates a data loader for the given dataset.

        Args:
        - dataset (list): List of samples for the data loader.

        Returns:
        - dataloader (TextDataLoader): Data loader for the dataset.
        """
        return TextDataLoader(dataset, self.batch_size, shuffle=self.shuffle)
    


class Seq2SeqTextDataset:
    """
    A dataset class for handling text data in a causal language modeling setup.

    Args:
    - root_dir (str): The root directory containing text files.
    - tokenizer (BytePairEncoder): A tokenizer to encode the text data.

    Attributes:
    - root_dir (str): The root directory containing text files.
      (one for source and target each)
    - file_list (list): List of filenames in the root directory ending with '.txt'.
      (one for source and target each)
    - tokenizer (BytePairEncoder): A tokenizer for text encoding.
      (one for source and target each)

    Methods:
    - __len__(): Returns the number of files in the dataset.
    - __getitem__(idx): Loads and encodes the text data at the given index.

    """

    def __init__(self, 
                 source_root_dir, 
                 target_root_dir, 
                 source_tokenizer,
                 target_tokenizer):
        """
        Initializes the CausalTextDataset with the root directory and tokenizer.

        Args:
        - root_dir (str): The root directory containing text files. 
          (one for source and target each)
        - tokenizer (BytePairEncoder): A tokenizer to encode the text data. 
          (one for source and target each)
        """
        self.source_root_dir = source_root_dir
        self.source_file_list = [f for f in os.listdir(source_root_dir) if f.endswith('.txt')]
        self.source_tokenizer = source_tokenizer

        self.target_root_dir = target_root_dir
        self.target_file_list = [f for f in os.listdir(target_root_dir) if f.endswith('.txt')]
        self.target_tokenizer = target_tokenizer
    
    def __len__(self):
        """
        Returns the number of files in the dataset.
        """
        return len(self.source_file_list)
    
    def __getitem__(self, idx):
        """
        Loads and encodes the text data at the given index.

        Args:
        - idx (int): Index of the file to load.

        Returns:
        - input_seq (list): Encoded input sequence.
        - target_seq (list): Encoded target sequence (shifted by one token).
        """
        file_path = os.path.join(self.source_root_dir, self.source_file_list[idx])
        with open(file_path, 'r', encoding='utf-8') as f:
            source_text = f.read()
        source_text = self.source_tokenizer.encode(source_text)

        file_path = os.path.join(self.target_root_dir, self.target_file_list[idx])
        with open(file_path, 'r', encoding='utf-8') as f:
            target_text = f.read()
        target_text = self.target_tokenizer.encode(target_text)

        return source_text, target_text



class Seq2SeqTextData:
    """
    A class for managing causal language modeling text data.

    Args:
    - data_directory (str): Directory containing text files.
      (one for source and target each)
    - batch_size (int): Batch size for data loaders.
    - max_length (int): Maximum sequence length for the tokenizer.
    - vocab_size (int, optional): Vocabulary size for the tokenizer. Default is 1000.
    - validation_split (float, optional): Proportion of data for validation. Default is 0.2.
    - shuffle (bool, optional): Whether to shuffle data. Default is True.
    - corpus_save_path (str, optional): Path to save the corpus. Default is None.
      (one for source and target each)
    - preprocess_fn: A function which accepts List[str] and return same
                         (one for source and target each)
                         This is where to include concepts like lower, etc.

    - data_directory (str): Directory containing text files.
    - batch_size (int): Batch size for data loaders.
    - validation_split (float): Proportion of data for validation.
    - shuffle (bool): Whether to shuffle data.
    - tokenizer (BytePairEncoder): Tokenizer for text encoding.
    - text_dataset (CausalTextDataset): Dataset containing encoded text.
    - train_dataset (list): List of training samples from the dataset.
    - val_dataset (list): List of validation samples from the dataset.
    - train_loader (TextDataLoader): Data loader for training data.
    - val_loader (TextDataLoader): Data loader for validation data.

    Methods:
    - create_corpus(data_directory, corpus_save_path=None): Creates a corpus from text files.
    - create_dataloader(dataset): Creates a data loader for the given dataset.

    Example:
    ```
    causal_text_data = CausalTextData(data_directory='data', batch_size=32, max_length=128)
    for batch in causal_text_data.train_loader:
        # Process training batch
    ```

    """

    def __init__(self, 
                 source_data_directory, 
                 target_data_directory,
                 batch_size, 
                 max_length,
                 vocab_size=1000, 
                 validation_split=0.2, 
                 shuffle=True, 
                 source_corpus_save_path=None,
                 target_corpus_save_path=None,
                 source_preprocess_fn=None,
                 target_preprocess_fn=None):
        """
        Initializes the CausalTextData with the specified parameters.

        Args:
        - data_directory (str): Directory containing text files.
          (one for source and target each)
        - batch_size (int): Batch size for data loaders.
        - max_length (int): Maximum sequence length for the tokenizer.
        - vocab_size (int, optional): Vocabulary size for the tokenizer. Default is 1000.
        - validation_split (float, optional): Proportion of data for validation. Default is 0.2.
        - shuffle (bool, optional): Whether to shuffle data. Default is True.
        - corpus_save_path (str, optional): Path to save the corpus. Default is None.
          (one for source and target each)
        - preprocess_fn: A function which accepts List[str] and return same
                         (one for source and target each)
                         This is where to include concepts like lower, etc.
        """
        self.batch_size = batch_size
        self.validation_split = validation_split
        self.shuffle = shuffle
        self.source_tokenizer = BytePairEncoder(vocab_size, max_length, source_preprocess_fn)
        self.source_tokenizer.learn_bpe(self.create_corpus(source_data_directory, source_corpus_save_path))
        self.target_tokenizer = BytePairEncoder(vocab_size, max_length, target_preprocess_fn)
        self.target_tokenizer.learn_bpe(self.create_corpus(target_data_directory, target_corpus_save_path))
        self.text_dataset = Seq2SeqTextDataset(source_data_directory, 
                                                 target_data_directory,
                                                 self.source_tokenizer,
                                                 self.target_tokenizer)

        # Split the dataset into training and validation sets
        num_samples = len(self.text_dataset)
        num_validation = int(num_samples * validation_split)
        num_train = num_samples - num_validation
        
        indices = list(range(num_samples))
        if shuffle:
            random.shuffle(indices)
        
        train_indices, val_indices = indices[:num_train], indices[num_train:]
        self.train_dataset = [self.text_dataset[idx] for idx in train_indices]
        self.val_dataset = [self.text_dataset[idx] for idx in val_indices]
        
        self.train_loader = self.create_dataloader(self.train_dataset)
        self.val_loader = self.create_dataloader(self.val_dataset)

    def create_corpus(self, data_directory, corpus_save_path=None):
        """
        Creates a corpus from text files in the given directory.

        Args:
        - data_directory (str): Directory containing text files.
        - corpus_save_path (str, optional): Path to save the corpus. Default is None.

        Returns:
        - corpus (list): List of text samples from the files.
        """
        corpus = []
        file_list = [f for f in os.listdir(data_directory) if f.endswith('.txt')]
        
        for filename in file_list:
            file_path = os.path.join(data_directory, filename)
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
                corpus.append(text)
        
        if corpus_save_path:
            with open(corpus_save_path, 'w', encoding='utf-8') as f:
                for text in corpus:
                    f.write(text + '\n')
            print(f"Corpus saved to {corpus_save_path}")
            
        return corpus
    
    def create_dataloader(self, dataset):
        """
        Creates a data loader for the given dataset.

        Args:
        - dataset (list): List of samples for the data loader.

        Returns:
        - dataloader (TextDataLoader): Data loader for the dataset.
        """
        return TextDataLoader(dataset, self.batch_size, shuffle=self.shuffle)
    


class ClassificationTextDataset:
    """
    A dataset class for handling text data in a causal language modeling setup.

    Args:
    - root_dir (str): The root directory containing text files.
    - tokenizer (BytePairEncoder): A tokenizer to encode the text data.
    - labels List[str]: List of the corresponding labels

    Attributes:
    - root_dir (str): The root directory containing text files.
    - file_list (list): List of filenames in the root directory ending with '.txt'.
    - tokenizer (BytePairEncoder): A tokenizer for text encoding.
    - labels List[str]: List of the corresponding labels

    Methods:
    - __len__(): Returns the number of files in the dataset.
    - __getitem__(idx): Loads and encodes the text data at the given index.

    """

    def __init__(self, root_dir, tokenizer, labels):
        """
        Initializes the CausalTextDataset with the root directory and tokenizer.

        Args:
        - root_dir (str): The root directory containing text files.
        - tokenizer (BytePairEncoder): A tokenizer to encode the text data.
        - labels List[str]: List of the corresponding labels
        """
        self.root_dir = root_dir
        self.file_list = [f for f in os.listdir(root_dir) if f.endswith('.txt')]
        self.tokenizer = tokenizer
        self.labels = labels
    
    def __len__(self):
        """
        Returns the number of files in the dataset.
        """
        return len(self.file_list)
    
    def __getitem__(self, idx):
        """
        Loads and encodes the text data at the given index.

        Args:
        - idx (int): Index of the file to load.

        Returns:
        - input_seq (list): Encoded input sequence.
        - target_seq (list): Encoded target sequence (shifted by one token).
        """
        file_path = os.path.join(self.root_dir, self.file_list[idx])
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        text = self.tokenizer.encode(text)
        return text, self.labels[idx]



class ClassificationTextData:
    """
    A class for managing causal language modeling text data.

    Args:
    - data_directory (str): Directory containing text files.
    - labels List[str]: List of the corresponding labels
    - batch_size (int): Batch size for data loaders.
    - max_length (int): Maximum sequence length for the tokenizer.
    - vocab_size (int, optional): Vocabulary size for the tokenizer. Default is 1000.
    - validation_split (float, optional): Proportion of data for validation. Default is 0.2.
    - shuffle (bool, optional): Whether to shuffle data. Default is True.
    - corpus_save_path (str, optional): Path to save the corpus. Default is None.
    - preprocess_fn: A function which accepts List[str] and return same
                         This is where to include concepts like lower, etc.

    Attributes:
    - batch_size (int): Batch size for data loaders.
    - validation_split (float): Proportion of data for validation.
    - shuffle (bool): Whether to shuffle data.
    - tokenizer (BytePairEncoder): Tokenizer for text encoding.
    - text_dataset (CausalTextDataset): Dataset containing encoded text.
    - train_dataset (list): List of training samples from the dataset.
    - val_dataset (list): List of validation samples from the dataset.
    - train_loader (TextDataLoader): Data loader for training data.
    - val_loader (TextDataLoader): Data loader for validation data.
    - labels List[str]: List of the corresponding labels

    Methods:
    - create_corpus(data_directory, corpus_save_path=None): Creates a corpus from text files.
    - create_dataloader(dataset): Creates a data loader for the given dataset.

    Example:
    ```
    causal_text_data = CausalTextData(data_directory='data', batch_size=32, max_length=128)
    for batch in causal_text_data.train_loader:
        # Process training batch
    ```

    """

    def __init__(self, 
                 data_directory, 
                 labels,
                 batch_size, 
                 max_length,
                 vocab_size=1000, 
                 validation_split=0.2, 
                 shuffle=True, 
                 corpus_save_path=None,
                 preprocess_fn=None):
        """
        Initializes the CausalTextData with the specified parameters.

        Args:
        - data_directory (str): Directory with text files
        - labels List[str]: List of the corresponding labels
        - batch_size (int): Batch size for data loaders.
        - max_length (int): Maximum sequence length for the tokenizer.
        - vocab_size (int, optional): Vocabulary size for the tokenizer. Default is 1000.
        - validation_split (float, optional): Proportion of data for validation. Default is 0.2.
        - shuffle (bool, optional): Whether to shuffle data. Default is True.
        - corpus_save_path (str, optional): Path to save the corpus. Default is None.
        - preprocess_fn: A function which accepts List[str] and return same
                         This is where to include concepts like lower, etc.
        """
        self.batch_size = batch_size
        self.validation_split = validation_split
        self.shuffle = shuffle
        self.tokenizer = BytePairEncoder(vocab_size, max_length+1, preprocess_fn)
        self.tokenizer.learn_bpe(self.create_corpus(data_directory, corpus_save_path))
        self.text_dataset = ClassificationTextDataset(data_directory, self.tokenizer, labels)
        
        # Split the dataset into training and validation sets
        num_samples = len(self.text_dataset)
        num_validation = int(num_samples * validation_split)
        num_train = num_samples - num_validation
        
        indices = list(range(num_samples))
        if shuffle:
            random.shuffle(indices)
        
        train_indices, val_indices = indices[:num_train], indices[num_train:]
        self.train_dataset = [self.text_dataset[idx] for idx in train_indices]
        self.val_dataset = [self.text_dataset[idx] for idx in val_indices]
        
        self.train_loader = self.create_dataloader(self.train_dataset)
        self.val_loader = self.create_dataloader(self.val_dataset)

    def create_corpus(self, data_directory, corpus_save_path=None):
        """
        Creates a corpus from text files in the given directory.

        Args:
        - data_directory (str): Directory containing text files.
        - corpus_save_path (str, optional): Path to save the corpus. Default is None.

        Returns:
        - corpus (list): List of text samples from the files.
        """
        corpus = []
        file_list = [f for f in os.listdir(data_directory) if f.endswith('.txt')]
        
        for filename in file_list:
            file_path = os.path.join(data_directory, filename)
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
                corpus.append(text)
        
        if corpus_save_path:
            with open(corpus_save_path, 'w', encoding='utf-8') as f:
                for text in corpus:
                    f.write(text + '\n')
            print(f"Corpus saved to {corpus_save_path}")
            
        return corpus
    
    def create_dataloader(self, dataset):
        """
        Creates a data loader for the given dataset.

        Args:
        - dataset (list): List of samples for the data loader.

        Returns:
        - dataloader (TextDataLoader): Data loader for the dataset.
        """
        return TextDataLoader(dataset, self.batch_size, shuffle=self.shuffle)