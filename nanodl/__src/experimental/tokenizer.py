import os
from typing import List, Optional

from sentencepiece import SentencePieceProcessor, SentencePieceTrainer


class Tokenizer:
    """
    A tokenizer class that utilizes SentencePiece to encode and decode text.

    This class can be initialized with either an existing SentencePiece model
    or a dataset to train a new model. It provides methods to encode a string
    to a list of token ids and decode a list of token ids back to a string.

    Attributes:
        sp_model (SentencePieceProcessor): The SentencePiece processor.
        n_words (int): Number of words in the vocabulary.
        bos_id (int): Token id for the beginning of a sentence.
        eos_id (int): Token id for the end of a sentence.
        pad_id (int): Token id for padding.

    Example usage:

    Training a new model and encoding/decoding a string:

    ```python
    # Initialize tokenizer with training data and train a new model.
    text_paths = ['/Users/mac1/Desktop/nanodl/nanodl/__src/utils/sample.txt']

    tokenizer = Tokenizer(training_data=text_paths,
                          vocab_size=100,
                          model_type='bpe',
                          max_sentence_length=50)

    # Encode a sentence.
    encoded_sentence = tokenizer.encode('Hello, world!')
    print(f'Encoded: {encoded_sentence}')

    # Decode the encoded sentence.
    decoded_sentence = tokenizer.decode(encoded_sentence)
    print(f'Decoded: {decoded_sentence}')
    ```

    Loading an existing model and encoding/decoding a string:

    ```python
    # Initialize tokenizer with a pre-trained model.
    tokenizer = Tokenizer(model_path='path/to/model.model')

    # Encode a sentence.
    encoded_sentence = tokenizer.encode('Hello, world!')
    print(f'Encoded: {encoded_sentence}')

    # Decode the encoded sentence.
    decoded_sentence = tokenizer.decode(encoded_sentence)
    print(f'Decoded: {decoded_sentence}')
    ```
    """

    def __init__(
        self,
        training_data: List[str] = None,
        vocab_size: int = None,
        model_type: str = "bpe",
        max_sentence_length: int = 512,
        model_path: Optional[str] = None,
    ):

        if model_path and os.path.isfile(model_path):
            # Load an existing model
            self.sp_model = SentencePieceProcessor(model_file=model_path)
        elif training_data and all(os.path.isfile(f) for f in training_data):
            # Train a new model using a list of data files
            input_files = ",".join(training_data)
            model_prefix = "trained_model"
            SentencePieceTrainer.train(
                input=input_files,
                model_prefix=model_prefix,
                vocab_size=vocab_size,
                model_type=model_type,
                max_sentence_length=max_sentence_length,
            )

            self.sp_model = SentencePieceProcessor(model_file=f"{model_prefix}.model")
        else:
            raise ValueError(
                "Must provide either a model_path or a non-empty training_data list"
            )

        # Initialize token IDs
        self.n_words: int = self.sp_model.vocab_size()
        self.bos_id: int = self.sp_model.bos_id()
        self.eos_id: int = self.sp_model.eos_id()
        self.pad_id: int = self.sp_model.pad_id()

        assert self.sp_model.vocab_size() == self.sp_model.get_piece_size()

    def encode(self, s: str, bos: bool = True, eos: bool = False) -> List[int]:
        """Converts a string into a list of tokens."""
        assert isinstance(s, str)
        t = self.sp_model.encode(s)
        if bos:
            t = [self.bos_id] + t
        if eos:
            t = t + [self.eos_id]
        return t

    def decode(self, t: List[int]) -> str:
        """Converts a list of tokens back into a string."""
        return self.sp_model.decode(t)
