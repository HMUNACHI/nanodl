"""implementations of various encodings"""

import jax.numpy as jnp
import flax.linen as nn

class PositionalEncoding(nn.Module):
    """
    Positional Encoding.

    Args:
        num_embeddings (int): Number of embeddings.
        features (int): Number of features in the embeddings.
    """
    num_embeddings: int
    features: int

    def setup(self):
        positional_encoding = jnp.zeros((self.features, self.num_embeddings))
        position = jnp.arange(0, self.features, dtype=jnp.float32)[:, None]
        div_term = jnp.exp(jnp.arange(0, self.num_embeddings, 2) * (-jnp.log(10000.0) / self.num_embeddings))
        positional_encoding = positional_encoding.at[:, 0::2].set(jnp.sin(position * div_term))
        positional_encoding = positional_encoding.at[:, 1::2].set(jnp.cos(position * div_term))
        self.positional_encoding = positional_encoding.T

    def __call__(self, x):
        x = x + self.positional_encoding[:x.shape[1]]
        return x


class TokenAndPositionEmbedding(nn.Module):
    """
    Token and Position Embedding.

    Args:
        max_len (int): Maximum sequence length.
        vocab_size (int): Vocabulary size.
        embed_dim (int): Embedding dimension.
    """
    max_len : int
    vocab_size : int
    embed_dim : int
    learned_position : bool
    
    def setup(self):
        self.token_embeddings = nn.Embed(num_embeddings=self.vocab_size, features=self.embed_dim)

        if self.learned_position:
            self.position_embeddings = nn.Embed(num_embeddings=self.max_len, features=self.embed_dim)
        else:
            self.position_embeddings = PositionalEncoding(num_embeddings=self.max_len, features=self.embed_dim)

    def __call__(self, x):
        x = self.token_embeddings(x)
        if self.learned_position:
            return x + self.position_embeddings(jnp.arange(x.shape[1]))
        else:
            return x + self.position_embeddings(x)
        


class RotaryPositionalEncoding():
    """
    The rotary position embeddings from RoFormer_ (Su et. al).
    A crucial insight from the method is that the query and keys are
    transformed by rotation matrices which depend on the relative positions.

    Other implementations are available in the Rotary Transformer repo_ and in
    GPT-NeoX_, GPT-NeoX was an inspiration

    .. _RoFormer: https://arxiv.org/abs/2104.09864
    .. _repo: https://github.com/ZhuiyiTechnology/roformer
    .. _GPT-NeoX: https://github.com/EleutherAI/gpt-neox


    .. This is implemented outside nn module as is modifies an external state
       It is also puporsefully broken down for explainability
    """

    def __init__(self, dim_model: int):
        """
        Args:
            dim_model: The dimension of the input and output embeddings.
        """
        super().__init__()
        self.dim_model = dim_model

        inv_freq = 1.0 / (10000 ** (jnp.arange(0, dim_model, 2, dtype=jnp.float32) / dim_model))
        self.inv_freq = inv_freq

        self._seq_len_cached = None
        self._cos_cached = None
        self._sin_cached = None

    def _update_cos_sin_tables(self, x, seq_dimension=1):
        """
        Update the cached cosine and sine tables, if necessary.

        Args:
            x: The input tensor, of shape `(batch_size, seq_len, dim)`.
            seq_dimension: The dimension that represents the sequence length.

        Returns:
            The updated cosine and sine tables.
        """
        seq_len = x.shape[seq_dimension]

        if seq_len != self._seq_len_cached:
            self._seq_len_cached = seq_len
            t = jnp.arange(seq_len, dtype=self.inv_freq.dtype)
            freqs = jnp.outer(t, self.inv_freq)
            emb = jnp.concatenate((freqs, freqs), axis=-1)
            self._cos_cached = jnp.cos(emb)[None, None, :, :]
            self._sin_cached = jnp.sin(emb)[None, None, :, :]

        return self._cos_cached, self._sin_cached

    def rotate_half(self, x):
        """
        Split the input tensor into two halves, rotate the second half by 180 degrees, and concatenate the two halves back together.

        Args:
            x: The input tensor, of shape `(batch_size, seq_len, dim)`.

        Returns:
            The rotated tensor.
        """
        x1, x2 = jnp.split(x, 2, axis=-1)
        return jnp.concatenate((-x2, x1), axis=-1)

    def apply_rotary_pos_emb(self, x, cos, sin):
        """
         Apply the rotary position embeddings to the input tensor.

        Args:
            x: The input tensor, of shape `(batch_size, seq_len, dim)`.
            cos: The cosine table, of shape `(batch_size, 1, seq_len, dim)`.
            sin: The sine table, of shape `(batch_size, 1, seq_len, dim)`.

        Returns:
            The embedded tensor.
        """
        cos = cos[:, :, : x.shape[-2], :]
        sin = sin[:, :, : x.shape[-2], :]
        return (x * cos) + (self.rotate_half(x) * sin)

    def __call__(self, q, k):
        """
         Apply the rotary position embeddings to the query and key tensors.

        Args:
            q: The query tensor, of shape `(batch_size, seq_len, dim)`.
            k: The key tensor, of shape `(batch_size, seq_len, dim)`.

        Returns:
            The embedded query and key tensors.
        """
        self._cos_cached, self._sin_cached = self._update_cos_sin_tables(k, seq_dimension=-2)

        return (
            self.apply_rotary_pos_emb(q, self._cos_cached, self._sin_cached)[0],
            self.apply_rotary_pos_emb(k, self._cos_cached, self._sin_cached)[0],
        )