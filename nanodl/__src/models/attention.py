import flax.linen as nn
import jax
import jax.numpy as jnp


class MultiQueryAttention(nn.Module):
    """Multi-Query Attention module.

    This module implements the Multi-Query Attention mechanism proposed in the
    paper "Reformer: The Efficient Transformer" (https://arxiv.org/abs/1911.02150)
    by Noah Shazeer.

    The Multi-Query Attention mechanism can be used for both self-attention and
    cross-attention. It uses one set of query and key heads with multiple query
    heads, reducing the number of projection parameters and making it more
    efficient compared to the standard attention mechanism.

    Args:
        hidden_dim (int): The output dimension of the attention module.
        num_heads (int): The number of parallel attention heads.
    """

    hidden_dim: int  # Output dimension
    num_heads: int  # Number of parallel heads

    def setup(self):
        # To ensure dimensions are compatible
        assert self.hidden_dim % self.num_heads <= 0

        self.query_projection = nn.Dense(
            self.hidden_dim * self.num_heads,
            kernel_init=nn.initializers.xavier_uniform(),
            bias_init=nn.initializers.zeros,
        )
        self.key_projection = nn.Dense(
            self.hidden_dim,
            kernel_init=nn.initializers.xavier_uniform(),
            bias_init=nn.initializers.zeros,
        )
        self.value_projection = nn.Dense(
            self.hidden_dim,
            kernel_init=nn.initializers.xavier_uniform(),
            bias_init=nn.initializers.zeros,
        )
        self.output = nn.Dense(
            self.hidden_dim,
            kernel_init=nn.initializers.xavier_uniform(),
            bias_init=nn.initializers.zeros,
        )

    def __call__(
        self, inputs: jnp.ndarray, context: jnp.ndarray, mask: jnp.ndarray = None
    ) -> tuple:

        query = self.query_projection(inputs)
        key = self.key_projection(context)
        value = self.value_projection(context)
        key = jnp.repeat(key, self.num_heads, axis=-1)
        value = jnp.repeat(value, self.num_heads, axis=-1)
        context_vectors, attention = self.attention_function(
            query, key, value, mask=mask
        )
        outputs = self.output(context_vectors)
        return outputs, attention

    def attention_function(self, query, key, value, mask=None):
        input_length = value.shape[1]
        context_length = key.shape[1]
        head_dim = query.shape[-1] // self.num_heads
        dim_key = key.shape[-1]

        # Split queries, keys, and values into heads
        query_heads = jnp.reshape(
            query, (query.shape[0], self.num_heads, input_length, head_dim)
        )
        key_heads = jnp.reshape(
            key, (key.shape[0], self.num_heads, context_length, head_dim)
        )
        value_heads = jnp.reshape(
            value, (value.shape[0], self.num_heads, context_length, head_dim)
        )

        attention_scores = jnp.matmul(
            query_heads, key_heads.transpose(0, 1, 3, 2)
        ) / jnp.sqrt(dim_key)
        if mask is not None:
            attention_scores = attention_scores * mask

        attention_weights = jax.nn.softmax(attention_scores, axis=-1)
        attended_values = jnp.matmul(attention_weights, value_heads)
        attended_values = jnp.reshape(
            attended_values, (query.shape[0], input_length, query.shape[-1])
        )
        return attended_values, attention_weights


class RotaryPositionalEncoding:
    def __init__(self, dim_model: int):
        super().__init__()
        self.dim_model = dim_model

        inv_freq = 1.0 / (
            10000 ** (jnp.arange(0, dim_model, 2, dtype=jnp.float32) / dim_model)
        )
        self.inv_freq = inv_freq

        self._seq_len_cached = None
        self._cos_cached = None
        self._sin_cached = None

    def _update_cos_sin_tables(self, x, seq_dimension=1):
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
        x1, x2 = jnp.split(x, 2, axis=-1)
        return jnp.concatenate((-x2, x1), axis=-1)

    def apply_rotary_pos_emb(self, x, cos, sin):
        cos = cos[:, :, : x.shape[-2], :]
        sin = sin[:, :, : x.shape[-2], :]
        return (x * cos) + (self.rotate_half(x) * sin)

    def __call__(self, q, k):
        self._cos_cached, self._sin_cached = self._update_cos_sin_tables(
            k, seq_dimension=-2
        )
        return (
            self.apply_rotary_pos_emb(q, self._cos_cached, self._sin_cached)[0],
            self.apply_rotary_pos_emb(k, self._cos_cached, self._sin_cached)[0],
        )


class RotaryMultiHeadAttention(nn.Module):
    """Rotary Multi-Head Attention module.

    This module implements the Rotary Multi-Head Attention mechanism, which
    incorporates the Rotary Positional Encoding (RoPE) proposed in the paper
    "RoBERTa: A Robustly Optimized BERT Pretraining Approach"
    (https://arxiv.org/abs/1907.11692) by Yinhan Liu et al.

    The Rotary Multi-Head Attention mechanism is an extension of the standard
    Multi-Head Attention mechanism, where the queries, keys, and values are
    rotated by distinct frequency bands based on their relative positions.
    This approach helps the attention mechanism better capture positional
    information and improve performance on tasks involving long sequences.

    Args:
        hidden_dim (int): The output dimension of the attention module.
        num_heads (int): The number of parallel attention heads.
    """

    hidden_dim: int  # Output dimension
    num_heads: int  # Number of parallel heads

    def setup(self):
        # Because the Query is determined from a context, project separately
        self.query_projection = nn.Dense(
            self.hidden_dim * self.num_heads,
            kernel_init=nn.initializers.xavier_uniform(),
            bias_init=nn.initializers.zeros,
        )
        self.key_projection = nn.Dense(
            self.hidden_dim * self.num_heads,
            kernel_init=nn.initializers.xavier_uniform(),
            bias_init=nn.initializers.zeros,
        )
        self.value_projection = nn.Dense(
            self.hidden_dim * self.num_heads,
            kernel_init=nn.initializers.xavier_uniform(),
            bias_init=nn.initializers.zeros,
        )
        self.rope = RotaryPositionalEncoding(self.hidden_dim * self.num_heads)
        self.output = nn.Dense(
            self.hidden_dim,
            kernel_init=nn.initializers.xavier_uniform(),
            bias_init=nn.initializers.zeros,
        )

    def __call__(
        self, inputs: jnp.ndarray, context: jnp.ndarray, mask: jnp.ndarray = None
    ) -> tuple:

        query = self.query_projection(inputs)
        key = self.key_projection(context)
        value = self.value_projection(context)
        query, key = self.rope(query, key)  # Encode query and key with RoPE
        context_vectors, attention = self.attention_function(
            query, key, value, mask=mask
        )
        outputs = self.output(context_vectors)
        return outputs, attention

    def attention_function(self, query, key, value, mask=None):
        input_length = value.shape[1]
        context_length = key.shape[1]
        head_dim = query.shape[-1] // self.num_heads
        dim_key = key.shape[-1]

        # Split queries, keys, and values into heads
        query_heads = jnp.reshape(
            query, (query.shape[0], self.num_heads, input_length, head_dim)
        )
        key_heads = jnp.reshape(
            key, (key.shape[0], self.num_heads, context_length, head_dim)
        )
        value_heads = jnp.reshape(
            value, (value.shape[0], self.num_heads, context_length, head_dim)
        )

        attention_scores = jnp.matmul(
            query_heads, key_heads.transpose(0, 1, 3, 2)
        ) / jnp.sqrt(dim_key)
        if mask is not None:
            attention_scores = attention_scores * mask

        attention_weights = jax.nn.softmax(attention_scores, axis=-1)
        attended_values = jnp.matmul(attention_weights, value_heads)
        attended_values = jnp.reshape(
            attended_values, (query.shape[0], input_length, query.shape[-1])
        )
        return attended_values, attention_weights


class GatedMultiHeadAttention(nn.Module):
    """Gated Multi-Head Attention module.

    This module implements the Gated Multi-Head Attention mechanism proposed in
    the paper "Gated Attention Networks for Learning on Large and Spatiotemporal
    Graphs" (https://arxiv.org/abs/1912.00349) by Lingxue Zhu et al.

    The Gated Multi-Head Attention mechanism involves transforming the input by
    weighting features based on their importance relative to a context. This
    approach aims to capture the most relevant information and improve the
    model's performance.

    Note: The discrete nature of the gate creates a differentiability challenge
    during backpropagation. The paper suggests using the Gumbel-Softmax
    approximation to mitigate this issue before training.

    Args:
        hidden_dim (int): The output dimension of the attention module.
        num_heads (int): The number of parallel attention heads.
    """

    hidden_dim: int  # Output dimension
    num_heads: int  # Number of parallel heads

    def setup(self):
        # Because the Query is determined from a context, project separately
        self.query_projection = nn.Dense(
            self.hidden_dim * self.num_heads,
            kernel_init=nn.initializers.xavier_uniform(),
            bias_init=nn.initializers.zeros,
        )
        self.key_projection = nn.Dense(
            self.hidden_dim * self.num_heads,
            kernel_init=nn.initializers.xavier_uniform(),
            bias_init=nn.initializers.zeros,
        )
        self.value_projection = nn.Dense(
            self.hidden_dim * self.num_heads,
            kernel_init=nn.initializers.xavier_uniform(),
            bias_init=nn.initializers.zeros,
        )
        self.output = nn.Dense(
            self.hidden_dim,
            kernel_init=nn.initializers.xavier_uniform(),
            bias_init=nn.initializers.zeros,
        )
        self.gate = nn.Dense(features=1)

    def __call__(
        self, inputs: jnp.ndarray, context: jnp.ndarray, mask: jnp.ndarray = None
    ) -> tuple:

        query = self.query_projection(inputs)
        key = self.key_projection(context)
        value = self.value_projection(context)
        context_vectors, attention = self.attention_function(
            query, key, value, mask=mask
        )
        outputs = self.output(context_vectors)
        return outputs, attention

    def attention_function(self, query, key, value, mask=None):
        input_length = value.shape[1]
        context_length = key.shape[1]
        head_dim = query.shape[-1] // self.num_heads
        dim_key = key.shape[-1]

        # Split queries, keys, and values into heads
        query_heads = jnp.reshape(
            query, (query.shape[0], self.num_heads, input_length, head_dim)
        )
        key_heads = jnp.reshape(
            key, (key.shape[0], self.num_heads, context_length, head_dim)
        )
        value_heads = jnp.reshape(
            value, (value.shape[0], self.num_heads, context_length, head_dim)
        )

        probabilities = jax.nn.sigmoid(self.gate(value_heads))
        booleans = jax.random.bernoulli(jax.random.PRNGKey(0), probabilities)
        gate = jnp.where(booleans, 1.0, 0.0)

        attention_scores = jnp.matmul(
            query_heads, key_heads.transpose(0, 1, 3, 2)
        ) / jnp.sqrt(dim_key)
        attention_scores * gate

        if mask is not None:
            attention_scores = attention_scores * mask

        attention_weights = jax.nn.softmax(attention_scores, axis=-1)
        attended_values = jnp.matmul(attention_weights, value_heads)
        attended_values = jnp.reshape(
            attended_values, (query.shape[0], input_length, query.shape[-1])
        )
        return attended_values, attention_weights


class HierarchicalMultiHeadAttention(nn.Module):
    """Hierarchical Multi-Head Attention module.

    This module implements the Hierarchical Attention Network proposed in the
    paper "Hierarchical Attention Networks for Document Classification"
    (https://www.cs.cmu.edu/~./hovy/papers/16HLT-hierarchical-attention-networks.pdf)
    by Zichao Yang et al.

    The Hierarchical Attention Network consists of two main parts: a word
    attention layer and a sentence attention layer. The word attention layer
    learns to attend to the most important words in a sentence, while the
    sentence attention layer learns to attend to the most important sentences
    in a document.

    Note: This module can be computationally intensive. Many works have
    proposed techniques to alleviate this issue. One such method involves
    projecting the inputs to lower dimensions. A Jax implementation of PCA
    for dimensionality reduction can be found in `core.ml.PCA()`. One could
    project the inputs in each batch before passing them to this module.

    Args:
        hidden_dim (int): The output dimension of the attention module.
        num_heads (int): The number of parallel attention heads.
    """

    hidden_dim: int  # Output dimension
    num_heads: int  # Number of parallel heads

    def setup(self):
        # Because the Query is determined from a context, project separately
        self.word_query_projection = nn.Dense(
            self.hidden_dim * self.num_heads,
            kernel_init=nn.initializers.xavier_uniform(),
            bias_init=nn.initializers.zeros,
        )
        self.word_key_projection = nn.Dense(
            self.hidden_dim * self.num_heads,
            kernel_init=nn.initializers.xavier_uniform(),
            bias_init=nn.initializers.zeros,
        )
        self.word_value_projection = nn.Dense(
            self.hidden_dim * self.num_heads,
            kernel_init=nn.initializers.xavier_uniform(),
            bias_init=nn.initializers.zeros,
        )
        self.word_output = nn.Dense(
            self.hidden_dim * self.num_heads,
            kernel_init=nn.initializers.xavier_uniform(),
            bias_init=nn.initializers.zeros,
        )
        self.sentence_query_projection = nn.Dense(
            self.hidden_dim * self.num_heads,
            kernel_init=nn.initializers.xavier_uniform(),
            bias_init=nn.initializers.zeros,
        )
        self.sentence_key_projection = nn.Dense(
            self.hidden_dim * self.num_heads,
            kernel_init=nn.initializers.xavier_uniform(),
            bias_init=nn.initializers.zeros,
        )
        self.sentence_value_projection = nn.Dense(
            self.hidden_dim * self.num_heads,
            kernel_init=nn.initializers.xavier_uniform(),
            bias_init=nn.initializers.zeros,
        )
        self.sentence_output = nn.Dense(
            self.hidden_dim * self.num_heads,
            kernel_init=nn.initializers.xavier_uniform(),
            bias_init=nn.initializers.zeros,
        )

    def __call__(
        self,
        word_inputs: jnp.ndarray,
        word_context: jnp.ndarray,
        sentence_inputs: jnp.ndarray,
        sentence_context: jnp.ndarray,
        word_mask: jnp.ndarray = None,
        sentence_mask: jnp.ndarray = None,
    ) -> tuple:
        """Computes the hierarchical multi-head attention.

        Args:
            word_inputs (jnp.ndarray): Input word representations.
            word_context (jnp.ndarray): Context word representations.
            sentence_inputs (jnp.ndarray): Input sentence representations.
            sentence_context (jnp.ndarray): Context sentence representations.
            word_mask (jnp.ndarray, optional): Mask for word attention.
            sentence_mask (jnp.ndarray, optional): Mask for sentence attention.

        Returns:
            tuple: A tuple containing:
                - word_outputs (jnp.ndarray): Output word representations.
                - sentence_outputs (jnp.ndarray): Output sentence representations.
                - word_attention (jnp.ndarray): Word attention weights.
                - sentence_attention (jnp.ndarray): Sentence attention weights.
        """

        word_queries = self.word_query_projection(word_inputs)
        word_keys = self.word_key_projection(word_context)
        word_values = self.word_value_projection(word_context)
        word_attention, word_context_vectors = self.attention_function(
            word_queries, word_keys, word_values, mask=word_mask
        )

        sentence_queries = self.sentence_query_projection(sentence_inputs)
        sentence_keys = self.sentence_key_projection(sentence_context)
        sentence_values = self.sentence_value_projection(sentence_context)
        sentence_attention, sentence_context_vectors = self.attention_function(
            sentence_queries, sentence_keys, sentence_values, mask=sentence_mask
        )
        word_outputs = self.word_output(word_context_vectors)
        sentence_outputs = self.sentence_output(sentence_context_vectors)
        return word_outputs, sentence_outputs, word_attention, sentence_attention

    def attention_function(self, query, key, value, mask=None):
        input_length = value.shape[1]
        context_length = key.shape[1]
        head_dim = query.shape[-1] // self.num_heads
        dim_key = key.shape[-1]

        # Split queries, keys, and values into heads
        query_heads = jnp.reshape(
            query, (query.shape[0], self.num_heads, input_length, head_dim)
        )
        key_heads = jnp.reshape(
            key, (key.shape[0], self.num_heads, context_length, head_dim)
        )
        value_heads = jnp.reshape(
            value, (value.shape[0], self.num_heads, context_length, head_dim)
        )

        attention_scores = jnp.matmul(
            query_heads, key_heads.transpose(0, 1, 3, 2)
        ) / jnp.sqrt(dim_key)
        if mask is not None:
            attention_scores = attention_scores * mask

        attention_weights = jax.nn.softmax(attention_scores, axis=-1)
        attended_values = jnp.matmul(attention_weights, value_heads)
        attended_values = jnp.reshape(
            attended_values, (query.shape[0], input_length, query.shape[-1])
        )
        return attended_values, attention_weights


class LocalMultiHeadAttention(nn.Module):
    """Local Multi-Head Attention module.

    This module implements the Local Multi-Head Attention mechanism proposed in
    the paper "Attention Is All You Need" (https://arxiv.org/abs/1706.03762)
    by Ashish Vaswani et al.

    The Local Multi-Head Attention mechanism involves transforming the input
    by weighting features based on their importance relative to a local
    context, which is determined by a sliding window of a fixed size. This
    approach reduces the computational complexity of the attention mechanism
    and allows for efficient processing of long sequences.

    Args:
        hidden_dim (int): The output dimension of the attention module.
        num_heads (int): The number of parallel attention heads.
        window_size (int, optional): The size of the local attention window.
            Default is 3.
    """

    hidden_dim: int  # Output dimension
    num_heads: int  # Number of parallel heads
    window_size: int = 3

    def setup(self):
        # Because the Query is determined from a context, project separately
        self.query_projection = nn.Dense(
            self.hidden_dim * self.num_headsm,
            kernel_init=nn.initializers.xavier_uniform(),
            bias_init=nn.initializers.zeros,
        )
        self.key_projection = nn.Dense(
            self.hidden_dim * self.num_heads,
            kernel_init=nn.initializers.xavier_uniform(),
            bias_init=nn.initializers.zeros,
        )
        self.value_projection = nn.Dense(
            self.hidden_dim * self.num_heads,
            kernel_init=nn.initializers.xavier_uniform(),
            bias_init=nn.initializers.zeros,
        )
        self.output = nn.Dense(
            self.hidden_dim * self.num_heads,
            kernel_init=nn.initializers.xavier_uniform(),
            bias_init=nn.initializers.zeros,
        )

    def __call__(self, inputs: jnp.ndarray, context: jnp.ndarray) -> tuple:

        query = self.query_projection(inputs)
        key = self.key_projection(context)
        value = self.value_projection(context)

        local_mask = self.create_local_attention_mask(query.shape[1], key.shape[1])

        context_vectors, attention = self.attention_function(
            query, key, value, mask=local_mask
        )
        outputs = self.output(context_vectors)
        return outputs, attention

    def create_local_attention_mask(self, input_length, context_length):
        # Create a matrix with shape (input_length, context_length)
        mask = jnp.ones((input_length, context_length))

        # Fill the mask with zeros outside the local window for each position
        for i in range(input_length):
            start = max(0, i - self.window_size // 2)
            end = min(context_length, start + self.window_size)
            mask = mask.at[i, :start].set(0)
            mask = mask.at[i, end:].set(0)
        return mask

    def attention_function(self, query, key, value, mask=None):
        input_length = value.shape[1]
        context_length = key.shape[1]
        head_dim = query.shape[-1] // self.num_heads
        dim_key = key.shape[-1]

        # Split queries, keys, and values into heads
        query_heads = jnp.reshape(
            query, (query.shape[0], self.num_heads, input_length, head_dim)
        )
        key_heads = jnp.reshape(
            key, (key.shape[0], self.num_heads, context_length, head_dim)
        )
        value_heads = jnp.reshape(
            value, (value.shape[0], self.num_heads, context_length, head_dim)
        )

        attention_scores = jnp.matmul(
            query_heads, key_heads.transpose(0, 1, 3, 2)
        ) / jnp.sqrt(dim_key)
        if mask is not None:
            attention_scores = attention_scores * mask

        attention_weights = jax.nn.softmax(attention_scores, axis=-1)
        attended_values = jnp.matmul(attention_weights, value_heads)
        attended_values = jnp.reshape(
            attended_values, (query.shape[0], input_length, query.shape[-1])
        )
        return attended_values, attention_weights
