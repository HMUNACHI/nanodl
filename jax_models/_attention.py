'''
Implementations of various attention mechanisms variants.
While multiple concepts can be merged into a generalised module,
Each technique is isolated for clarity and easy copy and paste **wink**
'''

import jax
import jax.numpy as jnp
import flax.linen as nn


class SelfMultiHeadAttention(nn.Module):
    """
    https://arxiv.org/abs/1706.03762 (Vaswani et. al. 2017)
    This involves transforming the input by weighting features by importance.
    """
    hidden_dim : int  # Output dimension
    num_heads : int  # Number of parallel heads

    def setup(self):
        # Stack all weight matrices together for efficiency
        self.projection = nn.Dense(3*self.hidden_dim,
                                 kernel_init=nn.initializers.xavier_uniform(),
                                 bias_init=nn.initializers.zeros 
                                )
        self.output = nn.Dense(self.hidden_dim,
                               kernel_init=nn.initializers.xavier_uniform(),
                               bias_init=nn.initializers.zeros)


    def __call__(self, 
                 inputs: jnp.ndarray, 
                 mask: jnp.ndarray = None) -> tuple:

        """
        Args:
            context: optional - context ((batch_size, seq_len, dims))
            Mask: optional - masks where reqions to ignore are flipped to os
                  regions to attend to are 1s (batch_size, seq_len, dims)

        Return: outputs (batch_size, seq_len, seq_len)
                attention matrixes (batch_size, heads, seq_len, seq_len)
        """
        projections = self.projection(inputs)
        query, key, value = jnp.array_split(projections, 3, axis=-1)
        context_vectors, attention = self.attention_function(query,key, value, mask=mask)
        outputs = self.output(context_vectors)
        return outputs, attention
    
    def attention_function(self, query, key, value, mask=None):
        input_length = query.shape[1]
        context_length = key.shape[1]
        head_dim = query.shape[-1] // self.num_heads
        dim_key = key.shape[-1]

        # Split queries, keys, and values into heads
        query_heads = jnp.reshape(query, (query.shape[0], self.num_heads, input_length, head_dim))
        key_heads = jnp.reshape(key, (key.shape[0], self.num_heads, context_length, head_dim))
        value_heads = jnp.reshape(value, (value.shape[0], self.num_heads, context_length, head_dim))

        attention_scores = jnp.matmul(query_heads, key_heads.transpose(0, 1, 3, 2)) / jnp.sqrt(dim_key)
        if mask is not None:
            attention_scores = attention_scores * mask

        attention_weights = jax.nn.softmax(attention_scores, axis=-1)
        attended_values = jnp.matmul(attention_weights, value_heads)
        attended_values = jnp.reshape(attended_values, (query.shape[0], input_length, query.shape[-1]))
        return attended_values, attention_weights


class CrossMultiHeadAttention(nn.Module):
    """
    https://arxiv.org/abs/1706.03762 (Vaswani et. al. 2017)
    This involves transforming the input by weighting features by importance relative to a context
    """
    hidden_dim : int  # Output dimension
    num_heads : int  # Number of parallel heads

    def setup(self):
        # Because the Query is determined from a context, project separately
        self.query_projection = nn.Dense(self.hidden_dim,
                                 kernel_init=nn.initializers.xavier_uniform(),
                                 bias_init=nn.initializers.zeros 
                                )
        self.key_projection = nn.Dense(self.hidden_dim,
                                 kernel_init=nn.initializers.xavier_uniform(),
                                 bias_init=nn.initializers.zeros 
                                )
        self.value_projection = nn.Dense(self.hidden_dim,
                                 kernel_init=nn.initializers.xavier_uniform(),
                                 bias_init=nn.initializers.zeros 
                                )
        self.output = nn.Dense(self.hidden_dim,
                               kernel_init=nn.initializers.xavier_uniform(),
                               bias_init=nn.initializers.zeros)


    def __call__(self, 
                 inputs: jnp.ndarray, 
                 context: jnp.ndarray, 
                 mask: jnp.ndarray = None) -> tuple:

        """
        Args:
            inputs: inputs ((batch_size, seq_len, dims))
            context: optional - context ((batch_size, seq_len, dims))
            Mask: optional - masks where reqions to ignore are flipped to os
                  regions to attend to are 1s (batch_size, seq_len, dims)

        Return: outputs (batch_size, seq_len, seq_len)
                attention matrixes (batch_size, heads, seq_len, seq_len)
        """
        query = self.query_projection(inputs)
        key = self.key_projection(context)
        value = self.value_projection(context)
        context_vectors, attention = self.attention_function(query,key, value, mask=mask)
        outputs = self.output(context_vectors)
        return outputs, attention
    
    def attention_function(self, query, key, value, mask=None):
        input_length = query.shape[1]
        context_length = key.shape[1]
        head_dim = query.shape[-1] // self.num_heads
        dim_key = key.shape[-1]

        # Split queries, keys, and values into heads
        query_heads = jnp.reshape(query, (query.shape[0], self.num_heads, input_length, head_dim))
        key_heads = jnp.reshape(key, (key.shape[0], self.num_heads, context_length, head_dim))
        value_heads = jnp.reshape(value, (value.shape[0], self.num_heads, context_length, head_dim))

        attention_scores = jnp.matmul(query_heads, key_heads.transpose(0, 1, 3, 2)) / jnp.sqrt(dim_key)
        if mask is not None:
            attention_scores = attention_scores * mask
        
        attention_weights = jax.nn.softmax(attention_scores, axis=-1)
        attended_values = jnp.matmul(attention_weights, value_heads)
        attended_values = jnp.reshape(attended_values, (query.shape[0], input_length, query.shape[-1]))
        return attended_values, attention_weights



class MultiQueryAttention(nn.Module):
    """
    https://arxiv.org/abs/1911.02150 (Noah Shazeer, 2019)
    This can be used for both self and cross attention
    One query and key heads are used with multiple query heads
    This reduces the number of projection parameters, hence efficient
    """
    hidden_dim : int  # Output dimension
    num_heads : int  # Number of parallel heads

    def setup(self):
        # To ensure dimensions are compatible
        assert self.hidden_dim % self.num_heads <= 0

        self.query_projection = nn.Dense(self.hidden_dim,
                                 kernel_init=nn.initializers.xavier_uniform(),
                                 bias_init=nn.initializers.zeros 
                                )
        self.key_projection = nn.Dense(self.hidden_dim//self.num_heads,
                                 kernel_init=nn.initializers.xavier_uniform(),
                                 bias_init=nn.initializers.zeros 
                                )
        self.value_projection = nn.Dense(self.hidden_dim//self.num_heads,
                                 kernel_init=nn.initializers.xavier_uniform(),
                                 bias_init=nn.initializers.zeros 
                                )
        self.output = nn.Dense(self.hidden_dim,
                               kernel_init=nn.initializers.xavier_uniform(),
                               bias_init=nn.initializers.zeros)


    def __call__(self, 
                 inputs: jnp.ndarray, 
                 context: jnp.ndarray, 
                 mask: jnp.ndarray = None) -> tuple:

        """
        Args:
            inputs: inputs ((batch_size, seq_len, dims))
            context: optional - context ((batch_size, seq_len, dims))
            Mask: optional - masks where reqions to ignore are flipped to os
                  regions to attend to are 1s (batch_size, seq_len, dims)

        Return: outputs (batch_size, seq_len, seq_len)
                attention matrixes (batch_size, heads, seq_len, seq_len)
        """
        query = self.query_projection(inputs)
        key = self.key_projection(context)
        value = self.value_projection(context)
        key = jnp.repeat(key, self.num_heads, axis=-1)
        value = jnp.repeat(value, self.num_heads, axis=-1)
        context_vectors, attention = self.attention_function(query,key, value, mask=mask)
        outputs = self.output(context_vectors)
        return outputs, attention
    
    def attention_function(self, query, key, value, mask=None):
        input_length = query.shape[1]
        context_length = key.shape[1]
        head_dim = query.shape[-1] // self.num_heads
        dim_key = key.shape[-1]

        # Split queries, keys, and values into heads
        query_heads = jnp.reshape(query, (query.shape[0], self.num_heads, input_length, head_dim))
        key_heads = jnp.reshape(key, (key.shape[0], self.num_heads, context_length, head_dim))
        value_heads = jnp.reshape(value, (value.shape[0], self.num_heads, context_length, head_dim))

        attention_scores = jnp.matmul(query_heads, key_heads.transpose(0, 1, 3, 2)) / jnp.sqrt(dim_key)
        if mask is not None:
            attention_scores = attention_scores * mask

        attention_weights = jax.nn.softmax(attention_scores, axis=-1)
        attended_values = jnp.matmul(attention_weights, value_heads)
        attended_values = jnp.reshape(attended_values, (query.shape[0], input_length, query.shape[-1]))
        return attended_values, attention_weights



class RelativeMultiHeadAttention(nn.Module):
    """
    
    """
    hidden_dim : int  # Output dimension
    num_heads : int  # Number of parallel heads

    def setup(self):
        # Because the Query is determined from a context, project separately
        self.query_projection = nn.Dense(self.hidden_dim,
                                 kernel_init=nn.initializers.xavier_uniform(),
                                 bias_init=nn.initializers.zeros 
                                )
        self.key_projection = nn.Dense(self.hidden_dim,
                                 kernel_init=nn.initializers.xavier_uniform(),
                                 bias_init=nn.initializers.zeros 
                                )
        self.value_projection = nn.Dense(self.hidden_dim,
                                 kernel_init=nn.initializers.xavier_uniform(),
                                 bias_init=nn.initializers.zeros 
                                )
        self.output = nn.Dense(self.hidden_dim,
                               kernel_init=nn.initializers.xavier_uniform(),
                               bias_init=nn.initializers.zeros)


    def __call__(self, 
                 inputs: jnp.ndarray, 
                 context: jnp.ndarray, 
                 mask: jnp.ndarray = None, 
                 clip: int = 3) -> tuple:

        """
        Args:
            inputs: inputs ((batch_size, seq_len, dims))
            context: optional - context ((batch_size, seq_len, dims))
            clip: the k value at which to clip the relative position by
            Mask: optional - masks where reqions to ignore are flipped to os
                  regions to attend to are 1s (batch_size, seq_len, dims)

        Return: outputs (batch_size, seq_len, seq_len)
                attention matrixes (batch_size, heads, seq_len, seq_len)
        """
        query = self.query_projection(inputs)
        key = self.key_projection(context)
        value = self.value_projection(context)

        query_relative_positions = jnp.expand_dims(jnp.arange(query.shape[2]), axis=0) 
        query_relative_positions -= jnp.expand_dims(jnp.arange(query.shape[1]), axis=1)
        query_relative_positions = jnp.where(query_relative_positions < clip, query_relative_positions, clip)
        query_relative_positions = jnp.where(query_relative_positions > -clip, query_relative_positions, -clip)
        query += query_relative_positions

        value_relative_positions = jnp.expand_dims(jnp.arange(value.shape[2]), axis=0) 
        value_relative_positions -= jnp.expand_dims(jnp.arange(value.shape[1]), axis=1)
        value_relative_positions = jnp.where(value_relative_positions < clip, value_relative_positions, clip)
        value_relative_positions = jnp.where(value_relative_positions > -clip, value_relative_positions, -clip)
        value += value_relative_positions
        context_vectors, attention = self.attention_function(query,key, value, mask=mask)
        outputs = self.output(context_vectors)
        return outputs, attention
    
    def attention_function(self, query, key, value, mask=None):
        input_length = query.shape[1]
        context_length = key.shape[1]
        head_dim = query.shape[-1] // self.num_heads
        dim_key = key.shape[-1]

        # Split queries, keys, and values into heads
        query_heads = jnp.reshape(query, (query.shape[0], self.num_heads, input_length, head_dim))
        key_heads = jnp.reshape(key, (key.shape[0], self.num_heads, context_length, head_dim))
        value_heads = jnp.reshape(value, (value.shape[0], self.num_heads, context_length, head_dim))

        attention_scores = jnp.matmul(query_heads, key_heads.transpose(0, 1, 3, 2)) / jnp.sqrt(dim_key)
        if mask is not None:
            attention_scores = attention_scores * mask

        attention_weights = jax.nn.softmax(attention_scores, axis=-1)
        attended_values = jnp.matmul(attention_weights, value_heads)
        attended_values = jnp.reshape(attended_values, (query.shape[0], input_length, query.shape[-1]))
        return attended_values, attention_weights
    


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
    

class RotaryMultiHeadAttention(nn.Module):
    """
    Attention which uses RoPE (Rotary Positional Encoding)
    """
    hidden_dim : int  # Output dimension
    num_heads : int  # Number of parallel heads

    def setup(self):
        # Because the Query is determined from a context, project separately
        self.query_projection = nn.Dense(self.hidden_dim,
                                 kernel_init=nn.initializers.xavier_uniform(),
                                 bias_init=nn.initializers.zeros 
                                )
        self.key_projection = nn.Dense(self.hidden_dim,
                                 kernel_init=nn.initializers.xavier_uniform(),
                                 bias_init=nn.initializers.zeros 
                                )
        self.value_projection = nn.Dense(self.hidden_dim,
                                 kernel_init=nn.initializers.xavier_uniform(),
                                 bias_init=nn.initializers.zeros 
                                )
        self.rope = RotaryPositionalEncoding(self.hidden_dim)
        self.output = nn.Dense(self.hidden_dim,
                               kernel_init=nn.initializers.xavier_uniform(),
                               bias_init=nn.initializers.zeros)


    def __call__(self, 
                 inputs: jnp.ndarray, 
                 context: jnp.ndarray, 
                 mask: jnp.ndarray = None) -> tuple:

        """
        Args:
            inputs: inputs ((batch_size, seq_len, dims))
            context: optional - context ((batch_size, seq_len, dims))
            Mask: optional - masks where reqions to ignore are flipped to os
                  regions to attend to are 1s (batch_size, seq_len, dims)

        Return: outputs (batch_size, seq_len, seq_len)
                attention matrixes (batch_size, heads, seq_len, seq_len)
        """

        query = self.query_projection(inputs)
        key = self.key_projection(context)
        value = self.value_projection(context)
        query, key = self.rope(query, key) # Encode query and key with RoPE
        context_vectors, attention = self.attention_function(query,key, value, mask=mask)
        outputs = self.output(context_vectors)
        return outputs, attention
    
    def attention_function(self, query, key, value, mask=None):
        input_length = query.shape[1]
        context_length = key.shape[1]
        head_dim = query.shape[-1] // self.num_heads
        dim_key = key.shape[-1]

        # Split queries, keys, and values into heads
        query_heads = jnp.reshape(query, (query.shape[0], self.num_heads, input_length, head_dim))
        key_heads = jnp.reshape(key, (key.shape[0], self.num_heads, context_length, head_dim))
        value_heads = jnp.reshape(value, (value.shape[0], self.num_heads, context_length, head_dim))

        attention_scores = jnp.matmul(query_heads, key_heads.transpose(0, 1, 3, 2)) / jnp.sqrt(dim_key)
        if mask is not None:
            attention_scores = attention_scores * mask

        attention_weights = jax.nn.softmax(attention_scores, axis=-1)
        attended_values = jnp.matmul(attention_weights, value_heads)
        attended_values = jnp.reshape(attended_values, (query.shape[0], input_length, query.shape[-1]))
        return attended_values, attention_weights


class GatedMultiHeadAttention(nn.Module):
    """
    https://arxiv.org/abs/1912.00349
    This involves transforming the input by weighting features by importance relative to a context
    Note: The discrete nature of the gate creates a differentiability challenge during backpropagation
         Please read the paper to see how to use Gumbel-Softmax approximation to mitigate this before training

    """
    hidden_dim : int  # Output dimension
    num_heads : int  # Number of parallel heads

    def setup(self):
        # Because the Query is determined from a context, project separately
        self.query_projection = nn.Dense(self.hidden_dim,
                                 kernel_init=nn.initializers.xavier_uniform(),
                                 bias_init=nn.initializers.zeros 
                                )
        self.key_projection = nn.Dense(self.hidden_dim,
                                 kernel_init=nn.initializers.xavier_uniform(),
                                 bias_init=nn.initializers.zeros 
                                )
        self.value_projection = nn.Dense(self.hidden_dim,
                                 kernel_init=nn.initializers.xavier_uniform(),
                                 bias_init=nn.initializers.zeros 
                                )
        self.output = nn.Dense(self.hidden_dim,
                               kernel_init=nn.initializers.xavier_uniform(),
                               bias_init=nn.initializers.zeros
                               )
        self.gate = nn.Dense(features=1)


    def __call__(self, 
                 inputs: jnp.ndarray, 
                 context: jnp.ndarray, 
                 mask: jnp.ndarray = None) -> tuple:

        """
        Args:
            inputs: inputs ((batch_size, seq_len, dims))
            context: optional - context ((batch_size, seq_len, dims))
            Mask: optional - masks where reqions to ignore are flipped to os
                  regions to attend to are 1s (batch_size, seq_len, dims)

        Return: outputs (batch_size, seq_len, seq_len)
                attention matrixes (batch_size, heads, seq_len, seq_len)
        """
        query = self.query_projection(inputs)
        key = self.key_projection(context)
        value = self.value_projection(context)
        context_vectors, attention = self.attention_function(query,key,value,mask=mask)
        outputs = self.output(context_vectors)
        return outputs, attention
    
    def attention_function(self, query, key, value,mask=None):
        input_length = query.shape[1]
        context_length = key.shape[1]
        head_dim = query.shape[-1] // self.num_heads
        dim_key = key.shape[-1]

        # Split queries, keys, and values into heads
        query_heads = jnp.reshape(query, (query.shape[0], self.num_heads, input_length, head_dim))
        key_heads = jnp.reshape(key, (key.shape[0], self.num_heads, context_length, head_dim))
        value_heads = jnp.reshape(value, (value.shape[0], self.num_heads, context_length, head_dim))

        probabilities = jax.nn.sigmoid(self.gate(value_heads))
        booleans = jax.random.bernoulli(jax.random.PRNGKey(0), probabilities)
        gate = jnp.where(booleans, 1.0, 0.0)

        attention_scores = jnp.matmul(query_heads, key_heads.transpose(0, 1, 3, 2)) / jnp.sqrt(dim_key)
        attention_scores * gate

        if mask is not None:
            attention_scores = attention_scores * mask

        attention_weights = jax.nn.softmax(attention_scores, axis=-1)
        attended_values = jnp.matmul(attention_weights, value_heads)
        attended_values = jnp.reshape(attended_values, (query.shape[0], input_length, query.shape[-1]))
        return attended_values, attention_weights
    

class HierarchicalMultiHeadAttention(nn.Module):
    """
    https://www.cs.cmu.edu/~./hovy/papers/16HLT-hierarchical-attention-networks.pdf 
    The hierarchical attention network consists of two main parts: a word attention layer and a sentence attention layer. 
    The word attention layer learns to attend to the most important words in a sentence, 
    and the sentence attention layer learns to attend to the most important sentences in a document.

    Note: This is computationally intensive, many works have proposed techniques to avert this. 
          One of such methods involves projecting the inputs to low dimensions, 
          A Jax implementation of PCA for dimensionality reduction can be found in the core.ml.PCA(),
          One could project the inputs in each batch before passing to this module.
    """
    hidden_dim : int  # Output dimension
    num_heads : int  # Number of parallel heads

    def setup(self):
        # Because the Query is determined from a context, project separately
        self.word_query_projection = nn.Dense(self.hidden_dim,
                                 kernel_init=nn.initializers.xavier_uniform(),
                                 bias_init=nn.initializers.zeros 
                                )
        self.word_key_projection = nn.Dense(self.hidden_dim,
                                 kernel_init=nn.initializers.xavier_uniform(),
                                 bias_init=nn.initializers.zeros 
                                )
        self.word_value_projection = nn.Dense(self.hidden_dim,
                                 kernel_init=nn.initializers.xavier_uniform(),
                                 bias_init=nn.initializers.zeros 
                                )
        self.word_output = nn.Dense(self.hidden_dim,
                               kernel_init=nn.initializers.xavier_uniform(),
                               bias_init=nn.initializers.zeros
                               )
        self.sentence_query_projection = nn.Dense(self.hidden_dim,
                                 kernel_init=nn.initializers.xavier_uniform(),
                                 bias_init=nn.initializers.zeros 
                                )
        self.sentence_key_projection = nn.Dense(self.hidden_dim,
                                 kernel_init=nn.initializers.xavier_uniform(),
                                 bias_init=nn.initializers.zeros 
                                )
        self.sentence_value_projection = nn.Dense(self.hidden_dim,
                                 kernel_init=nn.initializers.xavier_uniform(),
                                 bias_init=nn.initializers.zeros 
                                )
        self.sentence_output = nn.Dense(self.hidden_dim,
                               kernel_init=nn.initializers.xavier_uniform(),
                               bias_init=nn.initializers.zeros)


    def __call__(self,
                 word_inputs: jnp.ndarray,
                 word_context: jnp.ndarray,
                 sentence_inputs: jnp.ndarray,
                 sentence_context: jnp.ndarray,
                 word_mask: jnp.ndarray = None,
                 sentence_mask: jnp.ndarray = None) -> tuple:

        """
        Args:
            word_inputs: inputs ((batch_size, seq_len, dims)), list of word embeddings in sentence
            word_context: optional - context ((batch_size, seq_len, dims))
            word_Mask: optional - masks where reqions to ignore are flipped to os
                  regions to attend to are 1s (batch_size, seq_len, dims)

            sentence_inputs: inputs ((batch_size, seq_len, dims))
            sentence_context: optional - context ((batch_size, seq_len, dims))
            sentence_Mask: optional - masks where reqions to ignore are flipped to os
                  regions to attend to are 1s (batch_size, seq_len, dims)

        Return: outputs (batch_size, seq_len, seq_len), list of sentence embedding in a conversation (think average pool)
                attention matrixes (batch_size, heads, seq_len, seq_len)

        Note: The word-level and sentence-level can be concatenated or added, then forwarded to a single output layer
        """
        word_queries = self.word_query_projection(word_inputs)
        word_keys = self.word_key_projection(word_context)
        word_values = self.word_value_projection(word_context)
        word_attention, word_context_vectors = self.attention_function(word_queries,
                                                                       word_keys,
                                                                       word_values,
                                                                       mask=word_mask)
        
        sentence_queries = self.sentence_query_projection(sentence_inputs)
        sentence_keys = self.sentence_key_projection(sentence_context)
        sentence_values = self.sentence_value_projection(sentence_context)
        sentence_attention, sentence_context_vectors = self.attention_function(sentence_queries,
                                                                               sentence_keys,
                                                                               sentence_values,
                                                                               mask=sentence_mask)
        word_outputs = self.word_output(word_context_vectors)
        sentence_outputs = self.sentence_output(sentence_context_vectors)
        return word_outputs, sentence_outputs, word_attention, sentence_attention
    
    def attention_function(self, query, key, value, mask=None):
        input_length = query.shape[1]
        context_length = key.shape[1]
        head_dim = query.shape[-1] // self.num_heads
        dim_key = key.shape[-1]

        # Split queries, keys, and values into heads
        query_heads = jnp.reshape(query, (query.shape[0], self.num_heads, input_length, head_dim))
        key_heads = jnp.reshape(key, (key.shape[0], self.num_heads, context_length, head_dim))
        value_heads = jnp.reshape(value, (value.shape[0], self.num_heads, context_length, head_dim))

        attention_scores = jnp.matmul(query_heads, key_heads.transpose(0, 1, 3, 2)) / jnp.sqrt(dim_key)
        if mask is not None:
            attention_scores = attention_scores * mask

        attention_weights = jax.nn.softmax(attention_scores, axis=-1)
        attended_values = jnp.matmul(attention_weights, value_heads)
        attended_values = jnp.reshape(attended_values, (query.shape[0], input_length, query.shape[-1]))
        return attended_values, attention_weights



class LocalMultiHeadAttention(nn.Module):
    """
    https://arxiv.org/abs/1706.03762 (Vaswani et. al. 2017)
    This involves transforming the input by weighting features by importance relative to a context
    """
    hidden_dim : int  # Output dimension
    num_heads : int  # Number of parallel heads
    window_size : int = 3

    def setup(self):
        # Because the Query is determined from a context, project separately
        self.query_projection = nn.Dense(self.hidden_dim,
                                 kernel_init=nn.initializers.xavier_uniform(),
                                 bias_init=nn.initializers.zeros 
                                )
        self.key_projection = nn.Dense(self.hidden_dim,
                                 kernel_init=nn.initializers.xavier_uniform(),
                                 bias_init=nn.initializers.zeros 
                                )
        self.value_projection = nn.Dense(self.hidden_dim,
                                 kernel_init=nn.initializers.xavier_uniform(),
                                 bias_init=nn.initializers.zeros 
                                )
        self.output = nn.Dense(self.hidden_dim,
                               kernel_init=nn.initializers.xavier_uniform(),
                               bias_init=nn.initializers.zeros)


    def __call__(self, 
                 inputs: jnp.ndarray, 
                 context: jnp.ndarray) -> tuple:

        """
        Args:
            inputs: inputs ((batch_size, seq_len, dims))
            context: optional - context ((batch_size, seq_len, dims))
            Mask: optional - masks where reqions to ignore are flipped to os
                  regions to attend to are 1s (batch_size, seq_len, dims)

        Return: outputs (batch_size, seq_len, seq_len)
                attention matrixes (batch_size, heads, seq_len, seq_len)
        """
        query = self.query_projection(inputs)
        key = self.key_projection(context)
        value = self.value_projection(context)

        local_mask = self.create_local_attention_mask(query.shape[1], key.shape[1])

        context_vectors, attention = self.attention_function(query,key,value,mask=local_mask)
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
        input_length = query.shape[1]
        context_length = key.shape[1]
        head_dim = query.shape[-1] // self.num_heads
        dim_key = key.shape[-1]

        # Split queries, keys, and values into heads
        query_heads = jnp.reshape(query, (query.shape[0], self.num_heads, input_length, head_dim))
        key_heads = jnp.reshape(key, (key.shape[0], self.num_heads, context_length, head_dim))
        value_heads = jnp.reshape(value, (value.shape[0], self.num_heads, context_length, head_dim))

        attention_scores = jnp.matmul(query_heads, key_heads.transpose(0, 1, 3, 2)) / jnp.sqrt(dim_key)
        if mask is not None:
            attention_scores = attention_scores * mask

        attention_weights = jax.nn.softmax(attention_scores, axis=-1)
        attended_values = jnp.matmul(attention_weights, value_heads)
        attended_values = jnp.reshape(attended_values, (query.shape[0], input_length, query.shape[-1]))
        return attended_values, attention_weights