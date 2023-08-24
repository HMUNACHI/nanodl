'''
Implementations of various attention mechanisms variants.
While multiple concepts can be merged into a generalised module,
Each technique is isolated for clarity and easy copy and paste **wink**
'''

import jax.numpy as jnp
import flax.linen as nn
from jax.nn import softmax


class MultiHeadSelfAttention(nn.Module):
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


    def __call__(self, inputs, mask=None):

        """
        Args:
            context: optional - context ((batch_size, seq_len, dims))
            Mask: optional - masks where reqions to ignore are flipped to os
                  regions to attend to are 1s (batch_size, seq_len, dims)

        Return: outputs (batch_size, seq_len, seq_len)
                attention matrixes (batch_size, heads, seq_len, seq_len)
        """

        batch_size, seq_length, embed_dim = inputs.shape
        
        if mask is not None:
            mask = self.expand_mask(mask)

        projections = self.projection(inputs)
        projections = projections.reshape(batch_size, seq_length, self.num_heads, -1)
        projections = projections.transpose(0, 2, 1, 3) # [Batch, Head, SeqLen, Dims]
        query, key, value = jnp.array_split(projections, 3, axis=-1)

        outputs, attention = self.scaled_dot_product_attention(query,key,value,mask=mask)
        outputs = outputs.transpose(0, 2, 1, 3) # [Batch, SeqLen, Head, Dims]
        outputs = outputs.reshape(batch_size, seq_length, embed_dim)
        outputs = self.output(outputs)
        return outputs, attention
    

    def scaled_dot_product_attention(self, query, key, value, mask=None):
        """
        Args:
            Query: projections of the query (batch, n_sequence, dim)
            Key: projections of the key (batch, n_sequence, dim)
            Value: projections of the value (batch, n_sequence, dim)
            Mask: masks where reqions to ignore are flipped to os
                regions to attend to are 1s (batch, n_sequence, dim)

        Return: softmax((query * key.T) / √dim_key) * value
        """

        dim_key = key.shape[-1]
        key_transposed = jnp.swapaxes(key, -2, -1)
        attention_scores = jnp.matmul(query, key_transposed) / jnp.sqrt(dim_key)

        if mask is not None:
            attention_scores = jnp.where(mask == 0, -9e13, attention_scores)

        attention_weights = softmax(attention_scores, axis=-1)
        attented_outputs = jnp.matmul(attention_weights, value)
        return attented_outputs, attention_weights

    def expand_mask(self, mask):
        assert mask.ndim > 2, "Mask must be at least 2-dimensional with seq_length x seq_length"
        if mask.ndim == 3:
            mask = jnp.expand_dims(mask, 1)
        while mask.ndim < 4:
            mask = jnp.expand_dims(mask, 0)
        return mask




class MultiHeadCrossAttention(nn.Module):
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


    def __call__(self, inputs, context, mask=None):

        """
        Args:
            inputs: inputs ((batch_size, seq_len, dims))
            context: optional - context ((batch_size, seq_len, dims))
            Mask: optional - masks where reqions to ignore are flipped to os
                  regions to attend to are 1s (batch_size, seq_len, dims)

        Return: outputs (batch_size, seq_len, seq_len)
                attention matrixes (batch_size, heads, seq_len, seq_len)
        """

        batch_size, seq_length, embed_dim = inputs.shape

        if mask is not None:
            mask = self.expand_mask(mask)

        query = self.query_projection(inputs)
        key = self.key_projection(context)
        value = self.value_projection(context)

        query = query.reshape(batch_size, seq_length, self.num_heads, -1)
        key = key.reshape(batch_size, seq_length, self.num_heads, -1)
        value = value.reshape(batch_size, seq_length, self.num_heads, -1)

        context_vectors, attention = self.scaled_dot_product_attention(query,key,value,mask=mask)
        context_vectors = context_vectors.transpose(0, 2, 1, 3) # [Batch, SeqLen, Head, Dims]
        context_vectors = context_vectors.reshape(batch_size, seq_length, embed_dim)
        outputs = self.output(context_vectors)
        return outputs, attention
    

    def scaled_dot_product_attention(self, query, key, value, mask=None):
        """
        Args:
            Query: projections of the query (batch, n_sequence, dim)
            Key: projections of the key (batch, n_sequence, dim)
            Value: projections of the value (batch, n_sequence, dim)
            Mask: masks where reqions to ignore are flipped to os
                regions to attend to are 1s (batch, n_sequence, dim)

        Return: softmax((query * key.T) / √dim_key) * value
        """

        dim_key = key.shape[-1]
        key_transposed = jnp.swapaxes(key, -2, -1)
        attention_scores = jnp.matmul(query, key_transposed) / jnp.sqrt(dim_key)

        if mask is not None:
            attention_scores = jnp.where(mask == 0, -9e13, attention_scores)

        attention_weights = softmax(attention_scores, axis=-1)
        context_vector = jnp.matmul(attention_weights, value)
        return context_vector, attention_weights

    def expand_mask(self, mask):
        assert mask.ndim > 2, "Mask must be at least 2-dimensional with seq_length x seq_length"
        if mask.ndim == 3:
            mask = jnp.expand_dims(mask, 1)
        while mask.ndim < 4:
            mask = jnp.expand_dims(mask, 0)
        return mask



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


    def __call__(self, inputs, context, mask=None):

        """
        Args:
            inputs: inputs ((batch_size, seq_len, dims))
            context: optional - context ((batch_size, seq_len, dims))
            Mask: optional - masks where reqions to ignore are flipped to os
                  regions to attend to are 1s (batch_size, seq_len, dims)

        Return: outputs (batch_size, seq_len, seq_len)
                attention matrixes (batch_size, heads, seq_len, seq_len)
        """

        batch_size, seq_length, embed_dim = inputs.shape

        if mask is not None:
            mask = self.expand_mask(mask)

        query = self.query_projection(inputs)
        key = self.key_projection(context)
        value = self.value_projection(context)

        key = jnp.repeat(key, self.num_heads, axis=-1)
        value = jnp.repeat(value, self.num_heads, axis=-1)

        query = query.reshape(batch_size, seq_length, self.num_heads, -1)
        key = key.reshape(batch_size, seq_length, self.num_heads, -1)
        value = value.reshape(batch_size, seq_length, self.num_heads, -1)

        context_vectors, attention = self.scaled_dot_product_attention(query,key,value,mask=mask)
        context_vectors = context_vectors.transpose(0, 2, 1, 3) # [Batch, SeqLen, Head, Dims]
        context_vectors = context_vectors.reshape(batch_size, seq_length, embed_dim)
        outputs = self.output(context_vectors)
        return outputs, attention
    

    def scaled_dot_product_attention(self, query, key, value, mask=None):
        """
        Args:
            Query: projections of the query (batch, n_sequence, dim)
            Key: projections of the key (batch, n_sequence, dim)
            Value: projections of the value (batch, n_sequence, dim)
            Mask: masks where reqions to ignore are flipped to os
                regions to attend to are 1s (batch, n_sequence, dim)

        Return: softmax((query * key.T) / √dim_key) * value
        """

        dim_key = key.shape[-1]
        key_transposed = jnp.swapaxes(key, -2, -1)
        attention_scores = jnp.matmul(query, key_transposed) / jnp.sqrt(dim_key)

        if mask is not None:
            attention_scores = jnp.where(mask == 0, -9e13, attention_scores)

        attention_weights = softmax(attention_scores, axis=-1)
        context_vector = jnp.matmul(attention_weights, value)
        return context_vector, attention_weights

    def expand_mask(self, mask):
        assert mask.ndim > 2, "Mask must be at least 2-dimensional with seq_length x seq_length"
        if mask.ndim == 3:
            mask = jnp.expand_dims(mask, 1)
        while mask.ndim < 4:
            mask = jnp.expand_dims(mask, 0)
        return mask



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


class MultiHeadRotaryAttention(nn.Module):
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


    def __call__(self, inputs, context, mask=None):

        """
        Args:
            inputs: inputs ((batch_size, seq_len, dims))
            context: optional - context ((batch_size, seq_len, dims))
            Mask: optional - masks where reqions to ignore are flipped to os
                  regions to attend to are 1s (batch_size, seq_len, dims)

        Return: outputs (batch_size, seq_len, seq_len)
                attention matrixes (batch_size, heads, seq_len, seq_len)
        """

        batch_size, seq_length, embed_dim = inputs.shape

        if mask is not None:
            mask = self.expand_mask(mask)

        query = self.query_projection(inputs)
        key = self.key_projection(context)
        value = self.value_projection(context)

        query, key = self.rope(query, key) # Encode query and key with RoPE

        query = query.reshape(batch_size, seq_length, self.num_heads, -1)
        key = key.reshape(batch_size, seq_length, self.num_heads, -1)
        value = value.reshape(batch_size, seq_length, self.num_heads, -1)

        context_vectors, attention = self.scaled_dot_product_attention(query,key,value,mask=mask)
        context_vectors = context_vectors.transpose(0, 2, 1, 3) # [Batch, SeqLen, Head, Dims]
        context_vectors = context_vectors.reshape(batch_size, seq_length, embed_dim)
        outputs = self.output(context_vectors)
        return outputs, attention
    

    def scaled_dot_product_attention(self, query, key, value, mask=None):
        """
        Args:
            Query: projections of the query (batch, n_sequence, dim)
            Key: projections of the key (batch, n_sequence, dim)
            Value: projections of the value (batch, n_sequence, dim)
            Mask: masks where reqions to ignore are flipped to os
                regions to attend to are 1s (batch, n_sequence, dim)

        Return: softmax((query * key.T) / √dim_key) * value
        """

        dim_key = key.shape[-1]
        key_transposed = jnp.swapaxes(key, -2, -1)
        attention_scores = jnp.matmul(query, key_transposed) / jnp.sqrt(dim_key)

        if mask is not None:
            attention_scores = jnp.where(mask == 0, -9e13, attention_scores)

        attention_weights = softmax(attention_scores, axis=-1)
        context_vector = jnp.matmul(attention_weights, value)
        return context_vector, attention_weights

    def expand_mask(self, mask):
        assert mask.ndim > 2, "Mask must be at least 2-dimensional with seq_length x seq_length"
        if mask.ndim == 3:
            mask = jnp.expand_dims(mask, 1)
        while mask.ndim < 4:
            mask = jnp.expand_dims(mask, 0)
        return mask




class MultiHeadRelativeAttention(nn.Module):
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


    def __call__(self, inputs, context, mask=None):

        """
        Args:
            inputs: inputs ((batch_size, seq_len, dims))
            context: optional - context ((batch_size, seq_len, dims))
            Mask: optional - masks where reqions to ignore are flipped to os
                  regions to attend to are 1s (batch_size, seq_len, dims)

        Return: outputs (batch_size, seq_len, seq_len)
                attention matrixes (batch_size, heads, seq_len, seq_len)
        """

        batch_size, seq_length, embed_dim = inputs.shape

        if mask is not None:
            mask = self.expand_mask(mask)

        query = self.query_projection(inputs)
        key = self.key_projection(context)
        value = self.value_projection(context)

        query = query.reshape(batch_size, seq_length, self.num_heads, -1)
        key = key.reshape(batch_size, seq_length, self.num_heads, -1)
        value = value.reshape(batch_size, seq_length, self.num_heads, -1)

        context_vectors, attention = self.relative_attention(query,key,value,mask=mask)
        context_vectors = context_vectors.transpose(0, 2, 1, 3) # [Batch, SeqLen, Head, Dims]
        context_vectors = context_vectors.reshape(batch_size, seq_length, embed_dim)
        outputs = self.output(context_vectors)
        return outputs, attention
    

    def relative_attention(self, query, key, value, mask=None):
        """
        Computes relative attention between queries and keys.

        Args:
            queries: A tensor of shape [batch_size, query_length, hidden_size].
            keys: A tensor of shape [batch_size, key_length, hidden_size].
            values: A tensor of shape [batch_size, key_length, hidden_size].
            mask: A tensor of shape [batch_size, query_length, key_length], dtype=bool.

        Returns:
            A tensor of shape [batch_size, query_length, hidden_size].
        """
        relative_position_encoding = jnp.arange(0, query.shape[1], dtype=jnp.int32)[:, None]
        relative_position_encoding  -= jnp.arange(0, key.shape[1], dtype=jnp.int32)
        relative_position_encoding = jnp.expand_dims(relative_position_encoding, axis=2)

        dim_key = key.shape[-1]
        key_transposed = jnp.swapaxes(key, -2, -1)
        attention_scores = jnp.matmul(query, key_transposed) / jnp.sqrt(dim_key)
        attention_scores += relative_position_encoding

        if mask is not None:
            attention_scores = jnp.where(mask == 0, -9e13, attention_scores)

        attention_weights = softmax(attention_scores, axis=-1)
        context_vector = jnp.matmul(attention_weights, value)
        return context_vector, attention_weights

    def expand_mask(self, mask):
        assert mask.ndim > 2, "Mask must be at least 2-dimensional with seq_length x seq_length"
        if mask.ndim == 3:
            mask = jnp.expand_dims(mask, 1)
        while mask.ndim < 4:
            mask = jnp.expand_dims(mask, 0)
        return mask


# MHA test
from jax import random

key = random.PRNGKey(0)
main_rng, x_rng = random.split(key)
x = random.normal(x_rng, (3, 16, 128))
mh_attn = MultiHeadRotaryAttention(hidden_dim=128, num_heads=4)
main_rng, init_rng = random.split(main_rng)
params = mh_attn.init(init_rng, x, x)['params']
# Apply attention with parameters on the inputs
out, attn = mh_attn.apply({'params': params}, x, x)
print('Out', out.shape, 'Attention', attn.shape)

# import jax
# def relative_attention(queries, keys, values, mask=None):
#     """
#     Computes relative attention between queries and keys.

#     Args:
#         queries: A tensor of shape [batch_size, query_length, hidden_size].
#         keys: A tensor of shape [batch_size, key_length, hidden_size].
#         values: A tensor of shape [batch_size, key_length, hidden_size].
#         mask: A tensor of shape [batch_size, query_length, key_length], dtype=bool.

#     Returns:
#         A tensor of shape [batch_size, query_length, hidden_size].
#     """
#     batch_size, query_length, hidden_size = queries.shape
#     head_dim = hidden_size // 4

#     relative_position_encoding = jnp.arange(0, query_length, dtype=jnp.int32)
#     relative_position_encoding = relative_position_encoding[:, None] - jnp.arange(0, keys.shape[1], dtype=jnp.int32)
#     relative_position_encoding = jnp.expand_dims(relative_position_encoding, axis=2)

#     queries = queries.reshape((batch_size, query_length, 4, head_dim))
#     keys = keys.reshape((batch_size, keys.shape[1], 4, head_dim))
#     values = values.reshape((batch_size, keys.shape[1], 4, head_dim))

#     attention_scores = jnp.einsum("bqhd,bkhd->bqkh", queries, keys) + relative_position_encoding
#     print(attention_scores.shape, jnp.einsum("bqhd,bkhd->bqkh", queries, keys).shape, relative_position_encoding.shape)
#     attention_weights = jax.nn.softmax(attention_scores, axis=-1)

#     if mask is not None:
#         attention_weights = attention_weights * mask

#     context_vector = jnp.einsum("bqkh,bkhd->bqhd", attention_weights, values)
#     context_vector = context_vector.reshape((batch_size, query_length, hidden_size))
#     return context_vector, attention_weights


# x = jnp.ones((3,16,16))
# out, attn = relative_attention(x, x, x)
# print('Out', out.shape, 'Attention', attn.shape)


# import jax
# import jax.numpy as jnp
# from flax import linen as nn

# class RelativePosition(nn.Module):
#     num_units: int
#     max_relative_position: int

#     def setup(self):
#         self.embeddings_table = self.param('embeddings_table', 
#                                             lambda key, shape: jax.random.uniform(key, shape), 
#                                             (self.max_relative_position * 2 + 1, self.num_units))

#     def __call__(self, length_q, length_k):
#         range_vec_q = jnp.arange(length_q)
#         range_vec_k = jnp.arange(length_k)
#         distance_mat = range_vec_k[None, :] - range_vec_q[:, None]
#         distance_mat_clipped = jnp.clip(distance_mat, -self.max_relative_position, self.max_relative_position)
#         final_mat = distance_mat_clipped + self.max_relative_position
#         final_mat = jnp.array(final_mat, dtype=jnp.int32)
#         embeddings = self.embeddings_table[final_mat]

#         return embeddings

# class MultiHeadAttentionLayer(nn.Module):
#     hid_dim: int
#     n_heads: int

#     def setup(self):
#         assert self.hid_dim % self.n_heads == 0
#         self.head_dim = self.hid_dim // self.n_heads
#         self.max_relative_position = 2
#         self.relative_position_k = RelativePosition(self.head_dim, self.max_relative_position)
#         self.relative_position_v = RelativePosition(self.head_dim, self.max_relative_position)
#         self.fc_q = nn.Dense(self.hid_dim)
#         self.fc_k = nn.Dense(self.hid_dim)
#         self.fc_v = nn.Dense(self.hid_dim)
#         self.fc_o = nn.Dense(self.hid_dim)
#         self.scale = jnp.sqrt(jnp.array([self.head_dim], dtype=jnp.float32))

#     def __call__(self, query, key, value, mask=None):
#         batch_size = query.shape[0]
#         len_k = key.shape[1]
#         len_q = query.shape[1]
#         len_v = value.shape[1]

#         query = self.fc_q(query)
#         key = self.fc_k(key)
#         value = self.fc_v(value)

#         r_q1 = query.reshape(batch_size, -1, self.n_heads, self.head_dim).transpose((0, 2, 1, 3))
#         r_k1 = key.reshape(batch_size, -1, self.n_heads, self.head_dim).transpose((0, 2, 1, 3))
#         attn1 = jnp.matmul(r_q1, r_k1.transpose((0, 1, 3, 2)))

#         r_q2 = query.transpose((1, 0, 2)).reshape(len_q, batch_size*self.n_heads, self.head_dim)
#         r_k2 = self.relative_position_k(len_q, len_k)
#         attn2 = jnp.matmul(r_q2, r_k2.transpose((1, 2))).transpose((0, 1))
#         attn2 = attn2.reshape(batch_size, self.n_heads, len_q, len_k)
#         attn = (attn1 + attn2) / self.scale

#         if mask is not None:
#             attn = jax.ops.index_update(attn, mask == 0, -1e10)

#         attn = jax.nn.softmax(attn, axis=-1)

#         r_v1 = value.reshape(batch_size, -1, self.n_heads, self.head_dim).transpose((0, 2, 1, 3))
#         weight1 = jnp.matmul(attn, r_v1)
#         r_v2 = self.relative_position_v(len_q, len_v)
#         weight2 = attn.transpose((2, 0, 1, 3)).reshape(len_q, batch_size*self.n_heads, len_k)
#         weight2 = jnp.matmul(weight2, r_v2)
#         weight2 = weight2.transpose((0, 1)).reshape(batch_size, self.n_heads, len_q, self.head_dim)

#         x = weight1 + weight2
#         x = x.transpose((0, 2, 1, 3)).reshape(batch_size, -1, self.hid_dim)
#         x = self.fc_o(x)

#         return x