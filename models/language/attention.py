'''
Implementations of various attention mechanisms variants.
While multiple concepts can be merged into a generalised module,
Each technique is isolated for clarity and easy copy and paste
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

        projections = self.projection(x)
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
        "Mask must be at least 2-dimensional with seq_length x seq_length"
        assert mask.ndim > 2
        if mask.ndim == 3:
            mask = mask.unsqueeze(1)
        while mask.ndim < 4:
            mask = mask.unsqueeze(0)
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

        query = self.query_projection(context)
        key = self.key_projection(inputs)
        value = self.value_projection(inputs)

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
        key = self.key_projection(inputs)
        value = self.value_projection(inputs)

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