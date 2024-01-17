'''
Transformers are a groundbreaking class of deep learning models originally introduced in the paper "Attention Is All You Need" by Vaswani et al. 
Their motivation stems from addressing limitations in previous sequence-to-sequence models and enabling more efficient and parallelizable training. 
The key innovation of transformers is the self-attention mechanism, which allows the model to weigh the importance of different parts of the input sequence during processing. 
This architecture has had a profound impact on natural language processing and has been adapted for a wide range of tasks, including machine translation, text generation, image captioning, and more. 
Transformers have become the foundation for various state-of-the-art models, including BERT, GPT, and T5, which have achieved remarkable results across multiple domains, showcasing the power of attention-based architectures in deep learning.
'''

import jax
import time
import optax
import pickle
import jax.numpy as jnp
import flax.linen as nn
from flax.training import train_state
from typing import List, Tuple, Any, Optional, Dict, Iterable

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
    


class PositionWiseFFN(nn.Module):
    """
    Position-wise Feed-Forward Network.

    Args:
        num_hiddens (int): Number of hidden units in the feed-forward layers.
        num_outputs (int): Number of output units in the feed-forward layers.
    """
    num_hiddens: int
    num_outputs: int

    def setup(self):
        self.dense1 = nn.Dense(self.num_hiddens, kernel_init=nn.initializers.xavier_uniform())
        self.dense2 = nn.Dense(self.num_outputs, kernel_init=nn.initializers.xavier_uniform())

    def __call__(self, X: jnp.ndarray) -> jnp.ndarray:
        """
        Apply the PositionWiseFFN to input data.

        Args:
            X (jnp.ndarray): Input tensor.

        Returns:
            jnp.ndarray: Output tensor after applying the feed-forward network.
        """
        return self.dense2(nn.gelu(self.dense1(X)))

class AddNorm(nn.Module):
    """
    Residual connection followed by layer normalization.

    Args:
        dropout (float): Dropout rate for the residual connection.
    """
    dropout: int

    @nn.compact
    def __call__(self, 
                 X: jnp.ndarray, 
                 Y: jnp.ndarray, 
                 training=False) -> jnp.ndarray:
        """
        Apply AddNorm to input tensors.

        Args:
            X (jnp.ndarray): Input tensor X.
            Y (jnp.ndarray): Input tensor Y.
            training (bool): Training mode.

        Returns:
            jnp.ndarray: Output tensor after applying AddNorm.
        """
        return nn.LayerNorm()(
            nn.Dropout(self.dropout)(Y, deterministic=not training) + X)



class EncoderBlock(nn.Module):
    """
    Transformer Encoder Block.

    Args:
        input_dim (int): Input dimension.
        num_heads (int): Number of attention heads.
        feedforward_dim (int): Dimension of the feed-forward network.
        dropout (float): Dropout rate.
    """
    input_dim: int
    num_heads: int
    feedforward_dim: int
    dropout: float

    def setup(self):
        self.attention = SelfMultiHeadAttention(hidden_dim=self.input_dim, 
                                                num_heads=self.num_heads)
        self.ff = PositionWiseFFN(self.feedforward_dim, self.input_dim)
        self.add_norm1 = AddNorm(self.dropout)
        self.add_norm2 = AddNorm(self.dropout)

    def __call__(self, 
                 x: jnp.ndarray, 
                 mask: jnp.ndarray = None, 
                 training: bool = True) -> tuple:
        """
        Apply the EncoderBlock to input data.

        Args:
            x (jnp.ndarray): Input tensor.
            mask (jnp.ndarray, optional): Mask tensor. Defaults to None.
            training (bool): Training mode.

        Returns:
            tuple: Output tensor and attention tensor.
        """
        attended_x, attention = self.attention(x, mask=mask)
        x = self.add_norm1(x, attended_x, training)
        ff_output = self.ff(x)
        x = self.add_norm2(x, ff_output, training)
        return x, attention


class TransformerEncoder(nn.Module):
    """
    Transformer Encoder.

    Args:
        num_layers (int): Number of encoder layers.
        input_dim (int): Input dimension.
        num_heads (int): Number of attention heads.
        feedforward_dim (int): Dimension of the feed-forward network.
        dropout (float): Dropout rate.
    """
    num_layers: int
    input_dim: int
    num_heads: int
    feedforward_dim: int
    dropout: float
    max_len : int
    vocab_size : int
    embed_dim : int
    learned_position : bool = True


    def setup(self):
        self.embedding = TokenAndPositionEmbedding(self.max_len,
                                                   self.vocab_size,
                                                   self.embed_dim,
                                                   self.learned_position)
        self.layers = [EncoderBlock(self.input_dim, 
                                    self.num_heads, 
                                    self.feedforward_dim, 
                                    self.dropout)
                       for _ in range(self.num_layers)]

    def __call__(self, 
                 x: jnp.ndarray, 
                 mask: jnp.ndarray = None, 
                 training: bool = True) -> tuple:
        """
        Apply the TransformerEncoder to input data.

        Args:
            x (jnp.ndarray): Input tensor.
            mask (jnp.ndarray, optional): Mask tensor. Defaults to None.
            training (bool): Training mode.

        Returns:
            tuple: Output tensor and list of attention tensors.
            each attention map has dim (num_layers, batch_size, num_heads, seq_length, seq_length)
        """
        attention_maps = []
        x = self.embedding(x)
        for layer in self.layers:
            x, attention = layer(x, mask=mask, training=training)
            attention_maps.append(attention)
        return x, jnp.array(attention_maps)