'''
Implementations of various attention mechanisms variants.
While multiple concepts can be merged into a generalised module,
Each technique is isolated for clarity and easy copy and paste
'''

import math
import numpy as np
import jax.numpy as jnp
import flax.linen as nn
from jax.nn import softmax
from attention import *


class PositionWiseFFN(nn.Module):  #@save
    """
    Position-wise Feed-Forward Network.

    Args:
        num_hiddens (int): Number of hidden units in the feed-forward layers.
        num_outputs (int): Number of output units in the feed-forward layers.
    """
    num_hiddens: int
    num_outputs: int

    def setup(self):
        self.dense1 = nn.Dense(self.num_hiddens, 
                               kernel_init=nn.initializers.xavier_uniform())
        self.dense2 = nn.Dense(self.num_outputs,
                               kernel_init=nn.initializers.xavier_uniform())

    def __call__(self, X):
        return self.dense2(nn.relu(self.dense1(X)))


class AddNorm(nn.Module):  #@save
    """
    Residual connection followed by layer normalization.

    Args:
        dropout (float): Dropout rate for the residual connection.
    """
    dropout: int

    @nn.compact
    def __call__(self, X, Y, training=False):
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
    input_dim : int 
    num_heads : int
    feedforward_dim : int
    dropout : float

    def setup(self):
        # Attention layer
        self.attention = MultiHeadSelfAttention(hidden_dim=self.input_dim,
                                                num_heads=self.num_heads)
        self.linear = PositionWiseFFN(self.feedforward_dim, self.input_dim)
        self.add_norm1 =  AddNorm(self.dropout)
        self.add_norm2 =  AddNorm(self.dropout)

    def __call__(self, x, mask=None, training=True):
        attended_x, attention = self.attention(x, mask=mask)
        x = self.add_norm1(x, attended_x, training)
        linear_output = self.linear(x)
        x = self.add_norm1(x, linear_output, training)
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
    num_layers : int
    input_dim : int
    num_heads : int
    feedforward_dim : int
    dropout : float

    def setup(self):
        self.layers = [EncoderBlock(self.input_dim, 
                                    self.num_heads, 
                                    self.feedforward_dim,
                                    self.dropout) for _ in range(self.num_layers)]

    def __call__(self, x, mask=None, training=True):
        for layer in self.layers:
            x, _ = layer(x, mask=mask, training=training)
        return x

    def get_attention_maps(self, x, mask=None, training=False):
        attention_maps = []
        for layer in self.layers:
            x, attention = layer(x, mask=mask, training=training)
            attention_maps.append(attention)
        return attention_maps



class DecoderBlock(nn.Module):
    """
    Transformer Decoder Block.

    Args:
        input_dim (int): Input dimension.
        num_heads (int): Number of attention heads.
        feedforward_dim (int): Dimension of the feed-forward network.
        dropout (float): Dropout rate.
    """
    input_dim : int 
    num_heads : int
    feedforward_dim : int
    dropout : float

    def setup(self):
        self.attention1 = MultiHeadSelfAttention(hidden_dim=self.input_dim,
                                                num_heads=self.num_heads
                                                )
        self.attention2 = MultiHeadCrossAttention(hidden_dim=self.input_dim,
                                                num_heads=self.num_heads
                                                )
        self.linear1 = PositionWiseFFN(self.feedforward_dim, self.input_dim)
        self.linear2 = PositionWiseFFN(self.feedforward_dim, self.input_dim)
        self.add_norm1 =  AddNorm(self.dropout)
        self.add_norm2 =  AddNorm(self.dropout)
        self.add_norm3 =  AddNorm(self.dropout)


    def causal_mask(self, batch_size, destination_dim, source_dim):
        idx_source = jnp.arange(destination_dim)[:, None]
        idx_destination = jnp.arange(source_dim)
        mask = idx_source >= idx_destination - source_dim + destination_dim
        mask = mask.astype(jnp.int32) 
        mask = mask.reshape((1, destination_dim, source_dim))
        concatenator = jnp.concatenate([jnp.array([batch_size]), jnp.array([1, 1], dtype=jnp.int32)], 0)
        return jnp.tile(mask, concatenator)


    def __call__(self, x, context, mask=None, training=True):
        batch_size, sequence_length, _ = x.shape
        #mask = self.causal_mask(batch_size, sequence_length, sequence_length)
        attended_x, attention1 = self.attention1(x, mask=mask)
        x = self.add_norm1(x, attended_x, training)
        attended_x, attention2 = self.attention2(x, context, mask=mask)
        x = self.add_norm1(x, attended_x, training)
        linear_output = self.linear1(x)
        x = self.add_norm1(x, linear_output, training)
        linear_output = self.linear2(x)
        x = softmax(linear_output, axis=-1)
        return x, attention1, attention2


class TransformerDecoder(nn.Module):
    """
    Transformer Decoder.

    Args:
        num_layers (int): Number of decoder layers.
        input_dim (int): Input dimension.
        num_heads (int): Number of attention heads.
        feedforward_dim (int): Dimension of the feed-forward network.
        dropout (float): Dropout rate.
    """
    num_layers : int
    input_dim : int
    num_heads : int
    feedforward_dim : int
    dropout : float

    def setup(self):
        self.layers = [DecoderBlock(self.input_dim, 
                                    self.num_heads, 
                                    self.feedforward_dim,
                                    self.dropout) for _ in range(self.num_layers)]

    def __call__(self, x, context, mask=None, training=True):
        for layer in self.layers:
            x, _ , _ = layer(x, context, training=training)
        return x

    def get_attention_maps(self, x, context, training=False):
        attention_maps = []
        cross_attention_maps = []
        for layer in self.layers:
            x, attention, cross_attention = layer(x, context, training=training)
            attention_maps.append(attention)
            cross_attention_maps.append(cross_attention)
        return attention_maps, cross_attention_maps



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
        plt.imshow(positional_encoding.T)
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
        x = self.token_embeddings(inputs)
        if self.learned_position:
            return x + self.position_embeddings(jnp.arange(x.shape[1]))
        else:
            return x + self.position_embeddings(x)