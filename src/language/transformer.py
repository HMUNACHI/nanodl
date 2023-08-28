'''
Implementations of various attention mechanisms variants.
While multiple concepts can be merged into a generalised module,
Each technique is isolated for clarity and easy copy and paste
'''

import jax.numpy as jnp
import flax.linen as nn
from jax.nn import softmax
from attention import SelfMultiHeadAttention, CrossMultiHeadAttention


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
    

class RMSAddNorm(nn.Module):  #@save
    """
    Residual connection followed by layer normalization.

    Args:
        dropout (float): Dropout rate for the residual connection.
    """
    dropout: int

    @nn.compact
    def __call__(self, x, y, training=False):
        return self.rms_norm(nn.Dropout(self.dropout)(y, deterministic=not training) + x)
    
    def rms_norm(self, x, axis=None, epsilon=1e-8):
        rms = jnp.sqrt(jnp.mean(jnp.square(x), axis=axis, keepdims=True) + epsilon)
        return x / rms


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
        self.attention = SelfMultiHeadAttention(hidden_dim=self.input_dim,
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
        attention_maps = []
        for layer in self.layers:
            x, attention = layer(x, mask=mask, training=training)
            attention_maps.append(attention)
        return x, attention_maps



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
        self.attention1 = SelfMultiHeadAttention(hidden_dim=self.input_dim,
                                                num_heads=self.num_heads
                                                )
        self.attention2 = CrossMultiHeadAttention(hidden_dim=self.input_dim,
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
        concatenator = jnp.concatenate([jnp.array([batch_size]), 
                                        jnp.array([self.num_heads]), 
                                        jnp.array([1, 1], dtype=jnp.int32)], 0)
        return jnp.tile(mask, concatenator)


    def __call__(self, x, context, mask=None, training=True):
        mask = self.causal_mask(x.shape[0], x.shape[1], context.shape[1])
        attended_x, attention1 = self.attention1(x, mask=mask)
        x = self.add_norm1(x, attended_x, training)
        attended_x, attention2 = self.attention2(x, context, mask=mask)
        x = self.add_norm1(x, attended_x, training)
        linear_output = self.linear1(x)
        x = self.add_norm1(x, linear_output, training)
        linear_output = self.linear2(x)
        x = softmax(linear_output, axis=-1)
        return x, jnp.array(attention1), jnp.array(attention2)


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
        attention_maps = []
        cross_attention_maps = []
        for layer in self.layers:
            x, attention, cross_attention = layer(x, context, training=training)
            attention_maps.append(attention)
            cross_attention_maps.append(cross_attention)
        return x, jnp.array(attention_maps), jnp.array(cross_attention_maps)
    

# Transformer test
from jax import random

key = random.PRNGKey(0)
main_rng, x_rng = random.split(key)
x = random.normal(x_rng, (3, 16, 128))
encblock = TransformerDecoder(input_dim=128, num_heads=4, num_layers=2, feedforward_dim=256, dropout=0.2)
# Initialize parameters of encoder block with random key and inputs
main_rng, init_rng, dropout_init_rng = random.split(main_rng, 3)
params = encblock.init({'params': init_rng, 'dropout': dropout_init_rng}, x, x, training=True)['params']
# Apply encoder block with parameters on the inputs
# Since dropout is stochastic, we need to pass a rng to the forward
main_rng, dropout_apply_rng = random.split(main_rng)
out, att1, att2 = encblock.apply({'params': params}, x, x, training=True, rngs={'dropout': dropout_apply_rng})
print('Out', out.shape, att1.shape, att2.shape)