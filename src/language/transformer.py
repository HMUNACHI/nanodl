'''
Implementations of various attention mechanisms variants.
While multiple concepts can be merged into a generalised module,
Each technique is isolated for clarity and easy copy and paste
'''
import jax.numpy as jnp
import flax.linen as nn
from jax.nn import softmax
from attention import *


class PositionWiseFFN(nn.Module):  #@save
    """The positionwise feed-forward network."""
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
    """The residual connection followed by layer normalization."""
    dropout: int

    @nn.compact
    def __call__(self, X, Y, training=False):
        return nn.LayerNorm()(
            nn.Dropout(self.dropout)(Y, deterministic=not training) + X)


class EncoderBlock(nn.Module):
    """
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


class TokenAndPositionEmbedding(nn.Module):
    """
    """
    max_len : int
    vocab_size : int
    embed_dim : int
    
    def setup(self):
        self.token_embeddings = nn.Embed(num_embeddings=self.vocab_size, features=self.embed_dim)
        self.position_embeddings = nn.Embed(num_embeddings=self.max_len, features=self.embed_dim)

    def __call__(self, inputs):
        tokens = self.token_embeddings(inputs)
        positions = jnp.arange(start=0, stop=inputs.shape[-1], step=1)
        positions = self.position_embeddings(positions)
        return tokens + positions


# Test transformer and decoder
from jax import random

key = random.PRNGKey(0)
main_rng, x_rng = random.split(key)
x = random.normal(x_rng, (3, 16, 128))
# Create encoder block
encblock = TransformerEncoder(input_dim=128, num_heads=4, num_layers=2, feedforward_dim=256, dropout=0.2)
# Initialize parameters of encoder block with random key and inputs
main_rng, init_rng, dropout_init_rng = random.split(main_rng, 3)
params = encblock.init({'params': init_rng, 'dropout': dropout_init_rng}, x, training=True)['params']
# Apply encoder block with parameters on the inputs
# Since dropout is stochastic, we need to pass a rng to the forward
main_rng, dropout_apply_rng = random.split(main_rng)
out = encblock.apply({'params': params}, x, training=True, rngs={'dropout': dropout_apply_rng})
print('Out', out.shape)