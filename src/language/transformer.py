'''
Implementations of various attention mechanisms variants.
While multiple concepts can be merged into a generalised module,
Each technique is isolated for clarity and easy copy and paste
'''
import jax.numpy as jnp
import flax.linen as nn
from jax.nn import softmax
from attention import *


class EncoderBlock(nn.Module):
    """
    """
    input_dim : int 
    num_heads : int

    def setup(self):
        # Attention layer
        self.attention = MultiHeadSelfAttention(hidden_dim=self.input_dim,
                                                num_heads=self.num_heads
                                                )
        self.linear = nn.Dense(self.input_dim,
                               kernel_init=nn.initializers.xavier_uniform(),
                               bias_init=nn.initializers.zeros
                               )
        self.norm1 = nn.LayerNorm()
        self.norm2 = nn.LayerNorm()

    def __call__(self, x, mask=None, train=True):
        attended_x, attention = self.attention(x, mask=mask)
        x = self.norm1(x + attended_x)
        linear_output = self.linear(x)
        x = self.norm2(x + linear_output)
        return x, attention


class TransformerEncoder(nn.Module):
    """
    """
    num_layers : int
    input_dim : int
    num_heads : int

    def setup(self):
        self.layers = [EncoderBlock(self.input_dim, self.num_heads) for _ in range(self.num_layers)]

    def __call__(self, x, mask=None, train=True):
        for layer in self.layers:
            x, _ = layer(x, mask=mask, train=train)
        return x

    def get_attention_maps(self, x, mask=None, train=True):
        attention_maps = []
        for layer in self.layers:
            x, attention = layer(x, mask=mask, train=train)
            attention_maps.append(attention)
        return attention_maps



class DecoderBlock(nn.Module):
    """
    """
    input_dim : int 
    num_heads : int

    def setup(self):
        self.attention1 = MultiHeadSelfAttention(hidden_dim=self.input_dim,
                                                num_heads=self.num_heads
                                                )
        self.attention2 = MultiHeadCrossAttention(hidden_dim=self.input_dim,
                                                num_heads=self.num_heads
                                                )
        self.linear1 = nn.Dense(self.input_dim,
                               kernel_init=nn.initializers.xavier_uniform(),
                               bias_init=nn.initializers.zeros
                               )
        self.linear2 = nn.Dense(self.input_dim,
                               kernel_init=nn.initializers.xavier_uniform(),
                               bias_init=nn.initializers.zeros
                               )
        self.norm1 = nn.LayerNorm()
        self.norm2 = nn.LayerNorm()
        self.norm3 = nn.LayerNorm()


    def __call__(self, x, context, mask=None, train=True):

        attended_x, attention1 = self.attention1(x, mask=mask)
        x = self.norm1(x + attended_x)

        attended_x, attention2 = self.attention2(x, context, mask=mask)
        x = self.norm2(x + attended_x)

        linear_output = self.linear1(x)
        x = self.norm3(x + linear_output)

        linear_output = self.linear2(x)
        x = softmax(linear_output, axis=-1)

        return x, attention1, attention2


class TransformerDecoder(nn.Module):
    """
    """
    num_layers : int
    input_dim : int
    num_heads : int

    def setup(self):
        self.layers = [DecoderBlock(self.input_dim, self.num_heads) for _ in range(self.num_layers)]

    def __call__(self, x, context, mask=None, train=True):
        for l in self.layers:
            x, _ , _ = l(x, context, mask=mask, train=train)
        return x

    def get_attention_maps(self, x, context, mask=None, train=True):
        # A function to return the attention maps within the model for a single application
        # Used for visualization purpose later
        attention_maps = []
        cross_attention_maps = []
        for layer in self.layers:
            x, attention, cross_attention = layer(x, context, mask=mask, train=train)
            attention_maps.append(attention)
            cross_attention_maps.append(cross_attention)
        return attention_maps, cross_attention_maps


class PositionEncoder(nn.Module):
    """
    """
    pass

class T5(nn.Module):
    """
    """
    pass

class BERT(nn.Module):
    """
    """
    pass

class GPTBlocks(nn.Module):
    """
    """
    pass

class GPT(nn.Module):
    """
    """
    pass

class BART(nn.Module):
    """
    """
    pass

class TransformerXL(nn.Module):
    """
    """
    pass