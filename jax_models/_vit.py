import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Tuple
from _attention import SelfMultiHeadAttention

class PatchEmbedding(nn.Module):
    """
    A Flax module for patch embedding in a vision transformer.

    Args:
    patch_size (tuple): Size of the patches (height, width).
    embed_dim (int): Dimension of the embedded patches.

    Attributes:
    patch_size (tuple): Size of the patches (height, width).
    embed_dim (int): Dimension of the embedded patches.
    """

    patch_size: Tuple[int, int]
    embed_dim: int 

    @nn.compact
    def __call__(self, x):
        """
        Apply the PatchEmbedding module to input data.

        Args:
        x (jax.numpy.ndarray): Input data with shape (batch_size, height, width, channels).

        Returns:
        jax.numpy.ndarray: Embedded patches with shape (batch_size, num_patches, embed_dim).
        """
        x = nn.Dense(self.embed_dim)(self.extract_patches(x))
        return x + nn.Embed(num_embeddings=x.shape[1], features=x.shape[2])(jnp.arange(x.shape[1]))

    def extract_patches(self, images: jnp.ndarray) -> jnp.ndarray:
        """
        Split multiple images into patches of a specified size and flatten each patch.

        Args:
        images (jax.numpy.ndarray): Input images as a JAX array with shape (batch_size, height, width, channels).

        Returns:
        jax.numpy.ndarray: Flattened array containing image patches for all input images.
        """
        if len(images.shape) != 4:
            raise ValueError("Input images should have shape (batch_size, H, W, C)")
        
        batch_size, h, w, c = images.shape
        ph, pw = self.patch_size

        if h % ph != 0 or w % pw != 0:
            raise ValueError("Image dimensions must be divisible by patch size.")

        # Calculate the number of patches in each dimension
        num_patches_h = h // ph
        num_patches_w = w // pw

        # Reshape the images into patches and flatten each patch
        patches = jnp.reshape(images, (batch_size, num_patches_h, ph, num_patches_w, pw, c))
        patches = jnp.transpose(patches, (0, 1, 3, 2, 4, 5))
        patches = jnp.reshape(patches, (batch_size, -1, ph * pw * c))
        return patches


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
        self.linear = PositionWiseFFN(self.feedforward_dim, self.input_dim)
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
    num_layers: int
    input_dim: int
    num_heads: int
    feedforward_dim: int
    dropout: float

    def setup(self):
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
        for layer in self.layers:
            x, attention = layer(x, mask=mask, training=training)
            attention_maps.append(attention)
        return x, jnp.array(attention_maps)
    


class ViT(nn.Module):
    """
    Vision Transformer (ViT) model for image classification.

    Args:
    patch_size (tuple): Size of the patches (height, width).
    num_layers (int): Number of transformer encoder layers.
    input_dim (int): Input dimension for the transformer encoder.
    num_heads (int): Number of attention heads in the transformer encoder.
    feedforward_dim (int): Dimension of the feedforward layers in the transformer encoder.
    dropout (float): Dropout probability for regularization.
    n_outputs (int): Number of output classes.

    Note: The transformer MLP blocks were designed to have a bottleneck
          As such, the embeddining dim and feedforward dim should be the same to 
    """

    patch_size: Tuple[int, int]
    num_layers: int
    input_dim: int
    num_heads: int
    feedforward_dim: int
    dropout: float
    n_outputs: int

    def setup(self):
        """
        Setup the ViT model architecture by initializing its components.

        Initializes the embedding layer, transformer encoder blocks, and the output layer.
        """
        self.embedding = PatchEmbedding(self.patch_size, 
                                        self.feedforward_dim
                                        )
        self.blocks = TransformerEncoder(self.num_layers, 
                                         self.input_dim, 
                                         self.num_heads, 
                                         self.feedforward_dim, 
                                         self.dropout
                                         )
        self.output = nn.Dense(self.n_outputs)

    def __call__(self, 
                 x: jnp.ndarray, 
                 mask: jnp.ndarray = None, 
                 training: bool = True) -> tuple:
        """
        Apply the ViT model to input data.

        Args:
        x (jax.numpy.ndarray): Input data with shape (batch_size, height, width, channels).

        Returns:
        jax.numpy.ndarray: Predicted class scores for each input sample.
        jax.numpy.ndarray: Attention maps from the transformer encoder.
        """
        x = self.embedding(x)
        x, attention_maps = self.blocks(x=x, mask=mask, training=training)#
        return self.output(x), attention_maps