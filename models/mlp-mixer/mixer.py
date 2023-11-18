'''
MLP Mixers are a recent architectural innovation in the field of deep learning, introduced to address the limitations of traditional Convolutional Neural Networks (CNNs) and Transformers. 
The motivation behind MLP Mixers arises from the need to handle diverse data types and leverage multi-modal information efficiently. Unlike transformers that rely on self-attention mechanisms, 
MLP Mixers employ a simple yet powerful approach using Multi-Layer Perceptrons (MLPs) to process data. This architecture is designed to work with sequences, images, or even a combination of both, 
making it versatile for a wide range of tasks. MLP Mixers have demonstrated strong performance in various applications, including image classification, natural language understanding, and cross-modal learning, 
showcasing their potential in handling different modalities and promoting model efficiency and scalability in deep learning.
'''

from typing import Tuple
import jax.numpy as jnp
import flax.linen as nn

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


class MixerBlock(nn.Module):
    """
    Implements a single block of the MLP-Mixer architecture.
    
    Args:
        dim (int): The dimensionality of the block's hidden layers.
    """
    dim: int

    @nn.compact
    def __call__(self, x):
        """
        Applies the MixerBlock to an input tensor.

        Args:
            x (jax.numpy.ndarray): Input tensor of shape (batch_size, sequence_length, dim).

        Returns:
            jax.numpy.ndarray: Output tensor of the same shape as the input.
        """
        # Create a skip connection
        skip = x.copy()
        
        # Layer Normalization
        x = nn.LayerNorm()(x)
        
        # Transpose for processing across sequence_length
        x = jnp.transpose(x, axes=(0, 2, 1))
        
        # Fully connected layer with GELU activation
        x =  nn.gelu(nn.Dense(self.dim))(x)
        
        # Transpose back to the original shape and add the skip connection
        x = jnp.transpose(x, axes=(0, 2, 1)) + skip
        skip = x.copy()
        
        # Layer Normalization
        x = nn.LayerNorm()(x)
        
        # Fully connected layer with GELU activation & skip
        return nn.gelu(nn.Dense(self.dim))(x) + skip


class MLPMixer(nn.Module):
    """
    Implements the MLP-Mixer architecture.

    Args:
        num_blocks (int): Number of MixerBlocks to stack.
        patch_size (Tuple[int, int]): Size of image patches.
        embed_dim (int): Dimensionality of the patch embeddings.

    Example Usage:
        model = MLPMixer(num_blocks=1, patch_size=(16,16), embed_dim=256)
        x = jax.random.normal(jax.random.PRNGKey(0), (1, 256, 256, 3) )
        params = model.init(rng, x)
        output = model.apply(params, x)

    """
    num_blocks: int
    patch_size: Tuple[int, int]
    embed_dim: int 

    @nn.compact
    def __call__(self, x):
        """
        Applies the MLP-Mixer to an input tensor.

        Args:
            x (jax.numpy.ndarray): Input tensor of shape (batch_size, sequence_length, embed_dim).

        Returns:
            jax.numpy.ndarray: Output tensor of the same shape as the input.
        """
        # Apply PatchEmbedding (assuming it's defined elsewhere)
        x = PatchEmbedding(self.patch_size, self.embed_dim)(x)
        
        # Apply multiple MixerBlocks
        for _ in range(self.num_blocks):
            x = MixerBlock(x.shape[-1])(x)

        return x