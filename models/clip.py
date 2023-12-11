'''
CLIP (Contrastive Language-Image Pretraining) is designed to understand and connect vision and language. 
Its motivation arises from the need to bridge the gap between textual and visual information processing in AI. 
CLIP's architecture is based on a vision-language transformer, 
which is pretrained on a large corpus of text and images from the internet, 
allowing it to learn associations between text and visuals. 
Unlike traditional models that are pretrained on single-modal data, CLIP can perform a wide range of tasks, 
including image classification, zero-shot object recognition, and even generating textual descriptions for images. 
CLIP's versatility and performance stem from its ability to encode and compare text and image representations directly, 
enabling it to generalize well across various vision and language tasks while minimizing the need for task-specific fine-tuning.
'''

import os
import jax
import optax
import pickle
import jax.numpy as jnp
import flax.linen as nn

from vit import ViT
from transformer import TransformerEncoder
from typing import Tuple
from functools import partial

class Clip(nn.Module):
    """
    CLIP (Contrastive Language-Image Pretraining) model.

    Args:
    - embed_dim (int): Dimension of the shared embedding space.
    - dropout (float): Dropout rate for model layers.
    - n_outputs (int): Number of output classes.
    - num_heads (int): Number of attention heads in the transformer layers.
    - feedforward_dim (int): Dimension of the feedforward network in transformer layers.
    - num_layers_text (int): Number of transformer layers for text encoding.
    - input_dim_text (int): Input dimension for text data.
    - image_patch_size (int): Size of image patches.
    - input_dim_image (int): Input dimension for image data.
    - num_layers_images (int): Number of transformer layers for image encoding.

    Methods:
    - setup(): Initializes the model components and parameters.
    - __call__(texts, images, training): Computes embeddings for text and images.
    - get_attention_maps(texts, images): Computes attention maps for text and images.
    - encode_text(texts): Encodes text data using the text encoder.
    - encode_image(images): Encodes image data using the image encoder.
    - embed_text(texts): Embeds text data into the shared embedding space.
    - embed_image(images): Embeds image data into the shared embedding space.

    
    Note: 
        Text input shape: (batch_size, max_length, embed_dim)
        Image input shape: (batch_size, height, width, channels)

        Image shape after patch embedding: (batch_size, sequence_length, embed_dim)

        This image sequence length can be calculated with (height * width) / (patch_height * patch_width)
    """
    embed_dim : int
    dropout: float
    n_outputs: int
    num_heads: int
    feedforward_dim: int
    num_layers_text: int
    input_dim_text: int
    image_patch_size: int
    input_dim_image: int
    num_layers_images: int

    def setup(self):
        """
        Initializes the model components and parameters.
        """
        self.text_encoder = TransformerEncoder(
            input_dim=self.input_dim_text,
            num_heads=self.num_heads,
            num_layers=self.num_layers_text,
            feedforward_dim=self.feedforward_dim,
            dropout=self.dropout,
        )
        self.image_encoder = ViT(
            patch_size=self.image_patch_size,
            num_layers=self.num_layers_images,
            input_dim=self.input_dim_image,
            num_heads=self.num_heads,
            feedforward_dim=self.feedforward_dim,
            dropout=self.dropout,
            n_outputs=self.n_outputs,
        )
        self.text_pooler = nn.Dense(self.embed_dim)
        self.image_pooler = nn.Dense(self.embed_dim)
        self.temperature = self.param('temperature', nn.initializers.zeros, ())

    def __call__(self, texts, images, training):
        """
        Computes embeddings for text and images.

        Args:
        - texts (jax.numpy.ndarray): Input text data.
        - images (jax.numpy.ndarray): Input image data.
        - training (bool): Indicates whether the model is in training mode.

        Returns:
        - text_embedding (jax.numpy.ndarray): Embedding of the input text data.
        - image_embedding (jax.numpy.ndarray): Embedding of the input image data.
        - temperature (float): Scaling factor for logits.
        """
        # Get encoded representations
        text_latents, _ = self.text_encoder(texts, training=training)
        image_latents, _ = self.image_encoder(images, training=training)
        text_embedding = self.text_pooler(jnp.mean(text_latents, axis=1))
        image_embedding = self.image_pooler(jnp.mean(image_latents, axis=1))

        return text_embedding, image_embedding

    def get_attention_maps(self, texts, images):
        """
        Computes attention maps for text and images.

        Args:
        - texts (jax.numpy.ndarray): Input text data.
        - images (jax.numpy.ndarray): Input image data.

        Returns:
        - text_attention (jax.numpy.ndarray): Attention maps for text encoding.
        - image_attention (jax.numpy.ndarray): Attention maps for image encoding.
        """
        _, text_attention = self.text_encoder(texts, training=False)
        _, image_attention = self.image_encoder(images, training=False)
        return text_attention, image_attention

    def encode_text(self, texts):
        """
        Encodes text data using the text encoder.

        Args:
        - texts (jax.numpy.ndarray): Input text data.

        Returns:
        - text_encoding (jax.numpy.ndarray): Encoded text representation.
        """
        return self.text_encoder(texts)

    def encode_image(self, images):
        """
        Encodes image data using the image encoder.

        Args:
        - images (jax.numpy.ndarray): Input image data.

        Returns:
        - image_encoding (jax.numpy.ndarray): Encoded image representation.
        """
        return self.image_encoder(images)

    def embed_text(self, texts):
        """
        Embeds text data into the shared embedding space.

        Args:
        - texts (jax.numpy.ndarray): Input text data.

        Returns:
        - text_embedding (jax.numpy.ndarray): Embedded text representation.
        """
        return self.text_pooler(self.text_encoder(texts))

    def embed_image(self, images):
        """
        Embeds image data into the shared embedding space.

        Args:
        - images (jax.numpy.ndarray): Input image data.

        Returns:
        - image_embedding (jax.numpy.ndarray): Embedded image representation.
        """
        return self.image_pooler(self.image_encoder(images))
    

@jax.jit
def clip_loss(text_embeddings, image_embeddings, temperature):
    """
    Compute the CLIP loss between image and text embeddings.

    Args:
    - image_embeddings (jax.numpy.ndarray): Image embeddings with shape (batch_size, embedding_size).
    - text_embeddings (jax.numpy.ndarray): Text embeddings with shape (batch_size, embedding_size).
    - temperature (float): Scaling factor for the logits.

    Returns:
    - float: Mean CLIP loss.

    The function calculates the CLIP loss, which measures the similarity between image and text embeddings
    by computing the cross-entropy loss between predicted and target distributions.

    - Calculate L2 normalization for both image and text embeddings.
    - Calculate logits as the dot product of text and image embeddings, divided by the temperature.
    - Compute image and text similarity matrices.
    - Calculate the target distribution as the softmax of the average similarity matrix.
    - Calculate cross-entropy loss for both images and texts.
    - Compute the final loss as the average of both losses.

    Note:
    - The @partial decorator is used to apply jax.vmap with specific in_axes.
    - jax.nn.log_softmax and jax.nn.softmax are used for numerical stability.
    """
    def l2_normalise(x):
        return x / jnp.linalg.norm(x, axis=-1, keepdims=True)

    def cross_entropy(preds, targets):
        return (-targets * jax.nn.log_softmax(preds)).sum(axis=1).mean()
    
    text_embeddings = l2_normalise(text_embeddings)
    image_embeddings = l2_normalise(image_embeddings)
    similarity_matrix = image_embeddings @ text_embeddings.T / temperature
    labels = jnp.arange(similarity_matrix.shape[0])
    image_loss = cross_entropy(similarity_matrix, labels)
    text_loss = cross_entropy(similarity_matrix.T, labels)

    return (image_loss + text_loss) / 2


@jax.jit
def forward_pass(model,
               params,
               optax_optimizer: optax.GradientTransformation,
               rng: jax.random.PRNGKey,
               text_embedding: jnp.ndarray,
               image_embedding: jnp.ndarray,
               opt_state=None) -> tuple:
    """
    Perform a single training step for a contrastive learning model.

    Args:
        model (jax.lax.lax._jaxen.Model): The model to train.
        params (jax.lax.lax._core.FrozenDict): The model's parameters.
        optax_optimizer (optax.GradientTransformation): The optimizer.
        rng (jax.random.PRNGKey): The random number generator key.
        text_embedding (jax.interpreters.xla.DeviceArray): The text embedding.
        image_embedding (jax.interpreters.xla.DeviceArray): The image embedding.
        opt_state (optax.OptState, optional): The optimizer state. Default is None.

    Returns:
        tuple: A tuple containing updated parameters, updated optimizer state (if available),
               loss, and a new random number generator key.
    """
    
    # Split the random key into two for different uses
    rng, dropout_apply_rng = jax.random.split(rng, 2)

    # Apply the model to the text and image embeddings with dropout
    text_embedding, image_embedding = model.apply(
        {'params': params},
        text_embedding,
        image_embedding,
        training=True,
        rngs={'dropout': dropout_apply_rng}
    )

    # If there's no optimizer state, calculate the loss and return the updated parameters
    if not opt_state:
        loss, grads = clip_loss(text_embedding, image_embedding, model.temperature)
        return params, loss.mean(), rng

    # Calculate the loss and gradients with respect to the parameters
    loss, grads = jax.value_and_grad(clip_loss)(
        text_embedding,
        image_embedding,
        model.temperature
    )

    # Update the optimizer state and parameters using the gradients
    updates, opt_state = optax_optimizer.update(grads, opt_state, params=params)
    params = optax.apply_updates(params=params, updates=updates)

    return params, opt_state, loss.mean(), rng