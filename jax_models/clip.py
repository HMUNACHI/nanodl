import os
import jax
import optax
import pickle
import jax.numpy as jnp
import flax.linen as nn

from _vit import ViT
from _transformer import TransformerEncoder
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

    Example:
    key = jax.random.PRNGKey(0)
    main_rng, init_rng, dropout_init_rng = jax.random.split(key, 3)
    texts = jax.random.normal(jax.random.PRNGKey(10), (3, 16, 128))
    images = jax.random.normal(jax.random.PRNGKey(20),(3,256,256,3))
    patch_size = (16, 16)
    input_dim = int((images.shape[1] * images.shape[2]) / (patch_size[0] * patch_size[1]))

    # Initialise model
    clip = Clip(embed_dim = 256,
                dropout = 0.2,
                n_outputs = 256,
                num_heads = 4,
                feedforward_dim = 256,
                num_layers_text = 2,
                input_dim_text = 128,
                image_patch_size = patch_size,
                input_dim_image = input_dim,
                num_layers_images = 2)

    # Setup parameters
    params = clip.init({'params': init_rng, 'dropout': dropout_init_rng},
                        texts, 
                        images,
                        training=True)['params']

    print('Number of parameters:', sum(x.size for x in jax.tree_leaves(params)))
    main_rng, dropout_apply_rng = jax.random.split(main_rng)
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
        self.text_projection = nn.Dense(self.embed_dim)
        self.image_projection = nn.Dense(self.embed_dim)
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

        # Flatten tensors
        # text_latents = jax.vmap(jnp.ravel)(text_latents)
        # image_latents = jax.vmap(jnp.ravel)(image_latents)

        # Project latents onto shared embedding space
        text_embedding = self.text_projection(text_latents)
        image_embedding = self.image_projection(image_latents)

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
        return self.text_projection(self.text_encoder(texts))

    def embed_image(self, images):
        """
        Embeds image data into the shared embedding space.

        Args:
        - images (jax.numpy.ndarray): Input image data.

        Returns:
        - image_embedding (jax.numpy.ndarray): Embedded image representation.
        """
        return self.image_projection(self.image_encoder(images))
    

@partial(jax.vmap, in_axes=(0, 0, None))
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

    def cross_entropy(preds, targets, reduction='none'):
        log_softmax = jax.nn.log_softmax
        loss = (-targets * log_softmax(preds)).sum(axis=1)
        if reduction == "none":
            return loss
        elif reduction == "mean":
            return loss.mean()

    # L2 normalize both image and text embeddings
    text_embeddings = l2_normalise(text_embeddings)
    image_embeddings = l2_normalise(image_embeddings)
    
    # Calculate logits
    logits = (jnp.dot(text_embeddings, image_embeddings.T)) / temperature

    # Calculate image and text similarity matrices
    images_similarity = jnp.dot(image_embeddings, image_embeddings.T)
    texts_similarity = jnp.dot(text_embeddings, text_embeddings.T)

    # Calculate the target distribution as the softmax of the average similarity matrix
    targets = jax.nn.softmax(
        (images_similarity + texts_similarity) / 2 * temperature, axis=-1
    )

    # Calculate cross-entropy loss for both images and texts
    texts_loss = cross_entropy(logits, targets, reduction='none')
    images_loss = cross_entropy(jnp.transpose(logits), jnp.transpose(targets), reduction='none')

    # Compute the final loss as the average of both losses
    loss = (images_loss + texts_loss) / 2.0 

    return loss.mean()


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


@jax.jit
def train_step(model,
               params,
               optax_optimizer: optax.GradientTransformation,
               rng: jax.random.PRNGKey,
               text_embedding: jnp.ndarray,
               image_embedding: jnp.ndarray,
               opt_state=None) -> tuple:
    
    updates = jax.pmap(forward_pass, 
                       in_axes=(None, None, None, None, 0, 0, None)
                       )(model,
                        params,
                        optax_optimizer: optax.GradientTransformation,
                        rng: jax.random.PRNGKey,
                        text_embedding: jnp.ndarray,
                        image_embedding: jnp.ndarray,
                        opt_state=None)
    return jax.tree_util.tree_map(lambda x: x.mean(axis=0), updates)


def train(config: dict, 
          optax_optimizer: optax.GradientTransformation, 
          train_loader, 
          val_loader, 
          text_shape: tuple, 
          image_shape: tuple,
          checkpoint_directory: str,
          epochs: int):
    """
    Train a CLIP model using the specified parameters and data loaders.

    Args:
    - model_params (dict): Parameters for initializing the CLIP model.
    - optax_optimizer (optax.GradientTransformation): Optimizer for model training.
    - train_loader (DataLoader): DataLoader for training data.
    - val_loader (DataLoader): DataLoader for validation data.
    - text_shape (tuple): Shape of the text input (e.g., (batch_size, sequence_length)).
    - image_shape (tuple): Shape of the image input (e.g., (batch_size, height, width, channels)).
    - checkpoint_directory (str): Directory to save model checkpoints.
    - epochs (int): Number of training epochs.

    Returns:
    - None
    """
    # Create random keys
    num_devices = jax.device_count() # Remove if not distributing
    key = jax.random.PRNGKey(0)
    main_rng, init_rng, dropout_init_rng = jax.random.split(key, 3)
    
    # Initialise model
    clip = Clip(embed_dim = config['embed_dim'],
                dropout = config['dropout'],
                n_outputs = config['n_outputs'],
                num_heads = config['num_heads'],
                feedforward_dim = config['feedforward_dim'],
                num_layers_text = config['num_layers_text'],
                input_dim_text = config['input_dim_text'],
                image_patch_size = config['image_patch_size'],
                input_dim_image = config['input_dim_image'],
                num_layers_images = config['num_layers_images'])

    # Setup parameters
    params = clip.init({'params': init_rng, 'dropout': dropout_init_rng},
                       jnp.ones(text_shape), 
                       jnp.ones(image_shape),
                       training=True)['params']
    
    print('Number of parameters:', sum(x.size for x in jax.tree_leaves(params)))
    
    # Create optimizer state
    opt_state = optax_optimizer.init(params)

    # Replicate params across devices 
    params = jax.pmap(lambda x: params)(jnp.arange(num_devices)) # Remove if not distributing

    for epoch in epochs:
        
        train_losses = []
        for text_batch, image_batch in train_loader:
            # Split batch across devices
            text_batch = jnp.array(jnp.split(text_batch, num_devices, axis=0))
            image_batch = jnp.array(jnp.split(image_batch, num_devices, axis=0))
            params, opt_state, train_loss, main_rng = train_step(clip, 
                                                                params,
                                                                optax_optimizer,
                                                                main_rng,
                                                                text_batch, 
                                                                image_batch,
                                                                opt_state=opt_state)
            
            # To Do: Aggregate forward_pass results from multiple devices 

            train_losses.append(train_loss.mean())
            # ... do whatever with losses ...

        val_losses = []
        for text_batch, image_batch in val_loader:
            # Split batch across devices
            text_batch = jnp.array(jnp.split(text_batch, num_devices, axis=0))   # Remove if not distributing
            image_batch = jnp.array(jnp.split(image_batch, num_devices, axis=0)) # Remove if not distributing

            params, val_loss, main_rng = train_step(clip, 
                                                    params,
                                                    optax_optimizer,
                                                    main_rng,
                                                    text_batch, 
                                                    image_batch)
            
            # To Do: Aggregate forward_pass results from multiple devices 

            val_losses.append(val_loss.mean())
            # ... do whatever with losses ...

        # Save every version of the model, can be modified to track and save best only
        pickle.dump(params, open(os.path.join(checkpoint_directory, str(epoch)), "wb"))


    return
