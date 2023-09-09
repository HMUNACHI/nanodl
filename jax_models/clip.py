import os
import jax
import optax
import pickle
import jax.numpy as jnp
import flax.linen as nn

from _vit import ViT
from _transformer import TransformerEncoder
from typing import Tuple

class Clip(nn.Module):
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
        self.text_encoder = TransformerEncoder(input_dim=self.input_dim_text, 
                                               num_heads=self.num_heads, 
                                               num_layers=self.num_layers_text, 
                                               feedforward_dim=self.feedforward_dim, 
                                               dropout=self.dropout
                                               )
        self.image_encoder = ViT(patch_size=self.image_patch_size,
                                num_layers=self.num_layers_images,
                                input_dim=self.input_dim_image, 
                                num_heads=self.num_heads, 
                                feedforward_dim=self.feedforward_dim, 
                                dropout=self.dropout,
                                n_outputs=self.n_outputs
                                )
        self.text_projection = nn.Dense(self.embed_dim)
        self.image_projection = nn.Dense(self.embed_dim)
        self.temperature = self.param('temperature', nn.initializers.zeros, ())

    def __call__(self, texts, images, training):
        # Get encoded representations
        text_latents, text_attention = self.text_encoder(texts, training=training)
        image_latents, image_attention = self.image_encoder(images, training=training)

        # Project latents onto shared embedding space
        text_embedding = self.text_projection(text_latents)
        image_embedding = self.image_projection(image_latents)

        # L2 Normalisation
        text_embedding = text_embedding / jnp.sqrt(jnp.einsum('ij,ij->', text_embedding, text_embedding))
        image_embedding = image_embedding / jnp.sqrt(jnp.einsum('ij,ij->', image_embedding, image_embedding))

        # Scaled pairwise cosine similarities [n, n]
        logits = jnp.dot(image_embedding, text_embedding.T) * jnp.exp(self.temperature)
        return logits, text_attention, image_attention
    
    def encode_text(self, texts):
        return self.text_encoder(texts)

    def encode_image(self, images):
        return self.image_encoder(images)

    def embed_text(self, texts):
        return self.text_projection(self.text_encoder(texts))

    def embed_image(self, images):
        self.image_projection(self.image_encoder(images))
    
@jax.vmap
def l2_normalise(x):
    return x / jnp.linalg.norm(x)

@jax.value_and_grad
@jax.jit
def clip_loss (logits):
    "Uses cross entropy"
    labels = jnp.arange(logits.shape[0])
    text_loss = -jnp.sum(labels * jax.nn.log_softmax(logits, axis=1), axis=1)
    image_loss = -jnp.sum(labels * jax.nn.log_softmax(logits, axis=0), axis=0)
    return (image_loss + text_loss) / 2


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
    main_rng, dropout_apply_rng = jax.random.split(main_rng)

    for epoch in epochs:
        train_losses = []
        for text_batch, image_batch in train_loader:
            logits, _, _ = clip.apply({'params': params}, 
                                text_batch, 
                                image_batch, 
                                training=True, 
                                rngs={'dropout': dropout_apply_rng}
                                )
            train_loss, grads = clip_loss(logits)
            updates, opt_state = optax_optimizer.update(grads, opt_state, params=params)
            params =  optax.apply_updates(params=params, updates=updates)
            train_losses.append(train_loss)
            # ... do whatever with losses ...

        val_losses = []
        for text_batch, image_batch in val_loader:
            logits, _, _ = clip.apply({'params': params}, 
                                text_batch, 
                                image_batch, 
                                training=True, 
                                rngs={'dropout': dropout_apply_rng}
                                )
            val_loss, _ = clip_loss(logits)
            val_losses.append(val_loss)
            # ... do whatever with losses ...

        # Save every version of the model, can be modified to track and save best only
        pickle.dump(params, open(os.path.join(checkpoint_directory, str(epoch)), "wb"))

    return


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