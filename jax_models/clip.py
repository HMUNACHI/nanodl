import os
import sys
import jax
import optax
import pickle
import jax.numpy as jnp
import flax.linen as nn

from _transformer import TransformerEncoder

class Clip(nn.Module):
    embed_dim : int
    text_layers : int
    text_hidden : int
    image_layers : int
    image_hidden : int

    def setup(self):
        self.text_encoder = TransformerEncoder()
        self.image_encoder = ViT()
        self.text_projection = nn.Dense(self.embed_dim)
        self.image_projection = nn.Dense(self.embed_dim)
        self.temperature = self.param('temperature', nn.initializers.zeros, ())

    def __call__(self, texts, images):
        # Get encoded representations
        text_latents = self.text_encoder(texts)
        image_latents = self.image_encoder(images)

        # Project latents onto shared embedding space
        text_embedding = self.text_projection(text_latents)
        image_embedding = self.text_projection(image_latents)

        # L2 Normalisation
        text_embedding = text_embedding / jnp.sqrt(jnp.einsum('ij,ij->', text_embedding, text_embedding))
        image_embedding = image_embedding / jnp.sqrt(jnp.einsum('ij,ij->', image_embedding, image_embedding))

        # Scaled pairwise cosine similarities [n, n]
        logits = jnp.dot(image_embedding, text_embedding.T) * jnp.exp(self.temperature)
        return logits
    
    def encode_text(self, texts):
        return self.text_encoder(texts)

    def encode_image(self, images):
        return self.image_encoder(images)

    def embed_text(self, texts):
        return self.text_projection(self.text_encoder(texts))

    def embed_image(self, images):
        self.image_projection(self.image_encoder(images))
    

@jax.value_and_grad
@jax.jit
def clip_loss (logits):
    "Uses cross entropy"
    labels = jnp.arange(logits.shape[0])
    text_loss = -jnp.sum(labels * jax.nn.log_softmax(logits, axis=1), axis=1)
    image_loss = -jnp.sum(labels * jax.nn.log_softmax(logits, axis=0), axis=0)
    return (image_loss + text_loss) / 2


def train(model_params: dict, 
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
    
    # Initialise model
    clip = Clip(embed_dim = model_params['embed_dim'],
                text_layers = model_params['text_layers'],
                text_hidden = model_params['text_hidden'],
                image_layers = model_params['image_layers'],
                image_hidden = model_params['image_hidden'],)

    # Setup parameters
    params = clip.init(jax.random.PRNGKey(1),
                       jnp.ones(text_shape), 
                       jnp.ones(image_shape))
    
    print('Number of parameters:', sum(x.size for x in jax.tree_leaves(params)))
    
    # Create optimizer state
    opt_state = optax_optimizer.init(params)

    for epoch in epochs:
        train_losses = []
        for text_batch, image_batch in train_loader:
            logits = clip.apply(params, text_batch, image_batch)
            train_loss, grads = clip_loss(logits)
            updates, opt_state = optax_optimizer.update(grads, opt_state, params=params)
            params =  optax.apply_updates(params=params, updates=updates)
            train_losses.append(train_loss)
            # ... do whatever with losses ...

        val_losses = []
        for text_batch, image_batch in val_loader:
            logits = clip.apply(params, text_batch, image_batch)
            val_loss, _ = clip_loss(logits)
            val_losses.append(val_loss)
            # ... do whatever with losses ...

        # Save every version of the model, can be modified to track and save best only
        pickle.dump(params, open(os.path.join(checkpoint_directory, str(epoch)), "wb"))

    return