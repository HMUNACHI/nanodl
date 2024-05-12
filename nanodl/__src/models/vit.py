import time
from typing import Any, Iterable, Optional, Tuple

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import optax
from flax.training import train_state


class PatchEmbedding(nn.Module):
    """
    Implements patch embedding for vision transformers.

    This module extracts patches from input images, flattens them, and projects them to a specified embedding dimension. Optionally, learned position embeddings can be added to the patch embeddings.

    Attributes:
        patch_size (tuple): Size (height, width) of the patches to extract from input images.
        embed_dim (int): Dimension of the embeddings for the patches.

    """

    patch_size: Tuple[int, int]
    embed_dim: int

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.embed_dim)(self.extract_patches(x))
        return x + nn.Embed(num_embeddings=x.shape[1], features=x.shape[2])(
            jnp.arange(x.shape[1])
        )

    def extract_patches(self, images: jnp.ndarray) -> jnp.ndarray:
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
        patches = jnp.reshape(
            images, (batch_size, num_patches_h, ph, num_patches_w, pw, c)
        )
        patches = jnp.transpose(patches, (0, 1, 3, 2, 4, 5))
        patches = jnp.reshape(patches, (batch_size, -1, ph * pw * c))
        return patches


class SelfMultiHeadAttention(nn.Module):
    """
    Implements multi-head self-attention mechanism as described in "Attention is All You Need" by Vaswani et al 2017.

    This module splits the input into multiple heads, applies scaled dot-product attention independently on each head, and then concatenates the results. It allows the model to jointly attend to information from different representation subspaces at different positions.

    Attributes:
        hidden_dim (int): Dimensionality of the input and output features.
        num_heads (int): Number of attention heads.

    """

    hidden_dim: int  # Output dimension
    num_heads: int  # Number of parallel heads

    def setup(self):
        # Stack all weight matrices together for efficiency
        self.projection = nn.Dense(
            3 * self.hidden_dim,
            kernel_init=nn.initializers.xavier_uniform(),
            bias_init=nn.initializers.zeros,
        )
        self.output = nn.Dense(
            self.hidden_dim,
            kernel_init=nn.initializers.xavier_uniform(),
            bias_init=nn.initializers.zeros,
        )

    def __call__(self, inputs: jnp.ndarray, mask: jnp.ndarray = None) -> tuple:

        projections = self.projection(inputs)
        query, key, value = jnp.array_split(projections, 3, axis=-1)
        context_vectors, attention = self.attention_function(
            query, key, value, mask=mask
        )
        outputs = self.output(context_vectors)
        return outputs, attention

    def attention_function(self, query, key, value, mask=None):
        input_length = query.shape[1]
        context_length = key.shape[1]
        head_dim = query.shape[-1] // self.num_heads
        dim_key = key.shape[-1]

        # Split queries, keys, and values into heads
        query_heads = jnp.reshape(
            query, (query.shape[0], self.num_heads, input_length, head_dim)
        )
        key_heads = jnp.reshape(
            key, (key.shape[0], self.num_heads, context_length, head_dim)
        )
        value_heads = jnp.reshape(
            value, (value.shape[0], self.num_heads, context_length, head_dim)
        )

        attention_scores = jnp.matmul(
            query_heads, key_heads.transpose(0, 1, 3, 2)
        ) / jnp.sqrt(dim_key)
        if mask is not None:
            attention_scores = attention_scores * mask

        attention_weights = jax.nn.softmax(attention_scores, axis=-1)
        attended_values = jnp.matmul(attention_weights, value_heads)
        attended_values = jnp.reshape(
            attended_values, (query.shape[0], input_length, query.shape[-1])
        )
        return attended_values, attention_weights


class PositionWiseFFN(nn.Module):
    """
    Implements the position-wise feed-forward network of a transformer model.

    This module applies two linear transformations with a gelu activation in between, as per the original transformer model design. It is applied to each position separately and identically.

    Attributes:
        num_hiddens (int): The number of hidden units in the first linear layer.
        num_outputs (int): The number of output units in the second linear layer (usually the same as the model's hidden size).

    """

    num_hiddens: int
    num_outputs: int

    def setup(self):
        self.dense1 = nn.Dense(
            self.num_hiddens, kernel_init=nn.initializers.xavier_uniform()
        )
        self.dense2 = nn.Dense(
            self.num_outputs, kernel_init=nn.initializers.xavier_uniform()
        )

    def __call__(self, X: jnp.ndarray) -> jnp.ndarray:
        return self.dense2(nn.gelu(self.dense1(X)))


class AddNorm(nn.Module):
    """
    Implements a residual connection followed by layer normalization.

    This module is a common building block in transformer models, promoting easier optimization and enabling deeper networks.

    Attributes:
        dropout (float): Dropout rate for the residual connection.

    """

    dropout: int

    @nn.compact
    def __call__(self, X: jnp.ndarray, Y: jnp.ndarray, training=False) -> jnp.ndarray:
        return nn.LayerNorm()(
            nn.Dropout(self.dropout)(Y, deterministic=not training) + X
        )


class ViTBlock(nn.Module):
    """
    Represents a single block in the transformer encoder.

    Each encoder block consists of a multi-head self-attention layer and a position-wise feed-forward network. Both sublayers have residual connections and are followed by layer normalization.

    Attributes:
        hidden_dim (int): Dimensionality of the input and output features.
        num_heads (int): Number of attention heads.
        feedforward_dim (int): Dimension of the feed-forward network.
        dropout (float): Dropout rate.

    """

    hidden_dim: int
    num_heads: int
    feedforward_dim: int
    dropout: float

    def setup(self):
        self.attention = SelfMultiHeadAttention(
            hidden_dim=self.hidden_dim, num_heads=self.num_heads
        )
        self.ff = PositionWiseFFN(self.feedforward_dim, self.hidden_dim)
        self.add_norm1 = AddNorm(self.dropout)
        self.add_norm2 = AddNorm(self.dropout)

    def __call__(
        self, x: jnp.ndarray, mask: jnp.ndarray = None, training: bool = False
    ) -> tuple:

        attended_x, attention = self.attention(x, mask=mask)
        x = self.add_norm1(x, attended_x, training)
        ff_output = self.ff(x)
        x = self.add_norm2(x, ff_output, training)
        return x, attention


class ViTEncoder(nn.Module):
    """
    Implements a vision transformer (ViT) encoder for image processing.

    This module applies patch embedding to input images and then processes the resulting sequence of embedded patches through multiple transformer encoder blocks.

    Attributes:
        patch_size (tuple): Size of the patches (height, width) to be extracted from input images.
        num_layers (int): Number of transformer encoder blocks.
        hidden_dim (int): Dimensionality of the input and output features for the transformer encoder.
        num_heads (int): Number of attention heads in the transformer encoder.
        feedforward_dim (int): Dimension of the feed-forward network in the transformer encoder.
        dropout (float): Dropout rate for regularization.

    """

    patch_size: Tuple[int, int]
    num_layers: int
    hidden_dim: int
    num_heads: int
    feedforward_dim: int
    dropout: float

    def setup(self):
        self.embedding = PatchEmbedding(self.patch_size, self.feedforward_dim)

        self.layers = [
            ViTBlock(
                self.hidden_dim, self.num_heads, self.feedforward_dim, self.dropout
            )
            for _ in range(self.num_layers)
        ]

    def __call__(
        self, x: jnp.ndarray, mask: jnp.ndarray = None, training: bool = False
    ) -> tuple:

        attention_maps = []
        x = self.embedding(x)
        for layer in self.layers:
            x, attention = layer(x, mask=mask, training=training)
            attention_maps.append(attention)
        return x, jnp.array(attention_maps)


class ViT(nn.Module):
    """
    Implements the encoder component of the Vision Transformer (ViT) model.

    The ViTEncoder processes input images divided into patches through multiple Transformer encoder layers. It aims to capture complex patterns within the data by applying self-attention and feed-forward networks to the sequence of patches.

    Attributes:
        patch_size (Tuple[int, int]): Size of the patches the image is divided into.
        num_layers (int): Number of Transformer encoder layers in the encoder.
        hidden_dim (int): Dimensionality of the hidden features.
        num_heads (int): Number of attention heads in the self-attention mechanism.
        feedforward_dim (int): Dimensionality of the feedforward network within each Transformer encoder layer.
        dropout (float): Dropout rate for regularization.

    Vision Transformers, or ViTs, have emerged as a groundbreaking architectural paradigm in computer vision and deep learning.
    The motivation behind Vision Transformers lies in the desire to extend the success of transformers,
    originally designed for natural language processing, to visual data. These models aim to replace
    or complement traditional Convolutional Neural Networks (CNNs) in image-related tasks. ViTs employ a self-attention mechanism
    to capture global dependencies among pixels or patches of an image, which helps them understand context and relationships between different regions effectively.
    By utilizing pretraining on large-scale image datasets, ViTs have achieved remarkable performance in image classification, object detection, image generation, and various other computer vision tasks.
    Their modular design, scalability, and ability to handle both local and global information have made Vision Transformers a significant advancement in the field,
    offering promising avenues for future research and applications in computer vision.

    Example usage:
        ```py
        import jax
        import jax.numpy as jnp
        from nanodl import ArrayDataset, DataLoader
        from nanodl import ViT, ViTDataParallelTrainer

        # Dummy data parameters
        batch_size = 8
        max_length = 50
        n_outputs = 5
        embed_dim = 256
        patch_size = (16, 16)

        # Generate data
        dummy_inputs = jnp.ones((batch_size, 224, 224, 3))
        key = jax.random.PRNGKey(10)
        dummy_labels = jax.random.randint(key,
                                        shape=(batch_size,),
                                        minval=0,
                                        maxval=n_outputs-1)

        # Create dataset and dataloader
        dataset = ArrayDataset(dummy_inputs,
                            dummy_labels)

        dataloader = DataLoader(dataset,
                                batch_size=batch_size,
                                shuffle=True,
                                drop_last=False)

        # model parameters
        hyperparams = {
            "dropout": 0.1,
            "num_heads": 2,
            "feedforward_dim": embed_dim,
            "patch_size": patch_size,
            "hidden_dim": embed_dim,
            "num_layers": 4,
            "n_outputs": n_outputs
        }

        # Initialize model
        model = ViT(**hyperparams)
        rngs = {'params': jax.random.key(0), 'dropout': jax.random.key(1)}
        params = model.init(rngs, dummy_inputs)['params']
        outputs = model.apply({'params': params}, dummy_inputs, rngs=rngs)[0]
        print(outputs.shape)

        # Training on your data
        trainer = ViTDataParallelTrainer(model, dummy_inputs.shape, 'params.pkl')
        trainer.train(dataloader, 10, dataloader)
        ```
    """

    patch_size: Tuple[int, int]
    num_layers: int
    hidden_dim: int
    num_heads: int
    feedforward_dim: int
    dropout: float
    n_outputs: int

    def setup(self):
        self.encoder = ViTEncoder(
            patch_size=self.patch_size,
            num_layers=self.num_layers,
            hidden_dim=self.hidden_dim,
            num_heads=self.num_heads,
            feedforward_dim=self.feedforward_dim,
            dropout=self.dropout,
        )
        self.dropout_layer = nn.Dropout(self.dropout)
        self.output = nn.Dense(self.n_outputs)

    def __call__(
        self, x: jnp.ndarray, mask: jnp.ndarray = None, training: bool = False
    ) -> tuple:

        x, attention_maps = self.encoder(x=x, mask=mask, training=training)
        x = self.dropout_layer(x, deterministic=not training)
        return self.output(x[:, 0, :]), x, attention_maps


class ViTDataParallelTrainer:
    """
    Trainer class using data parallelism with JAX.
    This trainer leverages JAX's `pmap` for parallel training across multiple devices (GPUs/TPUs).
    It handles the model training loop, including gradient computation, parameter updates, and evaluation.

    Attributes:
        model (Any): The model to be trained.
        input_shape (Tuple[int, ...]): The shape of the image input tensor.
        weights_filename (str): Filename where the trained model weights will be saved.
        learning_rate (float): Learning rate for the optimizer.
        params_path (Optional[str]): Path to pre-trained model parameters for initializing the model, if available.

    Methods:
        create_train_state(learning_rate, text_input_shape, image_input_shape): Initializes the training state, including parameters and optimizer.
        train_step(state, texts, images): Performs a single training step, including forward pass, loss computation, and gradients update.
        train(train_loader, num_epochs, val_loader): Runs the training loop over the specified number of epochs, using the provided data loaders for training and validation.
        evaluation_step(state, texts, images): Performs an evaluation step, computing forward pass and loss without updating model parameters.
        evaluate(test_loader): Evaluates the model performance on a test dataset.
        save_params(): Saves the model parameters to a file.
        load_params(filename): Loads model parameters from a file.
    """

    def __init__(
        self,
        model: Any,
        input_shape: Tuple[int, ...],
        weights_filename: str,
        learning_rate: float = 1e-5,
        params_path: Optional[str] = None,
    ) -> None:
        self.model = model
        self.params = None
        self.params_path = params_path
        self.num_parameters = None
        self.best_val_loss = float("inf")
        self.weights_filename = weights_filename
        self.num_devices = jax.local_device_count()
        self.train_step = jax.pmap(
            ViTDataParallelTrainer.train_step, axis_name="devices"
        )
        self.evaluation_step = jax.pmap(
            ViTDataParallelTrainer.evaluation_step, axis_name="devices"
        )
        self.state = self.create_train_state(learning_rate, input_shape)
        print(f"Number of accelerators: {self.num_devices}")

    def create_train_state(
        self, learning_rate: float, input_shape: Tuple[int, ...]
    ) -> Any:

        rngs = {"params": jax.random.key(0), "dropout": jax.random.key(1)}
        params = self.model.init(rngs, jnp.ones(input_shape))["params"]

        if self.params_path is not None:
            params = self.load_params(self.params_path)

        self.num_parameters = sum(
            param.size for param in jax.tree_util.tree_leaves(params)
        )
        print(f"Number of parameters: {self.num_parameters}")
        state = train_state.TrainState.create(
            apply_fn=self.model.apply, params=params, tx=optax.adam(learning_rate)
        )
        return jax.device_put_replicated(state, jax.local_devices())

    @staticmethod
    def train_step(
        state: Any, inputs: jnp.ndarray, targets: jnp.ndarray
    ) -> Tuple[Any, jnp.ndarray]:

        def loss_fn(params):
            logits = state.apply_fn(
                {"params": params},
                inputs,
                training=True,
                rngs={"dropout": jax.random.PRNGKey(int(time.time()))},
            )[0]
            return -jnp.mean(
                jax.vmap(jax.nn.log_softmax)(logits)[jnp.arange(targets.size), targets]
            )

        loss, grads = jax.value_and_grad(loss_fn)(state.params)
        state = state.apply_gradients(grads=grads)
        return state, loss

    def train(
        self,
        train_loader: Iterable[Tuple[jnp.ndarray, jnp.ndarray]],
        num_epochs: int,
        val_loader: Optional[Iterable[Tuple[jnp.ndarray, jnp.ndarray]]] = None,
    ) -> None:

        for epoch in range(num_epochs):
            total_loss = 0.0
            count = 0
            for inputs, targets in train_loader:
                batch_size = inputs.shape[0]
                batch_size_per_device = batch_size // self.num_devices
                inputs = inputs.reshape(
                    (
                        self.num_devices,
                        batch_size_per_device,
                        inputs.shape[1],
                        inputs.shape[2],
                        inputs.shape[3],
                    )
                )
                targets = targets.reshape((self.num_devices, batch_size_per_device, -1))
                self.state, loss = self.train_step(
                    state=self.state, inputs=inputs, targets=targets
                )
                total_loss += jnp.mean(loss)
                count += 1

            mean_loss = total_loss / count
            print(f"Epoch {epoch+1}, Train Loss: {mean_loss}")

            if val_loader is not None:
                val_loss = self.evaluate(val_loader)
                print(f"Epoch {epoch+1}, Val Loss: {val_loss}")
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                print("New best validation score achieved, saving model...")
                self.save_params()
        return

    @staticmethod
    def evaluation_step(
        state: Any, inputs: jnp.ndarray, targets: jnp.ndarray
    ) -> Tuple[Any, jnp.ndarray]:

        logits = state.apply_fn(
            {"params": state.params}, inputs, rngs={"dropout": jax.random.PRNGKey(2)}
        )[0]
        return -jnp.mean(
            jax.vmap(jax.nn.log_softmax)(logits)[jnp.arange(targets.size), targets]
        )

    def evaluate(self, test_loader: Iterable[Tuple[jnp.ndarray, jnp.ndarray]]) -> None:

        total_loss = 0.0
        count = 0
        for inputs, targets in test_loader:
            batch_size = inputs.shape[0]
            batch_size_per_device = batch_size // self.num_devices
            inputs = inputs.reshape(
                (
                    self.num_devices,
                    batch_size_per_device,
                    inputs.shape[1],
                    inputs.shape[2],
                    inputs.shape[3],
                )
            )
            targets = targets.reshape((self.num_devices, batch_size_per_device, -1))
            loss = self.evaluation_step(self.state, inputs, targets)
            total_loss += jnp.mean(loss)
            count += 1

        mean_loss = total_loss / count
        return mean_loss

    def save_params(self) -> None:
        self.params = flax.jax_utils.unreplicate(self.state.params)
        with open(self.weights_filename, "wb") as f:
            f.write(flax.serialization.to_bytes(self.params))

    def load_params(self, filename: str):
        with open(filename, "rb") as f:
            self.params = flax.serialization.from_bytes(self.params, f.read())
        return self.params
