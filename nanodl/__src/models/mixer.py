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


class MixerBlock(nn.Module):
    """
    Implements a single Mixer block, part of the MLP-Mixer architecture.

    The Mixer block applies a two-step mixing process: the first step mixes per-location features across the channel dimension, and the second step mixes per-channel features across spatial locations. It aims to capture both channel-wise and spatial interactions within the input.

    """

    @nn.compact
    def __call__(self, x):
        # Create a skip connection
        skip = x.copy()
        x = nn.LayerNorm()(x)
        x = jnp.transpose(x, axes=(0, 2, 1))
        x = nn.gelu(nn.Dense(x.shape[-1])(x))
        x = jnp.transpose(x, axes=(0, 2, 1)) + skip
        skip = x.copy()
        x = nn.LayerNorm()(x)
        return nn.gelu(nn.Dense(x.shape[-1])(x)) + skip


class MixerEncoder(nn.Module):
    """
    Implements the encoder component of the MLP-Mixer model.

    The MixerEncoder processes input images through an embedding layer followed by multiple MixerBlocks. It is designed to capture complex patterns within the data through repeated application of mixing operations.

    Attributes:
        patch_size (Tuple[int, int]): Size of the patches the image is divided into.
        num_layers (int): Number of MixerBlocks in the encoder.
        hidden_dim (int): Dimensionality of the hidden features.
        num_heads (int): Number of attention heads (not directly used in MixerBlocks but kept for interface consistency).
        feedforward_dim (int): Dimensionality of the feedforward network within the MixerBlock.
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

        self.layers = [MixerBlock() for _ in range(self.num_layers)]

        self.dropout_layer = nn.Dropout(self.dropout)

    def __call__(self, x: jnp.ndarray, training: bool = False) -> tuple:

        x = self.embedding(x)
        for layer in self.layers:
            x = layer(x)
            self.dropout_layer(x, deterministic=not training)
        return x


class Mixer(nn.Module):
    """
    Implements the MLP-Mixer model for image classification tasks.

    The MLP-Mixer model uses an architecture based entirely on multi-layer perceptrons (MLPs), avoiding conventional convolution or attention mechanisms. It divides the image into patches and applies MixerBlocks to mix information across spatial and channel dimensions.

    Attributes:
        patch_size (Tuple[int, int]): Size of the patches the image is divided into.
        num_layers (int): Number of MixerBlocks in the model.
        hidden_dim (int): Dimensionality of the hidden features.
        num_heads (int): Number of attention heads (not directly used in this model but kept for interface consistency).
        feedforward_dim (int): Dimensionality of the feedforward network within the MixerBlock.
        dropout (float): Dropout rate for regularization.
        n_outputs (int): Number of output classes.

    MLP Mixers are a recent architectural innovation in the field of deep learning, introduced to address the limitations of traditional Convolutional Neural Networks (CNNs) and Transformers.
    The motivation behind MLP Mixers arises from the need to handle diverse data types and leverage multi-modal information efficiently. Unlike transformers that rely on self-attention mechanisms,
    MLP Mixers employ a simple yet powerful approach using Multi-Layer Perceptrons (MLPs) to process data. This architecture is designed to work with sequences, images, or even a combination of both,
    making it versatile for a wide range of tasks. MLP Mixers have demonstrated strong performance in various applications, including image classification, natural language understanding, and cross-modal learning,
    showcasing their potential in handling different modalities and promoting model efficiency and scalability in deep learning.

    Example usage:
        ```
        import jax
        import jax.numpy as jnp
        from nanodl import ArrayDataset, DataLoader
        from nanodl import Mixer, MixerDataParallelTrainer

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
        model = Mixer(**hyperparams)
        rngs = {'params': jax.random.key(0), 'dropout': jax.random.key(1)}
        params = model.init(rngs, dummy_inputs)['params']
        outputs = model.apply({'params': params}, dummy_inputs, rngs=rngs)[0]
        print(outputs.shape)

        # Training on your data
        trainer = MixerDataParallelTrainer(model, dummy_inputs.shape, 'params.pkl')
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
        self.encoder = MixerEncoder(
            patch_size=self.patch_size,
            num_layers=self.num_layers,
            hidden_dim=self.hidden_dim,
            num_heads=self.num_heads,
            feedforward_dim=self.feedforward_dim,
            dropout=self.dropout,
        )
        self.dropout_layer = nn.Dropout(self.dropout)
        self.output = nn.Dense(self.n_outputs)

    def __call__(self, x: jnp.ndarray, training: bool = False) -> tuple:
        x = self.encoder(x=x, training=training)
        x = self.dropout_layer(x, deterministic=not training)
        return self.output(x[:, 0, :]), x


class MixerDataParallelTrainer:
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
            MixerDataParallelTrainer.train_step, axis_name="devices"
        )
        self.evaluation_step = jax.pmap(
            MixerDataParallelTrainer.evaluation_step, axis_name="devices"
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
