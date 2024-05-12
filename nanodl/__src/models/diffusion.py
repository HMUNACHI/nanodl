import time
from typing import Any, Iterable, List, Optional, Tuple

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import optax
from flax.training import train_state


class SinusoidalEmbedding(nn.Module):
    """
    Implements sinusoidal embeddings as a layer in a neural network using JAX.

    This layer generates sinusoidal embeddings based on input positions and a range of frequencies, producing embeddings that capture positional information in a continuous manner. It's particularly useful in models where the notion of position is crucial, such as in generative models for images and audio.

    Attributes:
        embedding_dims (int): The dimensionality of the output embeddings.
        embedding_min_frequency (float): The minimum frequency used in the sinusoidal embedding.
        embedding_max_frequency (float): The maximum frequency used in the sinusoidal embedding.

    """

    embedding_dims: int
    embedding_min_frequency: float
    embedding_max_frequency: float

    def setup(self):
        num = self.embedding_dims // 2
        start = jnp.log(self.embedding_min_frequency)
        stop = jnp.log(self.embedding_max_frequency)
        frequencies = jnp.exp(jnp.linspace(start, stop, num))
        self.angular_speeds = 2.0 * jnp.pi * frequencies

    def __call__(self, x):
        embeddings = jnp.concatenate(
            [jnp.sin(self.angular_speeds * x), jnp.cos(self.angular_speeds * x)],
            axis=-1,
        )
        return embeddings


class UNetResidualBlock(nn.Module):
    """
    Implements a residual block within a U-Net architecture using JAX.

    This module defines a residual block with convolutional layers and normalization, followed by a residual connection. It's a fundamental building block in constructing deeper and more complex U-Net architectures for tasks like image segmentation and generation.

    Attributes:
        width (int): The number of output channels for the convolutional layers within the block.

    """

    width: int

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        input_width = x.shape[-1]

        # Define layers
        convolution_1 = nn.Conv(self.width, kernel_size=(1, 1))
        convolution_2 = nn.Conv(self.width, kernel_size=(3, 3), padding="SAME")
        convolution_3 = nn.Conv(self.width, kernel_size=(3, 3), padding="SAME")
        norm = nn.GroupNorm(num_groups=2, epsilon=1e-5, use_bias=False, use_scale=False)

        # Residual connection
        residual = convolution_1(x) if input_width != self.width else x

        # Forward pass
        x = norm(x)
        x = nn.swish(x)
        x = convolution_2(x)
        x = nn.swish(x)
        x = convolution_3(x)

        return x + residual


class UNetDownBlock(nn.Module):
    """
    Implements a down-sampling block in a U-Net architecture using JAX.

    This module consists of a sequence of residual blocks followed by an average pooling operation to reduce the spatial dimensions. It's used to capture higher-level features at reduced spatial resolutions in the encoding pathway of a U-Net.

    Attributes:
        width (int): The number of output channels for the convolutional layers within the block.
        block_depth (int): The number of residual blocks to include in the down-sampling block.

    """

    width: int
    block_depth: int

    def setup(self):
        self.residual_blocks = [
            UNetResidualBlock(self.width) for _ in range(self.block_depth)
        ]

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        for block in self.residual_blocks:
            x = block(x)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
        return x


class UNetUpBlock(nn.Module):
    """
    Implements an up-sampling block in a U-Net architecture using JAX.

    This module consists of a sequence of residual blocks and a bilinear up-sampling operation to increase the spatial dimensions. It's used in the decoding pathway of a U-Net to progressively recover spatial resolution and detail in the output image.

    Attributes:
        width (int): The number of output channels for the convolutional layers within the block.
        block_depth (int): The number of residual blocks to include in the up-sampling block.

    """

    width: int
    block_depth: int

    def setup(self):
        self.residual_blocks = [
            UNetResidualBlock(self.width) for _ in range(self.block_depth)
        ]

    def __call__(self, x: jnp.ndarray, skip: jnp.ndarray) -> jnp.ndarray:
        B, H, W, C = x.shape
        upsampled_shape = (B, H * 2, W * 2, C)
        x = jax.image.resize(x, shape=upsampled_shape, method="bilinear")
        x = jnp.concatenate([x, skip], axis=-1)
        for block in self.residual_blocks:
            x = block(x)
        return x


class UNet(nn.Module):
    """
    Implements the U-Net architecture for image processing tasks using JAX.

    This model is widely used for tasks such as image segmentation, denoising, and super-resolution. It features a symmetric encoder-decoder structure with skip connections between corresponding layers in the encoder and decoder to preserve spatial information.

    Attributes:
        image_size (Tuple[int, int]): The size of the input images (height, width).
        widths (List[int]): The number of output channels for each block in the U-Net architecture.
        block_depth (int): The number of residual blocks in each down-sampling and up-sampling block.
        embed_dims (int): The dimensionality of the sinusoidal embeddings for encoding positional information.
        embed_min_freq (float): The minimum frequency for the sinusoidal embeddings.
        embed_max_freq (float): The maximum frequency for the sinusoidal embeddings.

    """

    image_size: Tuple[int, int]
    widths: List[int]
    block_depth: int
    embed_dims: int
    embed_min_freq: float
    embed_max_freq: float

    def setup(self):
        self.sinusoidal_embedding = SinusoidalEmbedding(
            self.embed_dims, self.embed_min_freq, self.embed_max_freq
        )
        self.down_blocks = [
            UNetDownBlock(width, self.block_depth) for width in self.widths[:-1]
        ]
        self.residual_blocks = [
            UNetResidualBlock(self.widths[-1]) for _ in range(self.block_depth)
        ]
        self.up_blocks = [
            UNetUpBlock(width, self.block_depth) for width in reversed(self.widths[:-1])
        ]
        self.convolution_1 = nn.Conv(self.widths[0], kernel_size=(1, 1))
        self.convolution_2 = nn.Conv(
            3, kernel_size=(1, 1), kernel_init=nn.initializers.zeros
        )

    def __call__(
        self, noisy_images: jnp.ndarray, noise_variances: jnp.ndarray
    ) -> jnp.ndarray:

        e = self.sinusoidal_embedding(noise_variances)
        upsampled_shape = (
            noisy_images.shape[0],
            self.image_size[0],
            self.image_size[1],
            self.embed_dims,
        )
        e = jax.image.resize(e, upsampled_shape, method="nearest")

        x = self.convolution_1(noisy_images)
        x = jnp.concatenate([x, e], axis=-1)

        skips = []
        for block in self.down_blocks:
            skips.append(x)
            x = block(x)

        for block in self.residual_blocks:
            x = block(x)

        for block, skip in zip(self.up_blocks, reversed(skips)):
            x = block(x, skip)

        outputs = self.convolution_2(x)
        return outputs


class DiffusionModel(nn.Module):
    """
    Implements a diffusion model for image generation using JAX.

    Diffusion models are a class of generative models that learn to denoise images through a gradual process of adding and removing noise. This implementation uses a U-Net architecture for the denoising process and supports custom diffusion schedules.

    Attributes:
        image_size (int): The size of the generated images.
        widths (List[int]): The number of output channels for each block in the U-Net architecture.
        block_depth (int): The number of residual blocks in each down-sampling and up-sampling block.
        min_signal_rate (float): The minimum signal rate in the diffusion process.
        max_signal_rate (float): The maximum signal rate in the diffusion process.
        embed_dims (int): The dimensionality of the sinusoidal embeddings for encoding noise levels.
        embed_min_freq (float): The minimum frequency for the sinusoidal embeddings.
        embed_max_freq (float): The maximum frequency for the sinusoidal embeddings.

    Methods:
        diffusion_schedule(diffusion_times: jnp.ndarray): Computes the noise and signal rates for given diffusion times.
        denoise(noisy_images: jnp.ndarray, noise_rates: jnp.ndarray, signal_rates: jnp.ndarray): Denoises images given their noise and signal rates.
        reverse_diffusion(initial_noise: jnp.ndarray, diffusion_steps: int): Reverses the diffusion process to generate images from noise.
        generate(num_images: int, diffusion_steps: int): Generates images by reversing the diffusion process from random noise.

    Example usage:
        ```
        import jax
        import jax.numpy as jnp
        from nanodl import ArrayDataset, DataLoader
        from nanodl import DiffusionModel, DiffusionDataParallelTrainer

        image_size = 32
        block_depth = 2
        batch_size = 8
        widths = [32, 64, 128]
        key = jax.random.PRNGKey(0)
        input_shape = (101, image_size, image_size, 3)
        images = jax.random.normal(key, input_shape)

        # Use your own images
        dataset = ArrayDataset(images)
        dataloader = DataLoader(dataset,
                                batch_size=batch_size,
                                shuffle=True,
                                drop_last=False)

        # Create diffusion model
        diffusion_model = DiffusionModel(image_size, widths, block_depth)
        params = diffusion_model.init(key, images)
        pred_noises, pred_images = diffusion_model.apply(params, images)
        print(pred_noises.shape, pred_images.shape)

        # Training on your data
        # Note: saved params are often different from training weights, use the saved params for generation
        trainer = DiffusionDataParallelTrainer(diffusion_model,
                                            input_shape=images.shape,
                                            weights_filename='params.pkl',
                                            learning_rate=1e-4)
        trainer.train(dataloader, 10, dataloader)
        print(trainer.evaluate(dataloader))

        # Generate some samples
        params = trainer.load_params('params.pkl')
        generated_images = diffusion_model.apply({'params': params},
                                                num_images=5,
                                                diffusion_steps=5,
                                                method=diffusion_model.generate)
        print(generated_images.shape)
        ```
    """

    image_size: int
    widths: List[int]
    block_depth: int
    min_signal_rate: float = 0.02
    max_signal_rate: float = 0.95
    embed_dims: int = 64
    embed_min_freq: float = 1.0
    embed_max_freq: float = 1000.0

    def setup(self):
        self.unet = UNet(
            image_size=(self.image_size, self.image_size),
            widths=self.widths,
            block_depth=self.block_depth,
            embed_dims=self.embed_dims,
            embed_min_freq=self.embed_min_freq,
            embed_max_freq=self.embed_max_freq,
        )

    def diffusion_schedule(
        self, diffusion_times: jnp.ndarray
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        start_angle = jnp.arccos(self.max_signal_rate)
        end_angle = jnp.arccos(self.min_signal_rate)
        diffusion_angles = start_angle + diffusion_times * (end_angle - start_angle)
        signal_rates = jnp.cos(diffusion_angles)
        noise_rates = jnp.sin(diffusion_angles)
        return noise_rates, signal_rates

    def denoise(
        self,
        noisy_images: jnp.ndarray,
        noise_rates: jnp.ndarray,
        signal_rates: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        pred_noises = self.unet(noisy_images, noise_rates**2)
        pred_images = (noisy_images - noise_rates * pred_noises) / signal_rates
        return pred_noises, pred_images

    def __call__(self, images: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        key = jax.random.PRNGKey(int(time.time()))
        noises = jax.random.normal(
            key, shape=(images.shape[0], self.image_size, self.image_size, 3)
        )
        batch_size = images.shape[0]
        diffusion_times = jax.random.uniform(
            key, shape=(batch_size, 1, 1, 1), minval=0.0, maxval=1.0
        )
        noise_rates, signal_rates = self.diffusion_schedule(diffusion_times)
        noisy_images = signal_rates * images + noise_rates * noises
        pred_noises, pred_images = self.denoise(noisy_images, noise_rates, signal_rates)
        return pred_noises, pred_images

    def reverse_diffusion(
        self, initial_noise: jnp.ndarray, diffusion_steps: int
    ) -> jnp.ndarray:

        num_images = initial_noise.shape[0]
        step_size = 1.0 / diffusion_steps
        next_noisy_images = initial_noise

        for step in range(diffusion_steps):
            diffusion_times = jnp.ones((num_images, 1, 1, 1)) - step * step_size
            noise_rates, signal_rates = self.diffusion_schedule(diffusion_times)
            pred_noises, pred_images = self.denoise(
                next_noisy_images, noise_rates, signal_rates
            )
            next_diffusion_times = diffusion_times - step_size
            next_noise_rates, next_signal_rates = self.diffusion_schedule(
                next_diffusion_times
            )
            next_noisy_images = (
                next_signal_rates * pred_images + next_noise_rates * pred_noises
            )

        return pred_images

    def generate(self, num_images: int = 1, diffusion_steps: int = 20) -> jnp.ndarray:

        key = jax.random.PRNGKey(int(time.time()))
        noises = jax.random.normal(
            key, shape=(num_images, self.image_size, self.image_size, 3)
        )

        return self.reverse_diffusion(noises, diffusion_steps)


class DiffusionDataParallelTrainer:
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
        learning_rate: float = 1e-4,
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
            DiffusionDataParallelTrainer.train_step, axis_name="devices"
        )
        self.evaluation_step = jax.pmap(
            DiffusionDataParallelTrainer.evaluation_step, axis_name="devices"
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
    def train_step(state: Any, images: jnp.ndarray) -> Tuple[Any, jnp.ndarray]:

        def loss_fn(params):
            key = jax.random.PRNGKey(int(time.time()))
            noises = jax.random.normal(key, shape=images.shape)
            pred_noises, pred_images = state.apply_fn(
                {"params": params},
                images,
                rngs={"dropout": jax.random.PRNGKey(int(time.time()))},
            )
            return jnp.mean(jnp.square(pred_noises - noises)) + jnp.mean(
                jnp.square(pred_images - images)
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
            for images in train_loader:
                images = images[0] if len(images) == 1 else images
                batch_size = images.shape[0]
                batch_size_per_device = batch_size // self.num_devices
                images = images.reshape(
                    (
                        self.num_devices,
                        batch_size_per_device,
                        images.shape[1],
                        images.shape[2],
                        images.shape[3],
                    )
                )
                self.state, loss = self.train_step(state=self.state, images=images)
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
    def evaluation_step(state: Any, images: jnp.ndarray) -> Tuple[Any, jnp.ndarray]:

        key = jax.random.PRNGKey(int(time.time()))
        noises = jax.random.normal(key, shape=images.shape)
        pred_noises, pred_images = state.apply_fn(
            {"params": state.params},
            images,
            rngs={"dropout": jax.random.PRNGKey(int(time.time()))},
        )
        return jnp.mean(jnp.square(pred_noises - noises)) + jnp.mean(
            jnp.square(pred_images - images)
        )

    def evaluate(self, test_loader: Iterable[Tuple[jnp.ndarray, jnp.ndarray]]) -> None:

        total_loss = 0.0
        count = 0
        for images in test_loader:
            images = images[0] if len(images) == 1 else images
            batch_size = images.shape[0]
            batch_size_per_device = batch_size // self.num_devices
            images = images.reshape(
                (
                    self.num_devices,
                    batch_size_per_device,
                    images.shape[1],
                    images.shape[2],
                    images.shape[3],
                )
            )
            loss = self.evaluation_step(self.state, images)
            total_loss += jnp.mean(loss)
            count += 1

        mean_loss = total_loss / count
        return mean_loss

    def get_ema_weights(self, params, ema=0.999):
        def func(x):
            return x * ema + (1 - ema) * x

        return jax.tree_util.tree_map(func, params)

    def save_params(self) -> None:
        self.params = flax.jax_utils.unreplicate(self.state.params)
        self.params = self.get_ema_weights(self.params)
        with open(self.weights_filename, "wb") as f:
            f.write(flax.serialization.to_bytes(self.params))

    def load_params(self, filename: str):
        with open(filename, "rb") as f:
            self.params = flax.serialization.from_bytes(self.params, f.read())
        return self.params
