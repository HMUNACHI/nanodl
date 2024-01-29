"""

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

import jax
import flax
import time
import optax
import jax.numpy as jnp
import flax.linen as nn

from flax.training import train_state
from typing import Any, Iterable, Optional, Tuple, Dict, List


class SinusoidalEmbedding(nn.Module):
    """
    Sinusoidal Embedding for images.

    This class generates sinusoidal embeddings for a given input tensor. The embeddings are
    created using a range of frequencies determined by the minimum and maximum frequency parameters.

    Attributes:
        embedding_dims (int): The dimensionality of the embedding.
        embedding_min_frequency (float): The minimum frequency for the sinusoidal embedding.
        embedding_max_frequency (float): The maximum frequency for the sinusoidal embedding.
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
        embeddings = jnp.concatenate([jnp.sin(self.angular_speeds * x), jnp.cos(self.angular_speeds * x)], axis=-1)
        return embeddings
    

class UNetResidualBlock(nn.Module):
    width: int

    @nn.compact
    def __call__(self, 
                 x: jnp.ndarray) -> jnp.ndarray:
        """
        Applies a Residual Block to the input tensor.

        Args:
            x (jax.numpy.ndarray): The input tensor.

        Returns:
            jax.numpy.ndarray: The output tensor after applying the residual block.
        """
        input_width = x.shape[-1]

        # Define layers
        convolution_1 = nn.Conv(self.width, kernel_size=(1, 1))
        convolution_2 = nn.Conv(self.width, kernel_size=(3, 3), padding='SAME')
        convolution_3 = nn.Conv(self.width, kernel_size=(3, 3), padding='SAME')
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
    Downsampling block for U-Net architecture.

    This block applies a series of residual blocks followed by average pooling to downsample the input.

    Attributes:
        width (int): The number of channels in the residual blocks.
        block_depth (int): The number of residual blocks in the down block.
    """
    width: int
    block_depth: int

    def setup(self):
        self.residual_blocks = [UNetResidualBlock(self.width) for _ in range(self.block_depth)]

    def __call__(self, 
                 x: jnp.ndarray) -> jnp.ndarray:
        """
        Applies the downsampling block to the input tensor.

        Args:
            x (jax.numpy.ndarray): The input tensor.

        Returns:
            jax.numpy.ndarray: The downsampled output tensor.
        """
        for block in self.residual_blocks:
            x = block(x)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
        return x


class UNetUpBlock(nn.Module):
    """
    Upsampling block for U-Net architecture.

    This block applies bilinear upsampling to the input and concatenates it with a skip connection.
    It then applies a series of residual blocks to the concatenated tensor.

    Attributes:
        width (int): The number of channels in the residual blocks.
        block_depth (int): The number of residual blocks in the up block.
    """
    width: int
    block_depth: int

    def setup(self):
        self.residual_blocks = [UNetResidualBlock(self.width) for _ in range(self.block_depth)]

    def __call__(self, 
                 x: jnp.ndarray, 
                 skip: jnp.ndarray) -> jnp.ndarray:
        """
        Applies the upsampling block to the input tensor, concatenating it with the skip connection.

        Args:
            x (jax.numpy.ndarray): The input tensor to be upsampled.
            skip (jax.numpy.ndarray): The skip connection tensor to be concatenated with the input.

        Returns:
            jax.numpy.ndarray: The upsampled and concatenated output tensor.
        """
        B, H, W, C = x.shape
        upsampled_shape = (B, H * 2, W * 2, C)
        x = jax.image.resize(x, shape=upsampled_shape, method='bilinear')
        x = jnp.concatenate([x, skip], axis=-1)
        for block in self.residual_blocks:
            x = block(x)
        return x


class UNet(nn.Module):
    """
    U-Net architecture for image generation.

    This class implements a U-Net model, which is commonly used for image-to-image translation tasks. 
    It consists of an encoder (downsampling path), a bottleneck, and a decoder (upsampling path) 
    with skip connections.

    Attributes:
        image_size (Tuple[int, int]): The size of the input images.
        widths (List[int]): The number of channels in each block of the U-Net.
        block_depth (int): The depth of each block in the U-Net.
        embed_dims (int): The number of dimensions for the sinusoidal embedding.
        embed_min_freq (float): The minimum frequency for the sinusoidal embedding.
        embed_max_freq (float): The maximum frequency for the sinusoidal embedding.
    """
    image_size: Tuple[int, int]
    widths: List[int]
    block_depth: int
    embed_dims: int
    embed_min_freq: float
    embed_max_freq: float

    def setup(self):
        self.sinusoidal_embedding = SinusoidalEmbedding(self.embed_dims, self.embed_min_freq, self.embed_max_freq)
        self.down_blocks = [UNetDownBlock(width, self.block_depth) for width in self.widths[:-1]]
        self.residual_blocks = [UNetResidualBlock(self.widths[-1]) for _ in range(self.block_depth)]
        self.up_blocks = [UNetUpBlock(width, self.block_depth) for width in reversed(self.widths[:-1])]
        self.convolution_1 = nn.Conv(self.widths[0], kernel_size=(1, 1))
        self.convolution_2 = nn.Conv(3, kernel_size=(1, 1), kernel_init=nn.initializers.zeros)

    def __call__(self, 
                 noisy_images: jnp.ndarray, 
                 noise_variances: jnp.ndarray) -> jnp.ndarray:
        """
        Applies the U-Net model to the input images.

        Args:
            noisy_images (jax.numpy.ndarray): The input images to the U-Net.
            noise_variances (jax.numpy.ndarray): The noise variances for sinusoidal embedding.

        Returns:
            jax.numpy.ndarray: The output images generated by the U-Net.
        """
        e = self.sinusoidal_embedding(noise_variances)
        upsampled_shape = (noisy_images.shape[0], self.image_size[0], self.image_size[1], self.embed_dims)
        e = jax.image.resize(e, upsampled_shape, method='nearest')

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
    Image Generating Diffusion Model.

    This class implements a diffusion model for image generation.

    Attributes:
        image_size (int): The size of the input images.
        widths (List[int]): The number of channels in each block of the U-Net.
        block_depth (int): The depth of each block in the U-Net.
        min_signal_rate (float): Minimum signal rate for the diffusion process.
        max_signal_rate (float): Maximum signal rate for the diffusion process.
        ema (float): Exponential moving average rate for model weights.
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
        self.unet = UNet(image_size=(self.image_size, self.image_size),
                         widths=self.widths,
                         block_depth=self.block_depth,
                         embed_dims=self.embed_dims,
                         embed_min_freq=self.embed_min_freq,
                         embed_max_freq=self.embed_max_freq)

    def diffusion_schedule(self, 
                           diffusion_times: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        start_angle = jnp.arccos(self.max_signal_rate)
        end_angle = jnp.arccos(self.min_signal_rate)
        diffusion_angles = start_angle + diffusion_times * (end_angle - start_angle)
        signal_rates = jnp.cos(diffusion_angles)
        noise_rates = jnp.sin(diffusion_angles)
        return noise_rates, signal_rates

    def denoise(self, 
                noisy_images: jnp.ndarray, 
                noise_rates: jnp.ndarray, 
                signal_rates: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        pred_noises = self.unet(noisy_images, noise_rates ** 2)
        pred_images = (noisy_images - noise_rates * pred_noises) / signal_rates
        return pred_noises, pred_images

    def __call__(self, 
                 images: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        key = jax.random.PRNGKey(int(time.time()))
        noises = jax.random.normal(key, shape=(images.shape[0], self.image_size, self.image_size, 3))
        batch_size = images.shape[0]
        diffusion_times = jax.random.uniform(key, shape=(batch_size, 1, 1, 1), minval=0.0, maxval=1.0)
        noise_rates, signal_rates = self.diffusion_schedule(diffusion_times)
        noisy_images = signal_rates * images + noise_rates * noises
        pred_noises, pred_images = self.denoise(noisy_images, noise_rates, signal_rates)
        return pred_noises, pred_images

    def reverse_diffusion(self, 
                          initial_noise: jnp.ndarray, 
                          diffusion_steps: int) -> jnp.ndarray:
        """
        Performs reverse diffusion to generate images from noise.

        Args:
            initial_noise (jax.numpy.ndarray): The initial noise tensor.
            diffusion_steps (int): The number of diffusion steps.

        Returns:
            jax.numpy.ndarray: The generated images.
        """
        num_images = initial_noise.shape[0]
        step_size = 1.0 / diffusion_steps
        next_noisy_images = initial_noise

        for step in range(diffusion_steps):
            diffusion_times = jnp.ones((num_images, 1, 1, 1)) - step * step_size
            noise_rates, signal_rates = self.diffusion_schedule(diffusion_times)
            pred_noises, pred_images = self.denoise(next_noisy_images, noise_rates, signal_rates)
            next_diffusion_times = diffusion_times - step_size
            next_noise_rates, next_signal_rates = self.diffusion_schedule(next_diffusion_times)
            next_noisy_images = (next_signal_rates * pred_images + next_noise_rates * pred_noises)

        return pred_images
    
    def generate(self, 
                 num_images: int = 1, 
                 diffusion_steps: int = 20) -> jnp.ndarray:
        """
        Generates images using the diffusion model.

        Args:
            num_images (int): The number of images to generate.
            diffusion_steps (int): The number of diffusion steps for image generation.

        Returns:
            jax.numpy.ndarray: The generated images.
        """
        key = jax.random.PRNGKey(int(time.time()))
        noises = jax.random.normal(key, shape=(num_images, 
                                               self.image_size, 
                                               self.image_size, 
                                               3))
        
        return self.reverse_diffusion(noises, diffusion_steps)


class DiffusionDataParallelTrainer:
    """
    A class for training a GPT model using data parallelism.

    Attributes:
        model: The GPT model to be trained.
        num_parameters: The number of parameters in the model.
        best_val_loss: The best validation loss achieved during training.
        weights_filename: Filename for saving the model weights.
        num_devices: Number of local devices (GPUs/TPUs) used for parallel training.
        state: The current state of the model, including parameters and optimizer state.
    """
    def __init__(self, 
                 model: Any, 
                 input_shape: Tuple[int, ...],
                 weights_filename: str,
                 learning_rate: float = 1e-4,
                 params_path: Optional[str] = None) -> None:
        self.model = model
        self.params = None
        self.params_path = params_path
        self.num_parameters = None
        self.best_val_loss = float("inf")
        self.weights_filename = weights_filename
        self.num_devices = jax.local_device_count()
        self.train_step = jax.pmap(DiffusionDataParallelTrainer.train_step, axis_name='devices')
        self.evaluation_step = jax.pmap(DiffusionDataParallelTrainer.evaluation_step, axis_name='devices')
        self.state = self.create_train_state(learning_rate, input_shape)
        print(f'Number of accelerators: {self.num_devices}')
    

    def create_train_state(self, 
                           learning_rate: float, 
                           input_shape: Tuple[int, ...]) -> Any:
        """
        Creates and initializes the training state for the model.

        Args:
            learning_rate: The learning rate for the optimizer.
            text_input_shape: The shape of the text input.
            image_input_shape: The shape of the image input.

        Returns:
            The initialized training state.
        """
        rngs = {'params': jax.random.key(0), 'dropout': jax.random.key(1)}
        params = self.model.init(rngs, jnp.ones(input_shape))['params']

        if self.params_path is not None:
            params = self.load_params(self.params_path)

        self.num_parameters = sum(param.size for param in jax.tree_util.tree_leaves(params))
        print(f'Number of parameters: {self.num_parameters}')
        state = train_state.TrainState.create(apply_fn=self.model.apply, 
                                              params=params, 
                                              tx=optax.adam(learning_rate))
        return jax.device_put_replicated(state, jax.local_devices())
    
    @staticmethod
    def train_step(state: Any, 
                   images: jnp.ndarray) -> Tuple[Any, jnp.ndarray]:
        """
        Performs a single training step.

        Args:
            state: The current state of the model, including parameters and optimizer state.
            batch: A dictionary containing 'inputs' and 'targets' as keys, representing the input data.

        Returns:
            A tuple of the updated state and the loss value for this step.
        """
        def loss_fn(params):
            key = jax.random.PRNGKey(int(time.time()))
            noises = jax.random.normal(key, shape=images.shape)
            pred_noises, pred_images = state.apply_fn({'params': params}, 
                                      images,
                                      rngs={'dropout': jax.random.PRNGKey(int(time.time()))})
            return jnp.mean(jnp.square(pred_noises - noises)) + jnp.mean(jnp.square(pred_images - images))
        
        loss, grads = jax.value_and_grad(loss_fn)(state.params)
        state = state.apply_gradients(grads=grads)
        return state, loss

    def train(self, 
              train_loader: Iterable[Tuple[jnp.ndarray, jnp.ndarray]], 
              num_epochs: int, 
              val_loader: Optional[Iterable[Tuple[jnp.ndarray, jnp.ndarray]]] = None) -> None:
        """
        Trains the model for a specified number of epochs.

        Args:
            train_loader: An iterable of training data batches.
            num_epochs: The number of epochs to train for.
            val_loader: An optional iterable of validation data batches.
        """
        for epoch in range(num_epochs):
            total_loss = 0.0
            count = 0
            for images in train_loader:
                images = images[0] if len(images) == 1 else images
                batch_size = images.shape[0]
                batch_size_per_device = batch_size // self.num_devices
                images = images.reshape((self.num_devices, 
                                         batch_size_per_device, 
                                         images.shape[1], 
                                         images.shape[2], 
                                         images.shape[3]))
                self.state, loss = self.train_step(state=self.state, 
                                                   images=images)
                total_loss += jnp.mean(loss)
                count += 1
            
            mean_loss = total_loss / count
            print(f'Epoch {epoch+1}, Train Loss: {mean_loss}')

            if val_loader is not None:
                val_loss = self.evaluate(val_loader)
                print(f'Epoch {epoch+1}, Val Loss: {val_loss}')
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                print("New best validation score achieved, saving model...")
                self.save_params()
        return 
    
    @staticmethod
    def evaluation_step(state: Any, 
                        images: jnp.ndarray) -> Tuple[Any, jnp.ndarray]:
        """
        Performs a single training step.

        Args:
            state: The current state of the model, including parameters and optimizer state.
            batch: A dictionary containing 'inputs' and 'targets' as keys, representing the input data.

        Returns:
            A tuple of the updated state and the loss value for this step.
        """
        key = jax.random.PRNGKey(int(time.time()))
        noises = jax.random.normal(key, shape=images.shape)
        pred_noises, pred_images = state.apply_fn({'params': state.params}, 
                                                  images,  
                                                  rngs={'dropout': jax.random.PRNGKey(int(time.time()))})
        return jnp.mean(jnp.square(pred_noises - noises)) + jnp.mean(jnp.square(pred_images - images))

    def evaluate(self, 
                 test_loader: Iterable[Tuple[jnp.ndarray, jnp.ndarray]]) -> None:
        """
        evaluates the model using the provided validation loader.

        Args:
            val_loader: An iterable of validation data batches.
            epoch: The current epoch number.
            num_epochs: The total number of epochs.
        """
        total_loss = 0.0
        count = 0
        for images in test_loader:
            images = images[0] if len(images) == 1 else images
            batch_size = images.shape[0]
            batch_size_per_device = batch_size // self.num_devices
            images = images.reshape((self.num_devices, 
                                        batch_size_per_device, 
                                        images.shape[1], 
                                        images.shape[2], 
                                        images.shape[3]))
            loss = self.evaluation_step(self.state, images)
            total_loss += jnp.mean(loss)
            count += 1
        
        mean_loss = total_loss / count
        return mean_loss
    
    def get_ema_weights(self, params, ema=0.999):
        """
        Multiplies all values in a parameters dictionary by 2.

        Args:
            params (dict): A dictionary containing parameters.

        Returns:
            dict: A new dictionary with all values multiplied by 2.
        """
        new_params = {}
        for key, value in params.items():
            if isinstance(value, dict):
                # Recursively apply the function to nested dictionaries
                new_params[key] = self.get_ema_weights(value, ema)
            else:
                # Multiply the value by ema multiplier
                new_params[key] = ema * value + (1 - ema) * value
        return new_params

    def save_params(self) -> None:
        """
        Saves the unreplicated model parameters to a file.
        """
        self.params = flax.jax_utils.unreplicate(self.state.params)
        self.params = self.get_ema_weights(self.params)
        with open(self.weights_filename, 'wb') as f:
            f.write(flax.serialization.to_bytes(self.params))

    def load_params(self, filename: str):
        """
        Loads the model parameters from a file
        """
        with open(filename, 'rb') as f:
            self.params = flax.serialization.from_bytes(self.params, f.read())
        return self.params