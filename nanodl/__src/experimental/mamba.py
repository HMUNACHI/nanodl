import math
import time
from typing import Any, Iterable, Optional, Tuple

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import optax
from einops import einsum
from flax.training import train_state


class MambaBlock(nn.Module):
    """
    MambaBlock is a custom neural network block that incorporates normalization,
    convolution, and dense layers to process input sequences. This block is designed
    for sequence modeling tasks and includes specialized components like selective
    scan for dynamic computation.

    Attributes:
        d_inner (int): Dimensionality of the inner dense layer.
        d_conv (int): Size of the convolution kernel.
        dt_rank (int): Rank for delta transformations in the selective scan.
        d_state (int): Dimensionality of the state vector in the selective scan.
        d_model (int): Dimensionality of the input and output of the block.
        seq_len (int): Length of the input sequences.
        bias (bool): Flag indicating whether to use bias in dense layers.
        conv_bias (bool): Flag indicating whether to use bias in the convolution layer.
    """

    d_inner: int
    d_conv: int
    dt_rank: int
    d_state: int
    d_model: int
    seq_len: int
    bias: bool
    conv_bias: bool

    def setup(self):
        self.norm = nn.RMSNorm(self.d_model)
        self.in_proj = nn.Dense(features=self.d_inner * 2, use_bias=self.bias)

        self.conv1d = nn.Conv(
            features=self.seq_len,
            kernel_size=(self.d_conv,),
            strides=(1,),
            padding="SAME",
            use_bias=self.conv_bias,
            feature_group_count=self.d_inner,
        )

        self.x_proj = nn.Dense(features=self.dt_rank + self.d_state * 2, use_bias=False)
        self.dt_proj = nn.Dense(features=self.d_inner, use_bias=True)
        self.out_proj = nn.Dense(features=self.d_model, use_bias=self.bias)

        # Parameter initialization
        A = jnp.tile(jnp.arange(1, self.d_state + 1), (self.d_inner, 1))
        self.A_log = self.variable("params", "A_log", lambda: jnp.log(A))
        self.D = self.variable("params", "D", lambda: jnp.ones((self.d_inner,)))

    def __call__(self, inputs: jnp.ndarray):
        u = self.norm(inputs)
        A = -jnp.exp(self.A_log.value)
        D = self.D.value
        x_and_res = self.in_proj(u)
        x, res = jnp.split(x_and_res, 2, axis=-1)
        x = jnp.transpose(x, (0, 2, 1))
        x = self.conv1d(x)[:, :, : u.shape[1]]
        x = jnp.transpose(x, (0, 2, 1))
        x = nn.silu(x)

        x_dbl = self.x_proj(u)
        delta, B, C = jnp.split(
            x_dbl,
            indices_or_sections=[self.dt_rank, self.dt_rank + self.d_state],
            axis=-1,
        )
        delta = nn.softplus(self.dt_proj(delta))
        y = self.selective_scan(x, delta, A, B, C, D)
        y = y * nn.silu(res)
        return self.out_proj(y) + inputs

    def selective_scan(
        self,
        u: jnp.ndarray,
        delta: jnp.ndarray,
        A: jnp.ndarray,
        B: jnp.ndarray,
        C: jnp.ndarray,
        D: jnp.ndarray,
    ) -> jnp.ndarray:

        b, l, d_in = u.shape
        n = A.shape[1]

        deltaA = jnp.exp(einsum(delta, A, "b l d_in, d_in n -> b l d_in n"))

        deltaB_u = einsum(delta, B, u, "b l d_in, b l n, b l d_in -> b l d_in n")

        x = jnp.zeros((b, d_in, n))
        ys = []

        for i in range(l):
            x = deltaA[:, i] * x + deltaB_u[:, i]
            y = einsum(x, C[:, i, :], "b d_in n, b n -> b d_in")
            ys.append(y)

        return jnp.stack(ys, axis=1) + u * D


class Mamba(nn.Module):
    """
    MAMBA is an advanced ML model renowned for its exceptional linear-time processing efficiency,
    which notably enhances its inference speed to outperform traditional Transformer models by up to five times in throughput.
    Unlike conventional models that struggle with long sequence lengths, MAMBA demonstrates a linear scalability with sequence length,
    maintaining or even improving its performance with sequences that extend up to a million elements.
    This attribute makes MAMBA a highly versatile and efficient backbone for a variety of sequence modeling tasks across different domains,
    including but not limited to language processing, audio analysis, and genomic studies.

    Attributes:
        vocab_size (int): The size of the vocabulary.
        n_layer (int): The number of MambaBlock layers.
        d_conv (int): The convolution kernel size used within each MambaBlock.
        d_state (int): The dimensionality of the state vector in each MambaBlock's selective scan.
        d_model (int): The dimensionality of the embeddings and the input/output size of each layer.
        max_length (int): The maximum length of the input sequences.
        expand (int): Factor to determine the inner dimension size based on `d_model`.
        start_token (int): The token used to indicate the start of a sequence.
        end_token (int): The token used to indicate the end of a sequence.
        dropout (float): Dropout rate used in the dropout layer.
        bias (bool): Indicates whether to use bias in the Dense layers of MambaBlock. Defaults to True.
        conv_bias (bool): Indicates whether to use bias in the Conv layer of MambaBlock. Defaults to True.
        dt_rank (int or 'auto'): The rank for delta transformations in each MambaBlock's selective scan. If 'auto',
                                 it is calculated based on `d_model`.

    Example:
    ```python
        import jax
        import jax.numpy as jnp
        from nanodl import ArrayDataset, DataLoader
        #from nanodl import Mamba, MambaDataParallelTrainer

        # Generate dummy data
        batch_size = 8
        max_length = 128

        # Replace with actual tokenised data
        data = jnp.ones((101, max_length+1), dtype=jnp.int16)

        # Shift to create next-token prediction dataset
        dummy_inputs = data[:, :-1]
        dummy_targets = data[:, 1:]

        # Create dataset and dataloader
        dataset = ArrayDataset(dummy_inputs, dummy_targets)
        dataloader = DataLoader(dataset,
                                batch_size=batch_size,
                                shuffle=True,
                                drop_last=False)

        # How to loop through dataloader
        for batch in dataloader:
            x, y = batch
            print(x.shape, y.shape)
            break

        # model parameters
        hyperparams = {
            'vocab_size': 100,
            'expand': 2,
            'n_layer': 2,
            'd_conv': 3,
            'dt_rank': 16,
            'd_state': 8,
            'd_model': 64,
            'dropout': 0.2,
            'bias':True,
            'conv_bias': True,
            'max_length': max_length,
            'start_token': 0,
            'end_token': 50,
        }

        # Initialize model
        model = Mamba(**hyperparams)
        rngs = jax.random.PRNGKey(0)
        rngs, dropout_rng = jax.random.split(rngs)
        params = model.init({'params': rngs, 'dropout': dropout_rng},
                            dummy_inputs)['params']

        # Call as you would a Jax/Flax model
        outputs = model.apply({'params': params},
                            dummy_inputs,
                            rngs={'dropout': dropout_rng})

        print(outputs.shape)

        # Training on data
        trainer = MambaDataParallelTrainer(model, dummy_inputs.shape, 'params.pkl')
        trainer.train(train_loader=dataloader,
                    num_epochs=2,
                    val_loader=dataloader)

        print(trainer.evaluate(dataloader))

        # Generating from a start token
        start_tokens = jnp.array([[123, 456]])

        # Remember to load the trained parameters
        params = trainer.load_params('params.pkl')
        outputs = model.apply({'params': params},
                            start_tokens,
                            rngs={'dropout': jax.random.PRNGKey(2)},
                            method=model.generate)
        print(outputs)
        ```
    """

    vocab_size: int
    n_layer: int
    d_conv: int
    d_state: int
    d_model: int
    max_length: int
    expand: int
    max_length: int
    start_token: int
    end_token: int
    dropout: float
    bias: bool = True
    conv_bias: bool = True
    dt_rank: int = "auto"

    def setup(self):
        self.d_inner = int(self.expand * self.d_model)

        if self.dt_rank == "auto":
            self.dt_rank = math.ceil(self.d_model / 16)

        self.embedding = nn.Embed(self.vocab_size, self.d_model)

        self.layers = [
            MambaBlock(
                d_inner=self.d_inner,
                d_conv=self.d_conv,
                dt_rank=self.dt_rank,
                d_state=self.d_state,
                d_model=self.d_model,
                seq_len=self.max_length,
                bias=self.bias,
                conv_bias=self.conv_bias,
            )
            for _ in range(self.n_layer)
        ]

        self.norm_f = nn.RMSNorm(self.d_model)
        self.dropout1 = nn.Dropout(self.dropout)
        self.lm_head = nn.Dense(features=self.vocab_size, use_bias=False)
        # Note: Flax doesn't support parameter sharing like PyTorch's weight tying directly.
        # You might need to implement a custom method for weight tying or handle it outside the model definition.

    def __call__(self, input_ids: jnp.ndarray, training: bool = False) -> jnp.ndarray:

        x = self.embedding(input_ids)
        for layer in self.layers:
            x = self.dropout1(layer(x), deterministic=not training)

        x = self.norm_f(x)
        logits = self.lm_head(x)
        return logits

    def zero_pad(self, arr, max_length):
        current_length = arr.shape[1]
        num_zeros = max_length - current_length

        if num_zeros > 0:
            zeros = jnp.zeros((arr.shape[0], num_zeros), dtype=arr.dtype)
            padded_array = jnp.concatenate([arr, zeros], axis=1)
        else:
            padded_array = arr

        return padded_array

    def generate(
        self,
        x: Optional[jnp.ndarray] = None,
        temperature: float = 1.0,
        deterministic: bool = False,
    ) -> Tuple[jnp.ndarray]:

        if x is not None:
            assert x.shape[0] == 1, "Batch size must be 1, else use generate_batch()"

        decoder_input = x if x is not None else jnp.array([[self.start_token]])
        output_sequence = []

        # Autoregressive decoding loop
        print(self.zero_pad(decoder_input, self.max_length).shape)
        for _ in range(self.max_length - 1):
            decoder_output = self.__call__(
                self.zero_pad(decoder_input, self.max_length), training=False
            )[0]
            print(decoder_output.shape)
            last_token_logits = decoder_output[:, -1, :]
            scaled_logits = last_token_logits / temperature
            next_token_probabilities = jax.nn.softmax(scaled_logits, axis=-1)

            if deterministic:
                next_token = jnp.argmax(next_token_probabilities, axis=-1)
            else:
                next_token = jax.random.categorical(
                    jax.random.PRNGKey(int(time.time())),
                    next_token_probabilities,
                    axis=-1,
                )

            next_token = next_token[0]
            output_sequence.append(next_token.item())
            decoder_input = jnp.concatenate(
                [decoder_input, jnp.array([[next_token]])], axis=1
            )

            if (
                next_token.item() == self.end_token
                or len(output_sequence) == self.max_length
            ):
                break

        return jnp.array(output_sequence)

    def generate_batch(
        self,
        x: Optional[jnp.ndarray] = None,
        temperature: float = 1.0,
        deterministic: bool = False,
    ) -> jnp.ndarray:

        batch_size = x.shape[0] if x is not None else 1
        decoder_input = (
            x if x is not None else jnp.full((batch_size, 1), self.start_token)
        )
        output_sequences = jnp.zeros((batch_size, self.max_length), dtype=jnp.int32)

        for i in range(self.max_length - 1):
            decoder_output = self.__call__(
                self.zero_pad(decoder_input, self.max_length), training=False
            )[0]
            last_token_logits = decoder_output[:, -1, :]
            scaled_logits = last_token_logits / temperature
            next_token_probabilities = jax.nn.softmax(scaled_logits, axis=-1)

            if deterministic:
                next_token = jnp.argmax(next_token_probabilities, axis=-1)
            else:
                key = jax.random.PRNGKey(int(time.time()))
                next_token = jax.random.categorical(
                    key, next_token_probabilities, axis=-1
                )

            output_sequences = output_sequences.at[:, i].set(next_token)
            decoder_input = jnp.concatenate(
                [decoder_input, next_token[:, None]], axis=1
            )

            if (
                jnp.all(next_token == self.end_token)
                or len(output_sequences) == self.max_length
            ):
                break

        return output_sequences


class MambaDataParallelTrainer:
    """
    Trainer class using data parallelism with JAX.
    This trainer leverages JAX's `pmap` for parallel training across multiple devices (GPUs/TPUs).
    It handles the model training loop, including gradient computation, parameter updates, and evaluation.

    Attributes:
        model (Any): The model to be trained.
        input_shape (Tuple[int, ...]): The shape of the input tensor.
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
            MambaDataParallelTrainer.train_step, axis_name="devices"
        )
        self.evaluation_step = jax.pmap(
            MambaDataParallelTrainer.evaluation_step, axis_name="devices"
        )
        self.state = self.create_train_state(learning_rate, input_shape)
        print(f"Number of accelerators: {self.num_devices}")

    def create_train_state(
        self, learning_rate: float, input_shape: Tuple[int, ...]
    ) -> Any:

        rngs = {"params": jax.random.key(0), "dropout": jax.random.key(1)}
        params = self.model.init(rngs, jnp.ones(input_shape, dtype=jnp.int32))["params"]

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
            )
            return optax.softmax_cross_entropy_with_integer_labels(
                logits, targets
            ).mean()

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
                inputs = inputs.reshape((self.num_devices, batch_size_per_device, -1))
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
        )
        return optax.softmax_cross_entropy_with_integer_labels(logits, targets).mean()

    def evaluate(self, test_loader: Iterable[Tuple[jnp.ndarray, jnp.ndarray]]) -> None:

        total_loss = 0.0
        count = 0
        for inputs, targets in test_loader:
            batch_size = inputs.shape[0]
            batch_size_per_device = batch_size // self.num_devices
            inputs = inputs.reshape((self.num_devices, batch_size_per_device, -1))
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


import jax
import jax.numpy as jnp

from nanodl import ArrayDataset, DataLoader

# from nanodl import Mamba, MambaDataParallelTrainer

# Generate dummy data
batch_size = 8
max_length = 128

# Replace with actual tokenised data
data = jnp.ones((101, max_length + 1), dtype=jnp.int16)

# Shift to create next-token prediction dataset
dummy_inputs = data[:, :-1]
dummy_targets = data[:, 1:]

# Create dataset and dataloader
dataset = ArrayDataset(dummy_inputs, dummy_targets)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)

# How to loop through dataloader
for batch in dataloader:
    x, y = batch
    print(x.shape, y.shape)
    break

# model parameters
hyperparams = {
    "vocab_size": 100,
    "expand": 2,
    "n_layer": 2,
    "d_conv": 3,
    "dt_rank": 16,
    "d_state": 8,
    "d_model": 64,
    "dropout": 0.2,
    "bias": True,
    "conv_bias": True,
    "max_length": max_length,
    "start_token": 0,
    "end_token": 50,
}

# Initialize model
model = Mamba(**hyperparams)
rngs = jax.random.PRNGKey(0)
rngs, dropout_rng = jax.random.split(rngs)
params = model.init({"params": rngs, "dropout": dropout_rng}, dummy_inputs)["params"]

# Call as you would a Jax/Flax model
outputs = model.apply({"params": params}, dummy_inputs, rngs={"dropout": dropout_rng})

print(outputs.shape)

start_tokens = jnp.array([[123, 456]])
outputs = model.apply(
    {"params": params},
    start_tokens,
    rngs={"dropout": jax.random.PRNGKey(2)},
    method=model.generate,
)
print(outputs)
