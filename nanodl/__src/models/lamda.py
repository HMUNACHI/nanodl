import time
from typing import Any, Iterable, Optional, Tuple

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import optax
from flax.training import train_state


class RelativeMultiHeadAttention(nn.Module):
    """
    Implements relative multi-head attention mechanism for transformers.

    This module enhances the transformer architecture by incorporating relative position information directly into the attention mechanism, allowing the model to better capture sequence order and dependencies based on the relative positions of tokens.

    Attributes:
        hidden_dim (int): Dimensionality of the input and output features.
        num_heads (int): Number of attention heads.

    """

    hidden_dim: int
    num_heads: int

    def setup(self):
        # Because the Query is determined from a context, project separately
        self.query_projection = nn.Dense(
            self.hidden_dim,
            kernel_init=nn.initializers.xavier_uniform(),
            bias_init=nn.initializers.zeros,
        )
        self.key_projection = nn.Dense(
            self.hidden_dim,
            kernel_init=nn.initializers.xavier_uniform(),
            bias_init=nn.initializers.zeros,
        )
        self.value_projection = nn.Dense(
            self.hidden_dim,
            kernel_init=nn.initializers.xavier_uniform(),
            bias_init=nn.initializers.zeros,
        )
        self.output = nn.Dense(
            self.hidden_dim,
            kernel_init=nn.initializers.xavier_uniform(),
            bias_init=nn.initializers.zeros,
        )

    def __call__(
        self,
        inputs: jnp.ndarray,
        context: jnp.ndarray,
        mask: jnp.ndarray = None,
        clip: int = 3,
    ) -> tuple:

        query = self.query_projection(inputs)
        key = self.key_projection(context)
        value = self.value_projection(context)

        query_relative_positions = jnp.expand_dims(jnp.arange(query.shape[2]), axis=0)
        query_relative_positions -= jnp.expand_dims(jnp.arange(query.shape[1]), axis=1)
        query_relative_positions = jnp.where(
            query_relative_positions < clip, query_relative_positions, clip
        )
        query_relative_positions = jnp.where(
            query_relative_positions > -clip, query_relative_positions, -clip
        )
        query += query_relative_positions

        value_relative_positions = jnp.expand_dims(jnp.arange(value.shape[2]), axis=0)
        value_relative_positions -= jnp.expand_dims(jnp.arange(value.shape[1]), axis=1)
        value_relative_positions = jnp.where(
            value_relative_positions < clip, value_relative_positions, clip
        )
        value_relative_positions = jnp.where(
            value_relative_positions > -clip, value_relative_positions, -clip
        )
        value += value_relative_positions
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

    This module applies two linear transformations with a GEGLU activation in between, as per the original transformer model design. It is applied to each position separately and identically.

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
        self.activation = GEGLU(self.num_hiddens)
        self.dense2 = nn.Dense(
            self.num_outputs, kernel_init=nn.initializers.xavier_uniform()
        )

    def __call__(self, X: jnp.ndarray) -> jnp.ndarray:
        return self.dense2(self.activation(self.dense1(X)))


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


class GEGLU(nn.Module):
    """
    Gated GLU (Gated Linear Unit).
    GEGLU(x) = x * 0.5 * gate * (1 + tanh(gate * 0.7978845608 * (1 + 0.044715 * (gate**2))))

    Args:
        output_dim (int): Output dimension of the GLU layer.
    """

    output_dim: int

    def setup(self):
        self.dense = nn.Dense(
            self.output_dim * 2, kernel_init=nn.initializers.xavier_uniform()
        )

    def __call__(self, inputs):
        x = self.dense(inputs)
        x, gate = x[..., : self.output_dim], x[..., self.output_dim :]
        tanh_res = jnp.tanh(gate * 0.7978845608 * (1 + 0.044715 * (gate**2)))
        return x * 0.5 * gate * (1 + tanh_res)


class LaMDABlock(nn.Module):
    """
    Implements a transformer block for LaMDA with self-attention and feed-forward layers.

    This block is designed for the LaMDA model, focusing on generating conversational responses. It uses relative multi-head attention to incorporate positional information in a more nuanced way compared to traditional self-attention mechanisms.

    Attributes:
        hidden_dim (int): Dimensionality of the input and output features.
        num_heads (int): Number of attention heads in the multi-head self-attention mechanism.
        feedforward_dim (int): Dimensionality of the inner layer of the feed-forward network.
        dropout (float): Dropout rate for regularization.

    M"""

    hidden_dim: int
    num_heads: int
    feedforward_dim: int
    dropout: float

    def setup(self):
        self.attention1 = RelativeMultiHeadAttention(
            hidden_dim=self.hidden_dim, num_heads=self.num_heads
        )
        self.attention2 = RelativeMultiHeadAttention(
            hidden_dim=self.hidden_dim, num_heads=self.num_heads
        )
        self.feed_forward = PositionWiseFFN(self.feedforward_dim, self.hidden_dim)
        self.add_norm1 = AddNorm(self.dropout)
        self.add_norm2 = AddNorm(self.dropout)
        self.add_norm3 = AddNorm(self.dropout)

    def causal_mask(
        self, batch_size: int, destination_dim: int, source_dim: int
    ) -> jnp.ndarray:

        # Create index tensors for the source and destination dimensions
        idx_source = jnp.arange(destination_dim)[:, None]
        idx_destination = jnp.arange(source_dim)
        mask = idx_source >= idx_destination - source_dim + destination_dim
        mask = mask.astype(jnp.int32)

        # Expand dimensions to match the required output shape
        mask = mask[None, None, :, :]
        return jnp.broadcast_to(
            mask, (batch_size, self.num_heads, destination_dim, source_dim)
        )

    def __call__(
        self, x: jnp.ndarray, mask: jnp.ndarray = None, training: bool = False
    ) -> tuple:

        mask = self.causal_mask(x.shape[0], x.shape[1], x.shape[1])

        attended_x, attention1 = self.attention1(x, x, mask=mask)
        x = self.add_norm1(x, attended_x, training)

        attended_x, attention2 = self.attention2(x, x, mask=mask)
        x = self.add_norm2(x, attended_x, training)

        linear_output = self.feed_forward(x)
        x = self.add_norm3(x, linear_output, training)

        return x, jnp.array(attention1), jnp.array(attention2)


class LaMDADecoder(nn.Module):
    """
    Implements the decoder component of the LaMDA model.

    The decoder is composed of multiple LaMDA blocks, processing sequences of tokens to generate conversational text. It includes an embedding layer to convert tokens into vectors and an output layer to predict the next token in the sequence.

    Attributes:
        num_layers (int): Number of LaMDA blocks in the decoder.
        hidden_dim (int): Dimensionality of the input and output features for the blocks.
        num_heads (int): Number of attention heads in each LaMDA block.
        feedforward_dim (int): Dimensionality of the inner layer of the feed-forward networks in the blocks.
        dropout (float): Dropout rate used for regularization.
        vocab_size (float): Size of the vocabulary.
        embed_dim (float): Dimensionality of the token embeddings.

    """

    num_layers: int
    hidden_dim: int
    num_heads: int
    feedforward_dim: int
    dropout: float
    vocab_size: float
    embed_dim: float

    def setup(self):
        self.embedding = nn.Embed(
            num_embeddings=self.vocab_size, features=self.embed_dim
        )

        self.layers = [
            LaMDABlock(
                self.hidden_dim, self.num_heads, self.feedforward_dim, self.dropout
            )
            for _ in range(self.num_layers)
        ]

        self.outputs = nn.Dense(self.vocab_size)

    def __call__(
        self,
        x: jnp.ndarray,
        mask: jnp.ndarray = None,
        training: bool = False,
        drop_last_layer: bool = False,
    ) -> tuple:

        attention_maps = []
        x = self.embedding(x)
        cross_attention_maps = []
        for layer in self.layers:
            x, attention, cross_attention = layer(x, mask=mask, training=training)
            attention_maps.append(attention)
            cross_attention_maps.append(cross_attention)

        if not drop_last_layer:
            x = self.outputs(x)

        return x, jnp.array(attention_maps), jnp.array(cross_attention_maps)


class LaMDA(nn.Module):
    """
    Implements the LaMDA model for generating conversational responses.

    LaMDA is designed for conversational applications, leveraging the transformer architecture to generate high-quality text responses. It uses relative multi-head attention within its transformer blocks to account for the positional context of words more effectively.

    Attributes:
        num_layers (int): Number of layers (blocks) in the LaMDA model.
        num_heads (int): Number of attention heads in each block.
        hidden_dim (int): Dimensionality of the input and output features for the blocks.
        feedforward_dim (int): Dimensionality of the inner layer of the feed-forward networks in the blocks.
        dropout (float): Dropout rate used for regularization.
        vocab_size (float): Size of the vocabulary.
        embed_dim (float): Dimensionality of the token embeddings.
        max_length (int): Maximum length of the generated sequences.
        start_token (int): Token used to start the generation process.
        end_token (int): Token that indicates the end of a generated sequence.

    Methods:
        generate(x, temperature, deterministic): Generates a sequence of tokens autoregressively.
        generate_batch(x, temperature, deterministic): Generates sequences of tokens for a batch of initial sequences autoregressively.

    LaMBDA, which stands for "Language Model for Dialogue Applications," is a deep learning model developed by Google.
    Its primary motivation lies in addressing the limitations of existing conversational AI models, such as GPT-3,
    by explicitly targeting dialogue applications. LaMBDA's architecture is designed to excel in multi-turn conversations,
    offering improvements in several key aspects. It incorporates features like context windowing, which enables it to remember and track information over longer dialogues,
    and provides better control over generating detailed responses. LaMBDA also introduces a more controllable prompt engineering mechanism,
    allowing users to instruct the model more precisely for various dialogue tasks. Overall, LaMBDA represents a significant step forward in the development of conversational AI models,
    offering enhanced performance and usability in real-world dialogue applications.

    Note:
    This is the architecture for LaMDA itself for now, the system is a lot more complex. At inference, LaMDA makes use of a single model to perform multiple tasks.
    it generates potential responses, which are then filtered for safety, grounded on an external knowledge source, and re-ranked to find the highest-quality response.

    Example Usage:
        ```py
        import jax
        import jax.numpy as jnp
        from nanodl import ArrayDataset, DataLoader
        from nanodl import LaMDA, LaMDADataParallelTrainer

        # Generate dummy data
        batch_size = 8
        max_length = 10

        # Replace with actual tokenised data
        data = jnp.ones((101, max_length), dtype=jnp.int32)

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
            'num_layers': 1,
            'hidden_dim': 256,
            'num_heads': 2,
            'feedforward_dim': 256,
            'dropout': 0.1,
            'vocab_size': 1000,
            'embed_dim': 256,
            'max_length': max_length,
            'start_token': 0,
            'end_token': 50,
        }

        # Initialize model
        model = LaMDA(**hyperparams)
        rngs = jax.random.PRNGKey(0)
        rngs, dropout_rng = jax.random.split(rngs)
        params = model.init({'params': rngs, 'dropout': dropout_rng}, dummy_inputs)['params']

        # Call as you would a Jax/Flax model
        outputs = model.apply({'params': params},
                            dummy_inputs,
                            rngs={'dropout': dropout_rng})
        print(outputs.shape)

        # Training on data
        trainer = LaMDADataParallelTrainer(model, dummy_inputs.shape, 'params.pkl')
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

    num_layers: int
    num_heads: int
    hidden_dim: int
    feedforward_dim: int
    dropout: float
    vocab_size: float
    embed_dim: float
    max_length: int
    start_token: int
    end_token: int

    def setup(self):
        self.decoder = LaMDADecoder(
            self.num_layers,
            self.hidden_dim,
            self.num_heads,
            self.feedforward_dim,
            self.dropout,
            self.vocab_size,
            self.embed_dim,
        )

    def __call__(
        self, x: jnp.ndarray, training: bool = False, drop_last_layer: bool = False
    ) -> jnp.ndarray:

        return self.decoder(x=x, training=training, drop_last_layer=drop_last_layer)[0]

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
        for _ in range(self.max_length):
            decoder_output = self.decoder(decoder_input, training=False)[0]
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

            if next_token.item() == self.end_token:
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

        for i in range(self.max_length):
            decoder_output = self.decoder(decoder_input, training=False)[0]
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

            if jnp.all(next_token == self.end_token):
                break

        return output_sequences


class LaMDADataParallelTrainer:
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
            LaMDADataParallelTrainer.train_step, axis_name="devices"
        )
        self.evaluation_step = jax.pmap(
            LaMDADataParallelTrainer.evaluation_step, axis_name="devices"
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
