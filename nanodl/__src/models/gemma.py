import time
from typing import Any, Iterable, Optional, Tuple

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import optax
from flax.training import train_state


class RotaryPositionalEncoding:
    """
    Implements rotary positional encoding (RoPE) for transformers, enhancing their ability to capture sequence order.

    Rotary positional encoding applies a rotation to the embedding of each token based on its position in the sequence. This method helps preserve the relative positional information between tokens in a more effective manner compared to traditional positional encodings.

    Attributes:
        dim_model (int): The dimensionality of the model embeddings.

    """

    def __init__(self, dim_model: int):
        super().__init__()
        self.dim_model = dim_model
        inv_freq = 1.0 / (
            10000 ** (jnp.arange(0, dim_model, 2, dtype=jnp.float32) / dim_model)
        )
        self.inv_freq = inv_freq
        self._seq_len_cached = None
        self._cos_cached = None
        self._sin_cached = None

    def _update_cos_sin_tables(self, x, seq_dimension=1):
        seq_len = x.shape[seq_dimension]

        if seq_len != self._seq_len_cached:
            self._seq_len_cached = seq_len
            t = jnp.arange(seq_len, dtype=self.inv_freq.dtype)
            freqs = jnp.outer(t, self.inv_freq)
            emb = jnp.concatenate((freqs, freqs), axis=-1)
            self._cos_cached = jnp.cos(emb)[None, None, :, :]
            self._sin_cached = jnp.sin(emb)[None, None, :, :]

        return self._cos_cached, self._sin_cached

    def rotate_half(self, x):
        x1, x2 = jnp.split(x, 2, axis=-1)
        return jnp.concatenate((-x2, x1), axis=-1)

    def apply_rotary_pos_emb(self, x, cos, sin):
        cos = cos[:, :, : x.shape[-2], :]
        sin = sin[:, :, : x.shape[-2], :]
        return (x * cos) + (self.rotate_half(x) * sin)

    def __call__(self, q, k):
        self._cos_cached, self._sin_cached = self._update_cos_sin_tables(
            k, seq_dimension=-2
        )
        return (
            self.apply_rotary_pos_emb(q, self._cos_cached, self._sin_cached)[0],
            self.apply_rotary_pos_emb(k, self._cos_cached, self._sin_cached)[0],
        )


class GroupedRotaryMultiHeadAttention(nn.Module):
    """
    Implements multi-head self-attention with grouped rotary positional embeddings.

    This module extends the concept of multi-head attention by applying rotary positional embeddings to groups of attention heads. This approach allows for a more nuanced representation of positional information and potentially improves the model's understanding of sequence order and context.

    Attributes:
        hidden_dim (int): Dimensionality of the input and output features.
        num_heads (int): Number of attention heads.
        num_groups (int): Number of groups to split the heads into for applying rotary positional embeddings separately.

    """

    hidden_dim: int  # Output dimension
    num_heads: int  # Number of parallel heads
    num_groups: int  # Number of groups to split the heads into

    def setup(self):
        self.query_projection = nn.Dense(
            self.hidden_dim // self.num_heads,
            kernel_init=nn.initializers.xavier_uniform(),
            bias_init=nn.initializers.zeros,
        )
        self.key_projection = nn.Dense(
            self.hidden_dim // (self.num_heads * self.num_groups),
            kernel_init=nn.initializers.xavier_uniform(),
            bias_init=nn.initializers.zeros,
        )
        self.value_projection = nn.Dense(
            self.hidden_dim // (self.num_heads * self.num_groups),
            kernel_init=nn.initializers.xavier_uniform(),
            bias_init=nn.initializers.zeros,
        )
        self.rope = RotaryPositionalEncoding(self.hidden_dim // self.num_groups)
        self.output = nn.Dense(
            self.hidden_dim,
            kernel_init=nn.initializers.xavier_uniform(),
            bias_init=nn.initializers.zeros,
        )

    def __call__(
        self, inputs: jnp.ndarray, context: jnp.ndarray, mask: jnp.ndarray = None
    ) -> tuple:

        query = self.query_projection(inputs)
        key = self.key_projection(context)
        value = self.value_projection(context)

        # Break query into groups and transpose to (num_groups, batch_size, seq_len, dims)
        # This will allow vmapping over the groups for parallelization
        grouped_query = jnp.reshape(
            query, (query.shape[0], query.shape[1], self.num_groups, -1)
        )
        grouped_query = jnp.repeat(grouped_query, self.num_heads, axis=-1)
        grouped_query = jnp.transpose(grouped_query, (2, 0, 1, 3))

        # Repeat the key and values
        key = jnp.repeat(key, self.num_heads, axis=-1)
        value = jnp.repeat(value, self.num_heads, axis=-1)
        vectorized_process_group = jax.vmap(
            self.process_group, in_axes=(0, None, None, None)
        )
        results = vectorized_process_group(grouped_query, key, value, mask)

        # Merge the groups back together
        context_vectors = jnp.concatenate(results[0], axis=-1)
        return self.output(context_vectors), results[1]

    def process_group(self, query, key, value, mask):
        query, key = self.rope(query, key)
        return self.attention_function(query, key, value, mask=mask)

    def attention_function(self, query, key, value, mask=None):
        input_length = query.shape[1]
        context_length = key.shape[1]
        head_dim = query.shape[-1] // self.num_heads
        dim_key = key.shape[-1]

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


class GemmaMLP(nn.Module):
    """
    GemmaMLP is  processes inputs through a sequence of dense layers
    and applies a gating operation to enhance the representational capacity of the model.

    Attributes:
        hidden_size (int): The size of the output dimension of the MLP, also the dimensionality
                           of the input features.
        intermediate_size (int): The size of the intermediate layer, where the input is projected
                                 to before the gating mechanism and the subsequent projection back
                                 to the original dimensionality.
    """

    hidden_size: int
    intermediate_size: int

    def setup(self):
        self.gate_proj = nn.Dense(self.intermediate_size)
        self.up_proj = nn.Dense(self.intermediate_size)
        self.down_proj = nn.Dense(self.hidden_size)

    def __call__(self, x):
        gate = jax.nn.gelu(self.gate_proj(x))
        up = self.up_proj(x)
        fuse = gate * up
        outputs = self.down_proj(fuse)
        return outputs


class GemmaDecoderBlock(nn.Module):
    """
    Implements a decoder block for the Gemma model, incorporating grouped rotary positional embeddings.

    This block is designed to enhance the model's ability to understand and generate text by using grouped rotary positional embeddings for more nuanced positional encoding, alongside traditional transformer mechanisms like self-attention and feed-forward layers.

    Attributes:
        hidden_dim (int): Dimensionality of the input and output features.
        num_heads (int): Number of attention heads in the multi-head self-attention mechanism.
        feedforward_dim (int): Dimensionality of the inner layer of the feed-forward network.
        dropout (float): Dropout rate for regularization.
        num_groups (int): Number of groups for the grouped rotary positional embeddings.

    """

    hidden_dim: int
    num_heads: int
    feedforward_dim: int
    dropout: float
    num_groups: int

    def setup(self):
        self.attention = GroupedRotaryMultiHeadAttention(
            hidden_dim=self.hidden_dim,
            num_heads=self.num_heads,
            num_groups=self.num_groups,
        )
        self.feed_forward = GemmaMLP(self.feedforward_dim, self.hidden_dim)
        self.norm1 = nn.RMSNorm()
        self.norm2 = nn.RMSNorm()
        self.dropout1 = nn.Dropout(self.dropout)
        self.dropout2 = nn.Dropout(self.dropout)

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

    def __call__(self, x: jnp.ndarray, training: bool = False) -> tuple:

        mask = self.causal_mask(x.shape[0], x.shape[1], x.shape[1])

        x = self.norm1(x)
        attended_x, attention = self.attention(x, x, mask=mask)
        x = self.dropout1(x, deterministic=not training)
        x += attended_x

        x = self.norm2(x)
        output = self.feed_forward(x)
        x = self.dropout2(x, deterministic=not training)
        x += output

        return x, jnp.array(attention)


class GemmaDecoder(nn.Module):
    """
    Implements the decoder component of the LLaMA2 model.

    The decoder is composed of multiple LLaMA2DecoderBlocks, processing sequences of tokens to generate text. It includes an embedding layer to convert tokens into vectors and an output layer to predict the next token in the sequence.

    Attributes:
        num_layers (int): Number of LLaMA2DecoderBlocks in the decoder.
        hidden_dim (int): Dimensionality of the input and output features for the blocks.
        num_heads (int): Number of attention heads in each block.
        num_groups (int): Number of groups for the grouped rotary positional embeddings in each block.
        feedforward_dim (int): Dimensionality of the inner layer of the feed-forward networks in the blocks.
        dropout (float): Dropout rate used for regularization.
        vocab_size (float): Size of the vocabulary.
        embed_dim (float): Dimensionality of the token embeddings.

    """

    num_layers: int
    hidden_dim: int
    num_heads: int
    num_groups: int
    feedforward_dim: int
    dropout: float
    vocab_size: float
    embed_dim: float

    def setup(self):
        self.embedding = nn.Embed(
            num_embeddings=self.vocab_size, features=self.embed_dim
        )

        self.layers = [
            GemmaDecoderBlock(
                self.hidden_dim,
                self.num_heads,
                self.feedforward_dim,
                self.dropout,
                self.num_groups,
            )
            for _ in range(self.num_layers)
        ]

        self.outputs = nn.Dense(self.vocab_size)

    def __call__(
        self, x: jnp.ndarray, training: bool = False, drop_last_layer: bool = False
    ) -> tuple:

        attention_maps = []
        x = self.embedding(x)
        for layer in self.layers:
            x, attention = layer(x, training=training)
            attention_maps.append(attention)

        if not drop_last_layer:
            x = self.outputs(x)

        return x, jnp.array(attention_maps)


class Gemma(nn.Module):
    """
    Implements the Gemma model for text generation, featuring GQA + RMSNorm + RoPE.

    Attributes:
        num_layers (int): Number of layers (blocks) in the LLaMA2 model.
        num_heads (int): Number of attention heads in each block.
        num_groups (int): Number of groups for the grouped rotary positional embeddings in each block.
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

    LlaMA is built upon the transformer architecture, incorporating enhancements inspired by recent advancements in the field of large language models.
    These improvements are drawn from various sources, such as GPT-3, PaLM, and GPT-Neo. Notable modifications include the adoption of pre-normalization for enhanced training stability,
    employing the RMSNorm normalization function. Additionally, the ReLU non-linearity is replaced with the SwiGLU activation function, which is a variant of the GLU activation function.
    Absolute positional embeddings are replaced with rotary positional embeddings (RoPE), implemented at each layer of the network. For specific hyper-parameter details, refer to Table 2 in the document.

    Example usage:
        ```
        import jax
        import jax.numpy as jnp
        from nanodl import ArrayDataset, DataLoader
        from nanodl import Gemma, GemmaDataParallelTrainer

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
            'num_groups': 2,
        }

        # Initialize model
        model = Gemma(**hyperparams)
        rngs = jax.random.PRNGKey(0)
        rngs, dropout_rng = jax.random.split(rngs)
        params = model.init({'params': rngs, 'dropout': dropout_rng}, dummy_inputs)['params']

        # Call as you would a Jax/Flax model
        outputs = model.apply({'params': params},
                            dummy_inputs,
                            rngs={'dropout': dropout_rng})
        print(outputs.shape)

        # Training on data
        trainer = GemmaDataParallelTrainer(model, dummy_inputs.shape, 'params.pkl')
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
    num_groups: int
    hidden_dim: int
    feedforward_dim: int
    dropout: float
    vocab_size: float
    embed_dim: float
    max_length: int
    start_token: int
    end_token: int

    def setup(self):

        self.decoder = GemmaDecoder(
            self.num_layers,
            self.hidden_dim,
            self.num_heads,
            self.num_groups,
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


class GemmaDataParallelTrainer:
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
            GemmaDataParallelTrainer.train_step, axis_name="devices"
        )
        self.evaluation_step = jax.pmap(
            GemmaDataParallelTrainer.evaluation_step, axis_name="devices"
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
