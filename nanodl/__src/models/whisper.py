import time
from typing import Any, Iterable, Optional, Tuple

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import optax
from flax.training import train_state


class SpeechEmbedding(nn.Module):
    """
    Implements a speech embedding layer for processing audio signals.

    This layer applies two convolutional operations followed by GELU activations to the input audio signals. The first convolution maintains the sequence length, while the second halves it. Additionally, it adds sinusoidal embeddings to capture positional information within the audio sequence.

    """

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = nn.gelu(nn.Conv(features=x.shape[-1], kernel_size=(3,), padding="SAME")(x))
        x = nn.gelu(
            nn.Conv(
                features=x.shape[-1], kernel_size=(3,), strides=(2,), padding="SAME"
            )(x)
        )
        return jnp.concatenate((x, self.sinusoidal_embedding(x)), axis=-2)

    def sinusoidal_embedding(
        self, x: jnp.ndarray, max_position: int = 10000
    ) -> jnp.ndarray:
        batch_size, seq_len, hidden_dim = x.shape
        positions = jnp.arange(seq_len)[:, None]
        angles = (jnp.arange(hidden_dim) / hidden_dim)[None, :]
        encodings = jnp.sin(positions / jnp.power(max_position, angles))[None, :, :]
        encodings = jnp.repeat(encodings, batch_size, axis=0)
        return x + encodings


class PositionalEncoding(nn.Module):
    """
    Implements the positional encoding layer for adding positional information to embeddings in a transformer model.

    This layer generates a unique positional encoding for each position in the input sequence using a combination of sine and cosine functions. The encoding is added to the embedding vector to provide the model with information about the relative or absolute position of the tokens in the sequence.

    Attributes:
        num_embeddings (int): The maximum number of positions for which to generate positional encodings.
        features (int): The dimensionality of the embeddings/positional encodings.

    """

    num_embeddings: int
    features: int

    def setup(self):
        positional_encoding = jnp.zeros((self.features, self.num_embeddings))
        position = jnp.arange(0, self.features, dtype=jnp.float32)[:, None]
        div_term = jnp.exp(
            jnp.arange(0, self.num_embeddings, 2)
            * (-jnp.log(10000.0) / self.num_embeddings)
        )
        positional_encoding = positional_encoding.at[:, 0::2].set(
            jnp.sin(position * div_term)
        )
        positional_encoding = positional_encoding.at[:, 1::2].set(
            jnp.cos(position * div_term)
        )
        self.positional_encoding = positional_encoding.T

    def __call__(self, x):
        x = x + self.positional_encoding[: x.shape[1]]
        return x


class TokenAndPositionEmbedding(nn.Module):
    """
    Token and Position Embedding.

    Args:
        max_len (int): Maximum sequence length.
        vocab_size (int): Vocabulary size.
        embed_dim (int): Embedding dimension.
    """

    max_len: int
    vocab_size: int
    embed_dim: int
    learned_position: bool

    def setup(self):
        self.token_embeddings = nn.Embed(
            num_embeddings=self.vocab_size, features=self.embed_dim
        )

        if self.learned_position:
            self.position_embeddings = nn.Embed(
                num_embeddings=self.max_len, features=self.embed_dim
            )
        else:
            self.position_embeddings = PositionalEncoding(
                num_embeddings=self.max_len, features=self.embed_dim
            )

    def __call__(self, x):
        x = self.token_embeddings(x)
        if self.learned_position:
            return x + self.position_embeddings(jnp.arange(x.shape[1]))
        else:
            return x + self.position_embeddings(x)


class MultiHeadAttention(nn.Module):
    """
    Implements multi-head attention mechanism as described in "Attention is All You Need" by Vaswani et al 2017.

    This module splits the input into multiple heads, applies scaled dot-product attention independently on each head, and then concatenates the results. It allows the model to jointly attend to information from different representation subspaces at different positions.

    Attributes:
        hidden_dim (int): Dimensionality of the input and output features.
        num_heads (int): Number of attention heads.

    """

    hidden_dim: int  # Output dimension
    num_heads: int  # Number of parallel heads

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
        self, inputs: jnp.ndarray, context: jnp.ndarray, mask: jnp.ndarray = None
    ) -> tuple:

        query = self.query_projection(inputs)
        key = self.key_projection(context)
        value = self.value_projection(context)
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
        self.activation = nn.gelu
        self.dense2 = nn.Dense(
            self.num_outputs, kernel_init=nn.initializers.xavier_uniform()
        )

    def __call__(self, X: jnp.ndarray) -> jnp.ndarray:
        return self.dense2(self.activation(self.dense1(X)))


class AddNorm(nn.Module):
    """
    Residual connection followed by layer normalization.

    Args:
        dropout (float): Dropout rate for the residual connection.
    """

    dropout: int

    @nn.compact
    def __call__(self, X: jnp.ndarray, Y: jnp.ndarray, training=False) -> jnp.ndarray:
        return nn.LayerNorm()(
            nn.Dropout(self.dropout)(Y, deterministic=not training) + X
        )


class WhisperSpeechEncoderBlock(nn.Module):
    """
    Implements a single encoder block for the Whisper Speech model, combining self-attention with a feed-forward network.

    The WhisperSpeechEncoderBlock processes the input through a multi-head self-attention mechanism, allowing each position to attend to all positions. This is followed by a position-wise feed-forward network. Layer normalization and dropout are applied for regularization.

    Attributes:
        hidden_dim (int): Dimensionality of the input and output features.
        num_heads (int): Number of attention heads.
        feedforward_dim (int): Dimensionality of the inner layer of the feed-forward network.
        dropout (float): Dropout rate for regularization.

    """

    hidden_dim: int
    num_heads: int
    feedforward_dim: int
    dropout: float

    def setup(self):
        self.attention = MultiHeadAttention(
            hidden_dim=self.hidden_dim, num_heads=self.num_heads
        )
        self.linear = PositionWiseFFN(self.feedforward_dim, self.hidden_dim)
        self.add_norm1 = AddNorm(self.dropout)
        self.add_norm2 = AddNorm(self.dropout)

    def __call__(
        self, x: jnp.ndarray, mask: jnp.ndarray = None, training: bool = False
    ) -> tuple:

        attended_x, attention = self.attention(x, x, mask=mask)
        x = self.add_norm1(x, attended_x, training)
        linear_output = self.linear(x)
        x = self.add_norm2(x, linear_output, training)
        return x, attention


class WhisperSpeechEncoder(nn.Module):
    """
    Implements the encoder component of the Whisper Speech model.

    The WhisperSpeechEncoder processes input audio sequences through an embedding layer followed by multiple WhisperSpeechEncoderBlocks. It aims to capture complex patterns within the audio data by applying self-attention and feed-forward networks to the sequence of embeddings.

    Attributes:
        num_layers (int): Number of WhisperSpeechEncoderBlocks in the encoder.
        hidden_dim (int): Dimensionality of the hidden features.
        num_heads (int): Number of attention heads in the self-attention mechanism.
        feedforward_dim (int): Dimensionality of the feedforward network within each encoder block.
        dropout (float): Dropout rate used for regularization.

    """

    num_layers: int
    hidden_dim: int
    num_heads: int
    feedforward_dim: int
    dropout: float

    def setup(self):
        self.embedding = SpeechEmbedding()

        self.layers = [
            WhisperSpeechEncoderBlock(
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


class WhisperTextDecoderBlock(nn.Module):
    """
    Implements a single decoder block for the Transformer model, combining self-attention, encoder-decoder attention, and a feed-forward network.

    This block first processes the input through self-attention, allowing each position to attend to all positions up to and including itself. Then, it applies encoder-decoder attention, integrating information from the encoder's output. Finally, a position-wise feed-forward network is applied.

    Attributes:
        hidden_dim (int): Dimensionality of the input and output features.
        num_heads (int): Number of attention heads.
        feedforward_dim (int): Dimensionality of the inner layer of the feed-forward network.
        dropout (float): Dropout rate for regularization.

    """

    hidden_dim: int
    num_heads: int
    feedforward_dim: int
    dropout: float

    def setup(self):
        self.attention1 = MultiHeadAttention(
            hidden_dim=self.hidden_dim, num_heads=self.num_heads
        )
        self.attention2 = MultiHeadAttention(
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
        self, x: jnp.ndarray, context: jnp.ndarray, training: bool = False
    ) -> tuple:

        mask = self.causal_mask(x.shape[0], x.shape[1], context.shape[1])

        attended_x, attention1 = self.attention1(x, x)
        x = self.add_norm1(x, attended_x, training)

        attended_x, attention2 = self.attention2(x, context, mask=mask)
        x = self.add_norm2(x, attended_x, training)

        linear_output = self.feed_forward(x)
        x = self.add_norm3(x, linear_output, training)

        return x, jnp.array(attention1), jnp.array(attention2)


class WhisperTextDecoder(nn.Module):
    """
    Implements the decoder component of the Transformer model.

    The Transformer decoder generates output sequences by processing input through multiple layers of TransformerDecoderBlocks. It incorporates context from the encoder at each layer to generate predictions.

    Attributes:
        num_layers (int): Number of TransformerDecoderBlocks in the decoder.
        hidden_dim (int): Dimensionality of the input and output features for the blocks.
        num_heads (int): Number of attention heads in each block.
        feedforward_dim (int): Dimensionality of the inner layer of the feed-forward networks in the blocks.
        dropout (float): Dropout rate used for regularization.
        max_len (int): Maximum sequence length.
        vocab_size (float): Size of the vocabulary.
        embed_dim (float): Dimensionality of the token embeddings.
        learned_position (bool): Indicates if positional embeddings are learned or static.

    """

    num_layers: int
    hidden_dim: int
    num_heads: int
    feedforward_dim: int
    dropout: float
    max_len: int
    vocab_size: int
    embed_dim: int
    learned_position: bool = True

    def setup(self):
        self.embedding = TokenAndPositionEmbedding(
            self.max_len, self.vocab_size, self.embed_dim, self.learned_position
        )

        self.layers = [
            WhisperTextDecoderBlock(
                self.hidden_dim, self.num_heads, self.feedforward_dim, self.dropout
            )
            for _ in range(self.num_layers)
        ]

        self.outputs = nn.Dense(self.vocab_size)

    def __call__(
        self, x: jnp.ndarray, context: jnp.ndarray, training: bool = False
    ) -> tuple:

        attention_maps = []
        x = self.embedding(x)
        cross_attention_maps = []
        for layer in self.layers:
            x, attention, cross_attention = layer(x, context, training=training)
            attention_maps.append(attention)
            cross_attention_maps.append(cross_attention)
        return (
            self.outputs(x),
            jnp.array(attention_maps),
            jnp.array(cross_attention_maps),
        )


class Whisper(nn.Module):
    """
    Implements the Whisper model for speech-to-text tasks, such as speech recognition and transcription.

    The Whisper model utilizes a specialized encoder for processing audio input and a decoder for generating textual output. The encoder captures complex patterns in the audio data using self-attention mechanisms, while the decoder generates corresponding text based on the encoded audio context.

    Attributes:
        num_layers (int): Number of layers in both the encoder and decoder.
        num_heads (int): Number of attention heads in each layer.
        hidden_dim (int): Dimensionality of the input and output features for the layers.
        feedforward_dim (int): Dimensionality of the inner layer of the feed-forward networks in the layers.
        dropout (float): Dropout rate used for regularization.
        vocab_size (float): Size of the vocabulary.
        embed_dim (float): Dimensionality of the token embeddings.
        max_length (int): Maximum length of the generated text sequences.
        start_token (int): Token used to start the generation process.
        end_token (int): Token that indicates the end of a generated sequence.

    Methods:
        generate(x, temperature, deterministic): Generates textual output from input audio sequences.

    Whisper uses an encoder-decoder Transformer (Vaswani et al., 2017) as this, All  audio is re-sampled to 16,000 Hz, and an 80-channel logmagnitude Mel spectrogram representation is computed on
    25-millisecond windows with a stride of 10 milliseconds. For feature normalization, we globally scale the input to be between -1 and 1 with approximately zero mean across
    the pre-training dataset. The encoder processes this input representation with a small stem consisting of two convolution layers with a filter width of 3 and the GELU activation
    function (Hendrycks & Gimpel, 2016) where the second convolution layer has a stride of two. Sinusoidal position embeddings are then added to the output of the stem after
    which the encoder Transformer blocks are applied. The transformer uses pre-activation residual blocks (Child et al., 2019), and a final layer normalization is applied to the encoder output. The decoder uses learned position embeddings
    and tied input-output token representations (Press & Wolf, 2017). The encoder and decoder have the same width and number of transformer blocks. Figure 1 summarizes the model architecture
    https://cdn.openai.com/papers/whisper.pdf

    Example usage:
        ```py
        import jax
        import jax.numpy as jnp
        from nanodl import ArrayDataset, DataLoader
        from nanodl import Whisper, WhisperDataParallelTrainer

        # Dummy data parameters
        batch_size = 8
        max_length = 50
        embed_dim = 256
        vocab_size = 1000

        # Generate data: replace with actual tokenised/quantised data
        dummy_targets = jnp.ones((101, max_length), dtype=jnp.int32)
        dummy_inputs = jnp.ones((101, max_length, embed_dim))

        dataset = ArrayDataset(dummy_inputs,
                            dummy_targets)

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
            'embed_dim': embed_dim,
            'max_length': max_length,
            'start_token': 0,
            'end_token': 50,
        }

        # Initialize model
        model = Whisper(**hyperparams)
        rngs = {'params': jax.random.key(0), 'dropout': jax.random.key(1)}
        params = model.init(rngs, dummy_inputs, dummy_targets)['params']
        outputs = model.apply({'params': params}, dummy_inputs, dummy_targets, rngs=rngs)
        print(outputs.shape)

        # Training on your data
        trainer = WhisperDataParallelTrainer(model,
                                            dummy_inputs.shape,
                                            dummy_targets.shape,
                                            'params.pkl')
        trainer.train(dataloader, 2, dataloader)

        # Sample inference
        params = trainer.load_params('params.pkl')

        # for more than one sample, use model.generate_batch
        transcripts = model.apply({'params': params},
                                dummy_inputs[:1],
                                rngs=rngs,
                                method=model.generate)

        print(transcripts)
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
        self.encoder = WhisperSpeechEncoder(
            hidden_dim=self.hidden_dim,
            num_heads=self.num_heads,
            num_layers=self.num_layers,
            feedforward_dim=self.feedforward_dim,
            dropout=self.dropout,
        )

        self.decoder = WhisperTextDecoder(
            hidden_dim=self.hidden_dim,
            num_heads=self.num_heads,
            num_layers=self.num_layers,
            feedforward_dim=self.feedforward_dim,
            dropout=self.dropout,
            max_len=self.max_length,
            vocab_size=self.vocab_size,
            embed_dim=self.embed_dim,
        )

    def __call__(
        self, x: jnp.ndarray, y: jnp.ndarray, training: bool = False
    ) -> jnp.ndarray:

        z = self.encoder(x=x, training=training)[0]
        return self.decoder(x=y, context=z, training=training)[0]

    def generate(
        self, x: jnp.ndarray, temperature: float = 1.0, deterministic: bool = False
    ) -> Tuple[jnp.ndarray]:

        # Encode the input sequence
        encoded_sequence = self.encoder(x=x, training=False)[0]

        decoder_input = jnp.array([[self.start_token]])
        output_sequence = []

        # Autoregressive decoding loop
        for _ in range(self.max_length):
            decoder_output = self.decoder(
                x=decoder_input, context=encoded_sequence, training=False
            )[0]
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
        self, x: jnp.ndarray, temperature: float = 1.0, deterministic: bool = False
    ) -> jnp.ndarray:

        # Encode the input sequence
        encoded_sequence = self.encoder(x=x, training=False)[0]

        batch_size = x.shape[0] if x is not None else 1
        decoder_input = jnp.full((batch_size, 1), self.start_token)
        output_sequences = jnp.zeros((batch_size, self.max_length), dtype=jnp.int32)

        for i in range(self.max_length):
            decoder_output = self.decoder(
                decoder_input, context=encoded_sequence, training=False
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

            if jnp.all(next_token == self.end_token):
                break

        return output_sequences


class WhisperDataParallelTrainer:
    """
    Trainer class using data parallelism with JAX.
    This trainer leverages JAX's `pmap` for parallel training across multiple devices (GPUs/TPUs).
    It handles the model training loop, including gradient computation, parameter updates, and evaluation.

    Attributes:
        model (Any): The model to be trained.
        input_shape (Tuple[int, ...]): The shape of the input tensor.
        tarhet_shape (Tuple[int, ...]): The shape of the image target tensor.
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
        target_shape: Tuple[int, ...],
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
            WhisperDataParallelTrainer.train_step, axis_name="devices"
        )
        self.evaluation_step = jax.pmap(
            WhisperDataParallelTrainer.evaluation_step, axis_name="devices"
        )
        self.state = self.create_train_state(learning_rate, input_shape, target_shape)
        print(f"Number of accelerators: {self.num_devices}")

    def create_train_state(
        self,
        learning_rate: float,
        input_shape: Tuple[int, ...],
        target_shape: Tuple[int, ...],
    ) -> Any:

        rngs = {"params": jax.random.key(0), "dropout": jax.random.key(1)}
        params = self.model.init(
            rngs,
            jnp.ones(input_shape, dtype=jnp.int32),
            jnp.ones(target_shape, dtype=jnp.int32),
        )["params"]

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
                targets,
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
                inputs = inputs.reshape(
                    (
                        self.num_devices,
                        batch_size_per_device,
                        inputs.shape[1],
                        inputs.shape[2],
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
            {"params": state.params},
            inputs,
            targets,
            rngs={"dropout": jax.random.PRNGKey(2)},
        )
        return optax.softmax_cross_entropy_with_integer_labels(logits, targets).mean()

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
