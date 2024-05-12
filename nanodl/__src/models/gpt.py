import time
from typing import Any, Iterable, Optional, Tuple

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import optax
from flax.training import train_state


class SelfMultiHeadAttention(nn.Module):
    """
    Implements multi-head self-attention mechanism as described in "Attention is All You Need" by Vaswani et al 2017.

    This module splits the input into multiple heads, applies scaled dot-product attention independently on each head, and then concatenates the results. It allows the model to jointly attend to information from different representation subspaces at different positions.

    Attributes:
        hidden_dim (int): Dimensionality of the input and output features.
        num_heads (int): Number of attention heads.

    """

    hidden_dim: int
    num_heads: int

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


class GPT3Block(nn.Module):
    """
    A block of GPT-3 consisting of multi-head self-attention and feedforward neural network layers.

    This class implements a transformer block used in GPT-3, which includes two self-attention layers followed by a position-wise feedforward network. Layer normalization and dropout are applied for regularization and to prevent overfitting.

    Attributes:
        hidden_dim (int): Dimension of the hidden layer.
        num_heads (int): Number of attention heads in the multi-head attention mechanism.
        feedforward_dim (int): Dimension of the feedforward layer.
        dropout (float): Dropout rate for regularization.

    """

    hidden_dim: int
    num_heads: int
    feedforward_dim: int
    dropout: float

    def setup(self):
        self.attention1 = SelfMultiHeadAttention(
            hidden_dim=self.hidden_dim, num_heads=self.num_heads
        )
        self.attention2 = SelfMultiHeadAttention(
            hidden_dim=self.hidden_dim, num_heads=self.num_heads
        )
        self.feed_forward = PositionWiseFFN(self.feedforward_dim, self.hidden_dim)
        self.norm1 = nn.LayerNorm()
        self.norm2 = nn.LayerNorm()
        self.norm3 = nn.LayerNorm()
        self.dropout1 = nn.Dropout(self.dropout)
        self.dropout2 = nn.Dropout(self.dropout)
        self.dropout3 = nn.Dropout(self.dropout)

    def causal_mask(
        self, batch_size: int, destination_dim: int, source_dim: int
    ) -> jnp.ndarray:

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

        x = self.norm1(x)
        attended_x, attention1 = self.attention1(x, mask=mask)
        x = self.dropout1(x, deterministic=not training)
        x += attended_x

        x = self.norm2(x)
        attended_x, attention2 = self.attention2(x, mask=mask)
        x = self.dropout2(x, deterministic=not training)
        x += attended_x

        x = self.norm3(x)
        output = self.feed_forward(x)
        x = self.dropout3(output, deterministic=not training)
        x += attended_x

        return x, jnp.array(attention1), jnp.array(attention2)


class GPT3Decoder(nn.Module):
    """
    Implements the decoder component of the GPT-3 model.

    The decoder consists of multiple layers of GPT-3 blocks that process sequences of tokens to generate predictions. It includes an embedding layer to map tokens to high-dimensional vectors and an output layer to map the representations to logits over the vocabulary.

    Attributes:
        num_layers (int): The number of GPT-3 blocks in the decoder.
        hidden_dim (int): The dimensionality of the input and output features for the blocks.
        num_heads (int): The number of attention heads in each GPT-3 block.
        feedforward_dim (int): The dimensionality of the inner layer of the feed-forward networks in the blocks.
        dropout (float): The dropout rate used in the decoder for regularization.
        vocab_size (int): The size of the vocabulary.
        embed_dim (int): The dimensionality of the token embeddings.

    """

    num_layers: int
    hidden_dim: int
    num_heads: int
    feedforward_dim: int
    dropout: float
    vocab_size: int
    embed_dim: int

    def setup(self):
        self.embedding = nn.Embed(
            num_embeddings=self.vocab_size, features=self.embed_dim
        )

        self.layers = [
            GPT3Block(
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


class GPT3(nn.Module):
    """
    Implements the GPT-3 model for autoregressive language modeling.

    GPT-3 is a transformer-based model designed to generate text by predicting the next token in a sequence given the previous tokens. It can be used for a wide range of natural language processing tasks including text generation, completion, and more.

    Attributes:
        num_layers (int): The number of layers (blocks) in the GPT-3 model.
        hidden_dim (int): The dimensionality of the input and output features for the blocks.
        num_heads (int): The number of attention heads in each block.
        feedforward_dim (int): The dimensionality of the inner layer of the feed-forward networks in the blocks.
        dropout (float): The dropout rate used in the model for regularization.
        vocab_size (int): The size of the vocabulary.
        embed_dim (int): The dimensionality of the token embeddings.
        max_length (int): The maximum length of the generated sequences.
        start_token (int): The token used to start the generation process.
        end_token (int): The token that indicates the end of a generated sequence.

    Methods:
        generate(x, temperature, deterministic): Generates a sequence of tokens autoregressively, starting from an optional initial sequence.
        generate_batch(x, temperature, deterministic): Generates sequences of tokens for a batch of initial sequences autoregressively.

    The motivation behind GPT is to create a highly effective language model that can understand and generate human-like text.
    Its architecture is a decoder-only transformer trained on next-token prediction and generates autoregressively duting training.
    It's pre-trained on a massive amount of text data, which allows it to learn the patterns and nuances of language.
    GPT's strength lies in its ability to generalize this knowledge to perform a wide range of natural language processing tasks without the need for extensive task-specific training,
    making it a powerful tool for various applications in language understanding and generation.
    GPT3 uses prelayer normalisation opposed to classic transformers

    Note:
    This implementation excludes the modified initialization which accounts for the accumulation on the residual path with model depth.
    Such an intialisation involves scaling the weights of residual layers at initialization by a factor of 1/√N where N is the number of residual layers.
    Rather we use 'Xavier' initialization (https://proceedings.mlr.press/v9/glorot10a.html) for the weights and 'zeros' for the biases.


    example usage:
        ```py
        import jax
        import jax.numpy as jnp
        from nanodl import ArrayDataset, DataLoader
        from nanodl import GPT3, GPTDataParallelTrainer

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
        model = GPT3(**hyperparams)
        rngs = jax.random.PRNGKey(0)
        rngs, dropout_rng = jax.random.split(rngs)
        params = model.init({'params': rngs, 'dropout': dropout_rng}, dummy_inputs)['params']

        # Call as you would a Jax/Flax model
        outputs = model.apply({'params': params},
                            dummy_inputs,
                            rngs={'dropout': dropout_rng})
        print(outputs.shape)

        # Training on data
        trainer = GPTDataParallelTrainer(model, dummy_inputs.shape, 'params.pkl')
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
    hidden_dim: int
    num_heads: int
    feedforward_dim: int
    dropout: float
    vocab_size: int
    embed_dim: int
    max_length: int
    start_token: int
    end_token: int

    def setup(self):
        self.decoder = GPT3Decoder(
            self.num_layers,
            self.embed_dim,
            self.num_heads,
            self.feedforward_dim,
            self.dropout,
            self.vocab_size,
            self.embed_dim,
        )

    def __call__(
        self, x: jnp.ndarray, training: bool = True, drop_last_layer: bool = False
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


class SparseMixtureOfExperts(nn.Module):
    """
    Mixture of Experts Layer with Top-K Gating.

    This layer consists of multiple expert feed-forward networks and a gating mechanism
    that determines the contribution of each expert based on the input. Unlike the
    traditional Mixture of Experts, this implementation only computes the outputs
    for the top K experts as determined by the gating mechanism for each input.

    Attributes:
        num_hiddens (int): Number of hidden units in each expert.
        num_outputs (int): Number of output units in the final layer after combining expert outputs.
        num_experts (int): Number of experts in the mixture.
        top_k (int): Number of top experts to use for each input instance.

    Args:
        num_hiddens (int): Number of hidden units in each expert network.
        num_outputs (int): Number of output units in the final layer.
        num_experts (int): Number of experts.
        top_k (int): Number of top experts to use for each input instance.

    """

    num_hiddens: int
    num_outputs: int
    num_experts: int
    top_k: int  # Number of top experts to use each pass

    def setup(self):
        self.experts = [
            PositionWiseFFN(self.num_hiddens, self.num_outputs)
            for _ in range(self.num_experts)
        ]
        self.gate = nn.Dense(
            self.num_experts, kernel_init=nn.initializers.xavier_uniform()
        )
        self.dense_final = nn.Dense(
            self.num_outputs, kernel_init=nn.initializers.xavier_uniform()
        )

    def __call__(self, X: jnp.ndarray) -> jnp.ndarray:
        gating_weights = nn.softmax(self.gate(X), axis=-1)
        top_k_indices = jnp.argsort(gating_weights, axis=-1)[..., -self.top_k :]
        expert_outputs = jnp.stack([expert(X) for expert in self.experts], axis=2)

        # Select only the top K expert outputs
        batch_size, seq_length, _ = X.shape
        batch_indices = jnp.arange(batch_size)[:, None, None]
        seq_indices = jnp.arange(seq_length)[None, :, None]
        top_k_expert_outputs = expert_outputs[batch_indices, seq_indices, top_k_indices]

        # Compute the gating weights for the selected top K experts
        top_k_gating_weights = jnp.take_along_axis(
            gating_weights, top_k_indices, axis=-1
        )
        mixed_expert_output = jnp.sum(
            top_k_gating_weights[..., None] * top_k_expert_outputs, axis=2
        )
        return self.dense_final(mixed_expert_output)


class GPT4Block(nn.Module):
    """
    Implements a transformer block for GPT-4 with self-attention, feed-forward layers, and a sparse mixture of experts.

    This block extends the GPT-3 architecture by incorporating a sparse mixture of experts (MoE) for the feed-forward layer, enabling the model to route information through a subset of expert networks based on the input context, potentially increasing model capacity and efficiency.

    Attributes:
        hidden_dim (int): Dimensionality of the input and output features.
        num_heads (int): Number of attention heads in the multi-head self-attention mechanism.
        feedforward_dim (int): Dimensionality of the inner layer of the feed-forward network.
        dropout (float): Dropout rate for regularization.
        num_experts (int): Number of experts in the sparse mixture of experts layer.
        top_k (int): Number of experts to be activated for each input in the sparse mixture of experts layer.

    """

    hidden_dim: int
    num_heads: int
    feedforward_dim: int
    dropout: float
    num_experts: int
    top_k: int

    def setup(self):
        self.attention1 = SelfMultiHeadAttention(
            hidden_dim=self.hidden_dim, num_heads=self.num_heads
        )
        self.attention2 = SelfMultiHeadAttention(
            hidden_dim=self.hidden_dim, num_heads=self.num_heads
        )
        self.feed_forward = SparseMixtureOfExperts(
            self.feedforward_dim, self.hidden_dim, self.num_experts, self.top_k
        )
        self.norm1 = nn.LayerNorm()
        self.norm2 = nn.LayerNorm()
        self.norm3 = nn.LayerNorm()
        self.dropout1 = nn.Dropout(self.dropout)
        self.dropout2 = nn.Dropout(self.dropout)
        self.dropout3 = nn.Dropout(self.dropout)

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

        x = self.norm1(x)
        attended_x, attention1 = self.attention1(x, mask=mask)
        x = self.dropout1(x, deterministic=not training)
        x += attended_x

        x = self.norm2(x)
        attended_x, attention2 = self.attention2(x, mask=mask)
        x = self.dropout2(x, deterministic=not training)
        x += attended_x

        x = self.norm3(x)
        output = self.feed_forward(x)
        x = self.dropout3(output, deterministic=not training)
        x += attended_x

        return x, jnp.array(attention1), jnp.array(attention2)


class GPT4Decoder(nn.Module):
    """
    Implements the decoder component of the GPT-4 model.

    The decoder is composed of multiple GPT-4 blocks, processing sequences of tokens to generate text. It includes an embedding layer to convert tokens into vectors and an output layer to predict the next token in the sequence.

    Attributes:
        num_layers (int): Number of GPT-4 blocks in the decoder.
        hidden_dim (int): Dimensionality of the input and output features for the blocks.
        num_heads (int): Number of attention heads in each GPT-4 block.
        feedforward_dim (int): Dimensionality of the inner layer of the feed-forward networks in the blocks.
        dropout (float): Dropout rate used for regularization.
        vocab_size (int): Size of the vocabulary.
        embed_dim (int): Dimensionality of the token embeddings.
        num_experts (int): Number of experts in the sparse mixture of experts layer in each GPT-4 block.
        top_k (int): Number of experts to be activated for each input in the sparse mixture of experts layer in each GPT-4 block.

    """

    num_layers: int
    hidden_dim: int
    num_heads: int
    feedforward_dim: int
    dropout: float
    vocab_size: int
    embed_dim: int
    num_experts: int
    top_k: int

    def setup(self):
        self.embedding = nn.Embed(
            num_embeddings=self.vocab_size, features=self.embed_dim
        )

        self.layers = [
            GPT4Block(
                self.hidden_dim,
                self.num_heads,
                self.feedforward_dim,
                self.dropout,
                self.num_experts,
                self.top_k,
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


class GPT4(nn.Module):
    """
    Implements the GPT-4 model for autoregressive language modeling with a sparse mixture of experts.

    GPT-4 builds upon the GPT-3 architecture by introducing a sparse mixture of experts in the feed-forward layers, aiming to enhance model capacity and efficiency. It is designed for tasks such as text generation, completion, and more.

    Attributes:
        num_layers (int): Number of layers (blocks) in the GPT-4 model.
        hidden_dim (int): Dimensionality of the input and output features for the blocks.
        num_heads (int): Number of attention heads in each block.
        feedforward_dim (int): Dimensionality of the inner layer of the feed-forward networks in the blocks.
        dropout (float): Dropout rate used for regularization.
        vocab_size (int): Size of the vocabulary.
        embed_dim (int): Dimensionality of the token embeddings.
        max_length (int): Maximum length of the generated sequences.
        start_token (int): Token used to start the generation process.
        end_token (int): Token that indicates the end of a generated sequence.
        num_experts (int): Number of experts in the sparse mixture of experts layer.
        top_k (int): Number of experts to be activated for each input in the sparse mixture of experts layer.

    Methods:
        generate(x, temperature, deterministic): Generates a sequence of tokens autoregressively.
        generate_batch(x, temperature, deterministic): Generates sequences of tokens for a batch of initial sequences autoregressively.

    example usage:
        ```py
        import jax
        import jax.numpy as jnp
        from nanodl import ArrayDataset, DataLoader
        from nanodl import GPT4, GPTDataParallelTrainer

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
        model = GPT4(**hyperparams)
        rngs = jax.random.PRNGKey(0)
        rngs, dropout_rng = jax.random.split(rngs)
        params = model.init({'params': rngs, 'dropout': dropout_rng}, dummy_inputs)['params']

        # Call as you would a Jax/Flax model
        outputs = model.apply({'params': params},
                            dummy_inputs,
                            rngs={'dropout': dropout_rng})
        print(outputs.shape)

        # Training on data
        trainer = GPTDataParallelTrainer(model, dummy_inputs.shape, 'params.pkl')
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
    hidden_dim: int
    num_heads: int
    feedforward_dim: int
    dropout: float
    vocab_size: int
    embed_dim: int
    max_length: int
    start_token: int
    end_token: int
    num_experts: int = 10
    top_k: int = 2

    def setup(self):
        self.decoder = GPT4Decoder(
            self.num_layers,
            self.hidden_dim,
            self.num_heads,
            self.feedforward_dim,
            self.dropout,
            self.vocab_size,
            self.embed_dim,
            self.num_experts,
            self.top_k,
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


class GPTDataParallelTrainer:
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
            GPTDataParallelTrainer.train_step, axis_name="devices"
        )
        self.evaluation_step = jax.pmap(
            GPTDataParallelTrainer.evaluation_step, axis_name="devices"
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
