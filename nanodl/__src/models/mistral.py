'''
Mistral 7B is a large language model (LLM) designed for enhanced efficiency and performance. It utilizes Grouped-Query Attention (GQA) to achieve quicker inference times. 
It incorporates Sliding Window Attention (SWA), enabling it to efficiently process sequences of any length while minimizing the cost of inference. 
Additionally, the ReLU non-linearity is replaced with the SwiGLU activation function, which is a variant of the GLU activation function.
Absolute positional embeddings are replaced with rotary positional embeddings (RoPE), implemented at each layer of the network. For specific hyper-parameter details, refer to Table 2 in the document.

Mixtral is an architectural upgrade within Mistral. Leverages "Sparse Mixture-of-Experts" (MoE). Each layer has 8 expert groups, 
but a "router network" selects only 2 relevant ones per token, reducing active calculations and boosting efficiency.

Example usage:
```
import jax
import jax.numpy as jnp
from nanodl import ArrayDataset, DataLoader
from nanodl import Mistral, MistralDataParallelTrainer

# Generate dummy data
batch_size = 8
max_length = 10

# Replace with actual tokenised data
data = jnp.ones((101, max_length+1), dtype=jnp.int32)

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
    'window_size': 5,
    'shift_size': 2
}

# Initialize model
model = Mistral(**hyperparams)
rngs = jax.random.PRNGKey(0)
rngs, dropout_rng = jax.random.split(rngs)
params = model.init({'params': rngs, 'dropout': dropout_rng}, dummy_inputs)['params']

# Call as you would a Jax/Flax model
outputs = model.apply({'params': params}, 
                      dummy_inputs, 
                      rngs={'dropout': dropout_rng})
print(outputs.shape)

# Training on data
trainer = MistralDataParallelTrainer(model, dummy_inputs.shape, 'params.pkl')
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
'''

import jax
import flax
import time
import optax
import jax.numpy as jnp
import flax.linen as nn
from flax.training import train_state
from typing import Tuple, Any, Optional, Iterable


class RotaryPositionalEncoding():
    """
    The rotary position embeddings from RoFormer_ (Su et. al).
    A crucial insight from the method is that the query and keys are
    transformed by rotation matrices which depend on the relative positions.

    Other implementations are available in the Rotary Transformer repo_ and in
    GPT-NeoX_, GPT-NeoX was an inspiration

    .. _RoFormer: https://arxiv.org/abs/2104.09864
    .. _repo: https://github.com/ZhuiyiTechnology/roformer
    .. _GPT-NeoX: https://github.com/EleutherAI/gpt-neox


    .. This is implemented outside nn module as is modifies an external state
       It is also puporsefully broken down for explainability
    """

    def __init__(self, dim_model: int):
        """
        Args:
            dim_model: The dimension of the input and output embeddings.
        """
        super().__init__()
        self.dim_model = dim_model

        inv_freq = 1.0 / (10000 ** (jnp.arange(0, dim_model, 2, dtype=jnp.float32) / dim_model))
        self.inv_freq = inv_freq

        self._seq_len_cached = None
        self._cos_cached = None
        self._sin_cached = None

    def _update_cos_sin_tables(self, x, seq_dimension=1):
        """
        Update the cached cosine and sine tables, if necessary.

        Args:
            x: The input tensor, of shape `(batch_size, seq_len, dim)`.
            seq_dimension: The dimension that represents the sequence length.

        Returns:
            The updated cosine and sine tables.
        """
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
        """
        Split the input tensor into two halves, rotate the second half by 180 degrees, and concatenate the two halves back together.

        Args:
            x: The input tensor, of shape `(batch_size, seq_len, dim)`.

        Returns:
            The rotated tensor.
        """
        x1, x2 = jnp.split(x, 2, axis=-1)
        return jnp.concatenate((-x2, x1), axis=-1)

    def apply_rotary_pos_emb(self, x, cos, sin):
        """
         Apply the rotary position embeddings to the input tensor.

        Args:
            x: The input tensor, of shape `(batch_size, seq_len, dim)`.
            cos: The cosine table, of shape `(batch_size, 1, seq_len, dim)`.
            sin: The sine table, of shape `(batch_size, 1, seq_len, dim)`.

        Returns:
            The embedded tensor.
        """
        cos = cos[:, :, : x.shape[-2], :]
        sin = sin[:, :, : x.shape[-2], :]
        return (x * cos) + (self.rotate_half(x) * sin)

    def __call__(self, q, k):
        """
         Apply the rotary position embeddings to the query and key tensors.

        Args:
            q: The query tensor, of shape `(batch_size, seq_len, dim)`.
            k: The key tensor, of shape `(batch_size, seq_len, dim)`.

        Returns:
            The embedded query and key tensors.
        """
        self._cos_cached, self._sin_cached = self._update_cos_sin_tables(k, seq_dimension=-2)

        return (
            self.apply_rotary_pos_emb(q, self._cos_cached, self._sin_cached)[0],
            self.apply_rotary_pos_emb(k, self._cos_cached, self._sin_cached)[0],
        )
    

class GroupedRotaryShiftedWindowMultiHeadAttention(nn.Module):
    """
    Attention which uses RoPE, Grouped Query and Sliding Window Attention.
    """
    hidden_dim : int  # Output dimension
    num_heads : int  # Number of parallel heads
    num_groups : int  # Number of groups to split the heads into
    window_size: int
    shift_size: int

    def setup(self):
        self.query_projection = nn.Dense(self.hidden_dim // self.num_heads,
                                 kernel_init=nn.initializers.xavier_uniform(),
                                 bias_init=nn.initializers.zeros,
                                )
        self.key_projection = nn.Dense(self.hidden_dim // (self.num_heads * self.num_groups),
                                 kernel_init=nn.initializers.xavier_uniform(),
                                 bias_init=nn.initializers.zeros 
                                )
        self.value_projection = nn.Dense(self.hidden_dim // (self.num_heads * self.num_groups),
                                 kernel_init=nn.initializers.xavier_uniform(),
                                 bias_init=nn.initializers.zeros 
                                )
        self.rope = RotaryPositionalEncoding(self.hidden_dim // self.num_groups)
        self.output = nn.Dense(self.hidden_dim,
                               kernel_init=nn.initializers.xavier_uniform(),
                               bias_init=nn.initializers.zeros)

    def __call__(self, 
                 inputs: jnp.ndarray, 
                 context: jnp.ndarray, 
                 mask: jnp.ndarray) -> tuple:

        """
        Args:
            inputs: inputs ((batch_size, seq_len, dims))
            context: optional - context ((batch_size, seq_len, dims))
            Mask: optional - masks where reqions to ignore are flipped to os
                  regions to attend to are 1s (batch_size, seq_len, dims)

        Return: outputs (batch_size, seq_len, seq_len)
                attention matrixes (batch_size, heads, seq_len, seq_len)
        """
        query = self.query_projection(inputs)
        key = self.key_projection(context)
        value = self.value_projection(context)
        
        # Break query into groups and transpose to (num_groups, batch_size, seq_len, dims)
        # This will allow vmapping over the groups for parallelization
        grouped_query = jnp.reshape(query, (query.shape[0], query.shape[1], self.num_groups, -1))
        grouped_query = jnp.repeat(grouped_query, self.num_heads, axis=-1)
        grouped_query = jnp.transpose(grouped_query, (2, 0, 1, 3))

        # Repeat the key and values
        key = jnp.repeat(key, self.num_heads, axis=-1)
        value = jnp.repeat(value, self.num_heads, axis=-1)
        
        # Vectorize the process_group function
        vectorized_process_group = jax.vmap(self.process_group, in_axes=(0, None, None, None))
        results = vectorized_process_group(grouped_query, key, value, mask)

        # Merge the groups back together
        context_vectors = jnp.concatenate(results[0], axis=-1)
        return self.output(context_vectors), results[1]
    
    def process_group(self, query, key, value, mask):
        query, key = self.rope(query, key)
        query_windows = self.window_partition(query)
        key_windows = self.window_partition(key)
        value_windows = self.window_partition(value)
        attention_windows, attention_maps = self.attention_function(query_windows, 
                                                                    key_windows, 
                                                                    value_windows,
                                                                    mask)

        attention_windows = jnp.roll(attention_windows, -self.shift_size, axis=1)
        merged = attention_windows.transpose((1, 0, 2, 3))
        return jnp.reshape(merged, query.shape), attention_maps

    def window_partition(self, x):
        B, N, C = x.shape
        assert N % self.window_size == 0, "Sequence length must be a multiple of the window size"
        windows = jnp.reshape(x, (B, -1, self.window_size, C))  # (batch_size, num_windows, window_size, dim)
        windows = windows.transpose((1, 0, 2, 3))  # Transpose to (num_windows, batch_size, window_size, dim)
        return windows

    def attention_function(self, query, key, value, mask):
        input_length = query.shape[-2]
        context_length = key.shape[-2]
        head_dim = query.shape[-1] // self.num_heads
        dim_key = key.shape[-1]

        # Split keys, and values into heads
        query_heads = jnp.reshape(query, (query.shape[0], query.shape[1], self.num_heads, input_length, head_dim))
        key_heads = jnp.reshape(key, (key.shape[0], key.shape[1], self.num_heads, context_length, head_dim))
        value_heads = jnp.reshape(value, (value.shape[0], value.shape[1], self.num_heads, context_length, head_dim))

        attention_scores = jnp.matmul(query_heads, key_heads.transpose(0, 1, 2, 4, 3)) / jnp.sqrt(dim_key)

        if mask is not None:
            mask = self.causal_mask(attention_scores.shape)
            attention_scores = attention_scores * mask

        attention_weights = jax.nn.softmax(attention_scores, axis=-1)
        attended_values = jnp.matmul(attention_weights, value_heads)
        attended_values = attended_values.transpose(0, 1, 3, 2, 4)
        attended_values = jnp.reshape(attended_values, (query.shape[0], query.shape[1], input_length, query.shape[-1]))
        return attended_values, attention_weights
    
    def causal_mask(self, 
                shape: Tuple[int, ...]) -> jnp.ndarray:
        """
        Generate a causal mask for attention.

        Args:
            batch_size (int): Batch size.
            destination_dim (int): Dimension of the destination sequence.
            source_dim (int): Dimension of the source sequence.

        Returns:
            jnp.ndarray: Causal mask with shape (batch_size, num_heads, destination_dim, source_dim).
        """
        # Create index tensors for the source and destination dimensions
        source_dim, destination_dim = shape[-2], shape[-2]
        idx_source = jnp.arange(destination_dim)[:, None]
        idx_destination = jnp.arange(source_dim)
        mask = idx_source >= idx_destination - source_dim + destination_dim
        mask = mask.astype(jnp.int32) 

        # Expand dimensions to match the required output shape
        mask = mask[None, None, None, :, :]
        return jnp.broadcast_to(mask, (shape[0], shape[1], shape[2], destination_dim, source_dim))
    

class PositionWiseFFN(nn.Module):
    """
    Position-wise Feed-Forward Network which incorporates SwiGLU activation.

    Args:
        num_hiddens (int): Number of hidden units in the feed-forward layers.
        num_outputs (int): Number of output units in the feed-forward layers.
    """
    hidden_dim: int
    dim: int

    def setup(self):
        self.dense1 = nn.Dense(self.hidden_dim, kernel_init=nn.initializers.xavier_uniform())
        self.dense2 = nn.Dense(self.dim, kernel_init=nn.initializers.xavier_uniform())
        self.dense3 = nn.Dense(self.hidden_dim, kernel_init=nn.initializers.xavier_uniform())

    def __call__(self, X: jnp.ndarray) -> jnp.ndarray:
        return self.dense2(nn.silu(self.dense1(X) * self.dense3(X)))
    
    
class MistralDecoderBlock(nn.Module):
    """
    Transformer Decoder Block.

    Args:
        hidden_dim (int): Input dimension.
        num_heads (int): Number of attention heads.
        feedforward_dim (int): Dimension of the feed-forward network.
        dropout (float): Dropout rate.
    """
    hidden_dim: int
    num_heads: int
    feedforward_dim: int
    dropout: float
    num_groups: int
    window_size: int
    shift_size: int

    def setup(self):
        self.attention1 = GroupedRotaryShiftedWindowMultiHeadAttention(hidden_dim=self.hidden_dim, 
                                                          num_heads=self.num_heads,
                                                          num_groups=self.num_groups,
                                                          window_size=self.window_size,
                                                          shift_size=self.shift_size)
        
        self.attention2 = GroupedRotaryShiftedWindowMultiHeadAttention(hidden_dim=self.hidden_dim, 
                                                          num_heads=self.num_heads,
                                                          num_groups=self.num_groups,
                                                          window_size=self.window_size,
                                                          shift_size=self.shift_size)
        
        self.feed_forward = PositionWiseFFN(self.feedforward_dim, self.hidden_dim)
        self.norm1 = nn.RMSNorm(self.dropout)
        self.norm2 = nn.RMSNorm(self.dropout)
        self.norm3 = nn.RMSNorm(self.dropout)
        self.dropout1 = nn.Dropout(self.dropout)
        self.dropout2 = nn.Dropout(self.dropout)
        self.dropout3 = nn.Dropout(self.dropout)

    def __call__(self, 
                x: jnp.ndarray,
                training: bool = False) -> tuple:
        """
        Apply the DecoderBlock to input data.

        Args:
            x (jnp.ndarray): Input tensor.
            context (jnp.ndarray): Context tensor.
            mask (jnp.ndarray, optional): Mask tensor. Defaults to None.
            training (bool): Training mode.

        Returns:
            tuple: Output tensor, attention tensor, and cross-attention tensor.
        """
        x = self.norm1(x)
        attended_x, attention1 = self.attention1(x, x, mask=True)
        x = self.dropout1(x, deterministic=not training)
        x += attended_x

        x = self.norm2(x)
        attended_x, attention2 = self.attention2(x, x, mask=True)
        x = self.dropout2(x, deterministic=not training)
        x += attended_x

        x = self.norm3(x)
        output = self.feed_forward(x)
        x = self.dropout3(x, deterministic=not training)
        x += output

        return x, jnp.array(attention1), jnp.array(attention2)


class MistralDecoder(nn.Module):
    """
    Transformer Decoder.

    Args:
        num_layers (int): Number of decoder layers.
        hidden_dim (int): Input dimension.
        num_heads (int): Number of attention heads.
        feedforward_dim (int): Dimension of the feed-forward network.
        dropout (float): Dropout rate.
    """
    num_layers: int
    hidden_dim: int
    num_heads: int
    num_groups: int
    feedforward_dim: int
    dropout: float
    vocab_size: float
    embed_dim: float
    window_size: int
    shift_size: int

    def setup(self):
        self.embedding = nn.Embed(num_embeddings=self.vocab_size, 
                                  features=self.embed_dim)
        
        self.layers = [MistralDecoderBlock(self.hidden_dim, 
                                    self.num_heads, 
                                    self.feedforward_dim, 
                                    self.dropout,
                                    self.num_groups,
                                    self.window_size,
                                    self.shift_size) for _ in range(self.num_layers)]
        
        self.outputs = nn.Dense(self.vocab_size)
        

    def __call__(self, 
                 x: jnp.ndarray,
                 training: bool = False,
                 drop_last_layer: bool = False) -> tuple:
        """
        Apply the TransformerDecoder to input data.

        Args:
            x (jnp.ndarray): Input tensor.
            context (jnp.ndarray): Context tensor.
            mask (jnp.ndarray, optional): Mask tensor. Defaults to None.
            training (bool): Training mode.

        Returns:
            tuple: Output tensor, list of attention tensors, and list of cross-attention tensors.
            each attention map has dim (num_layers, batch_size, num_heads, seq_length, seq_length)
        """
        attention_maps = []
        x = self.embedding(x)
        cross_attention_maps = []
        for layer in self.layers:
            x, attention, cross_attention = layer(x, training=training)
            attention_maps.append(attention)
            cross_attention_maps.append(cross_attention)

        if not drop_last_layer:
            x = self.outputs(x)

        return x, jnp.array(attention_maps), jnp.array(cross_attention_maps)
    


class Mistral(nn.Module):
    """
    Args:
        num_layers (int): Number of layers in the encoder and decoder.
        num_heads (int): Number of attention heads in the multi-head attention layers.
        hidden_dim (int): Dimensionality of input embeddings.
        feedforward_dim (int): Dimensionality of the feedforward layers.
        dropout (float): Dropout probability.
        vocab_size (int): Size of the vocabulary.
        embed_dim (int): Dimensionality of token embeddings.
        max_length (int): Maximum length of generated sequences.
        start_token (int): Token ID for the start of sequence.
        end_token (int): Token ID for the end of sequence.
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
    window_size: int
    shift_size: int

    def setup(self):
        """
        Initialize the Mistral model 
        """
        self.decoder = MistralDecoder(self.num_layers,
                                self.hidden_dim,
                                self.num_heads,
                                self.num_groups,
                                self.feedforward_dim,
                                self.dropout,
                                self.vocab_size,
                                self.embed_dim,
                                self.window_size,
                                self.shift_size)
        
    def __call__(self, 
                 x: jnp.ndarray,
                 training: bool = False,
                 drop_last_layer: bool = False) -> jnp.ndarray:
        
        """ 
        Sequence-to-sequence models use teacher forcing during training and as such, 
        the decoder input is the ground truth sequence.
        """
        return self.decoder(x=x, 
                            training=training,
                            drop_last_layer=drop_last_layer)[0]
    
    def zero_pad(self, arr, max_length):
        """Zero-pad the given array to the specified maximum length along axis=1."""
        current_length = arr.shape[1] 
        num_zeros = max_length - current_length 

        if num_zeros > 0:
            zeros = jnp.zeros((arr.shape[0], num_zeros), dtype=arr.dtype)
            padded_array = jnp.concatenate([arr, zeros], axis=1)
        else:
            padded_array = arr

        return padded_array
    

    def generate(self, 
                 x: Optional[jnp.ndarray] = None,
                 temperature: float = 1.0,
                 deterministic: bool = False) -> Tuple[jnp.ndarray]:
        """
        Generate sequences either from scratch or continues from the input sequence.

        Args:
            x (jax.numpy.ndarray, optional): Input sequence.
            temperature (float, optional): Temperature for token sampling. Higher values result in more randomness.
            seed (int, optional): Random seed for reproducibility.
            deterministic (bool, optional): If True, selects the most probable next word without random sampling.

        Returns:
            Tuple[jax.numpy.ndarray]: A tuple containing the generated sequence.
        """
        if x is not None:
            assert x.shape[0] == 1, "Batch size must be 1, else use generate_batch()"

        decoder_input = x if x is not None else jnp.array([[self.start_token]])
        output_sequence = []

        # Autoregressive decoding loop
        for _ in range(self.max_length - 1):
            decoder_output = self.decoder(self.zero_pad(decoder_input, self.max_length), training=False)[0]
            last_token_logits = decoder_output[:, -1, :]
            scaled_logits = last_token_logits / temperature
            next_token_probabilities = jax.nn.softmax(scaled_logits, axis=-1)

            if deterministic:
                next_token = jnp.argmax(next_token_probabilities, axis=-1)
            else:
                next_token = jax.random.categorical(jax.random.PRNGKey(int(time.time())), next_token_probabilities, axis=-1)

            next_token = next_token[0]
            output_sequence.append(next_token.item())
            decoder_input = jnp.concatenate([decoder_input, jnp.array([[next_token]])], axis=1)

            if next_token.item() == self.end_token:
                break

        return jnp.array(output_sequence)
    

    def generate_batch(self, 
                 x: Optional[jnp.ndarray] = None,
                 temperature: float = 1.0,
                 deterministic: bool = False) -> jnp.ndarray:
        """
        Generate sequences either from scratch or continues from the input sequence in batch.

        Args:
            x (jax.numpy.ndarray, optional): Batch of input sequences.
            temperature (float, optional): Temperature for token sampling. Higher values result in more randomness.
            deterministic (bool, optional): If True, selects the most probable next word without random sampling.

        Returns:
            jax.numpy.ndarray: An array containing the generated sequences for each sample in the batch.
        """

        batch_size = x.shape[0] if x is not None else 1
        decoder_input = x if x is not None else jnp.full((batch_size, 1), self.start_token)
        output_sequences = jnp.zeros((batch_size, self.max_length), dtype=jnp.int32)

        for i in range(self.max_length-1):
            decoder_output = self.decoder(self.zero_pad(decoder_input, self.max_length), training=False)[0]
            last_token_logits = decoder_output[:, -1, :]
            scaled_logits = last_token_logits / temperature
            next_token_probabilities = jax.nn.softmax(scaled_logits, axis=-1)

            if deterministic:
                next_token = jnp.argmax(next_token_probabilities, axis=-1)
            else:
                key = jax.random.PRNGKey(int(time.time()))
                next_token = jax.random.categorical(key, next_token_probabilities, axis=-1)

            output_sequences = output_sequences.at[:, i].set(next_token)
            decoder_input = jnp.concatenate([decoder_input, next_token[:, None]], axis=1)

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

    Methods:
        setup(): Initializes the experts, the gating mechanism, and the final dense layer.

        __call__(X: jnp.ndarray) -> jnp.ndarray:
            Performs a forward pass through the Mixture of Experts layer.

            Args:
                X (jnp.ndarray): Input tensor of shape (batch_size, seq_length, input_dim).

            Returns:
                jnp.ndarray: Output tensor after processing through the MoE layer. The output
                tensor has the same batch and sequence length dimensions as the input tensor,
                but the last dimension is equal to num_outputs.
    """
    num_hiddens: int
    num_outputs: int
    num_experts: int = 8
    top_k: int = 2  # Number of top experts to use

    def setup(self):
        self.experts = [PositionWiseFFN(self.num_hiddens, 
                                        self.num_outputs) for _ in range(self.num_experts)
                                ]
        self.gate = nn.Dense(self.num_experts, 
                            kernel_init=nn.initializers.xavier_uniform()
                            )
        self.dense_final = nn.Dense(self.num_outputs, 
                                    kernel_init=nn.initializers.xavier_uniform()
                                    )

    def __call__(self, X: jnp.ndarray) -> jnp.ndarray:
        gating_weights = nn.softmax(self.gate(X), axis=-1)

        # Get top K experts for each example in the batch
        top_k_indices = jnp.argsort(gating_weights, axis=-1)[..., -self.top_k:]

        # Get expert outputs
        expert_outputs = jnp.stack([expert(X) for expert in self.experts], axis=2)

        # Select only the top K expert outputs
        batch_size, seq_length, _ = X.shape
        batch_indices = jnp.arange(batch_size)[:, None, None]
        seq_indices = jnp.arange(seq_length)[None, :, None]
        top_k_expert_outputs = expert_outputs[batch_indices, seq_indices, top_k_indices]

        # Compute the gating weights for the selected top K experts
        top_k_gating_weights = jnp.take_along_axis(gating_weights, top_k_indices, axis=-1)

        # Compute the mixed expert output
        mixed_expert_output = jnp.sum(top_k_gating_weights[..., None] * top_k_expert_outputs, axis=2)

        return self.dense_final(mixed_expert_output)
    

class MixtralDecoderBlock(nn.Module):
    """
    Transformer Decoder Block with Mixture-of-Experts Feed Forward..

    Args:
        hidden_dim (int): Input dimension.
        num_heads (int): Number of attention heads.
        feedforward_dim (int): Dimension of the feed-forward network.
        dropout (float): Dropout rate.
    """
    hidden_dim: int
    num_heads: int
    feedforward_dim: int
    dropout: float
    num_groups: int
    window_size: int
    shift_size: int

    def setup(self):
        self.attention1 = GroupedRotaryShiftedWindowMultiHeadAttention(hidden_dim=self.hidden_dim, 
                                                          num_heads=self.num_heads,
                                                          num_groups=self.num_groups,
                                                          window_size=self.window_size,
                                                          shift_size=self.shift_size)
        
        self.attention2 = GroupedRotaryShiftedWindowMultiHeadAttention(hidden_dim=self.hidden_dim, 
                                                          num_heads=self.num_heads,
                                                          num_groups=self.num_groups,
                                                          window_size=self.window_size,
                                                          shift_size=self.shift_size)
        
        self.feed_forward = SparseMixtureOfExperts(self.feedforward_dim, self.hidden_dim)
        self.norm1 = nn.RMSNorm(self.dropout)
        self.norm2 = nn.RMSNorm(self.dropout)
        self.norm3 = nn.RMSNorm(self.dropout)
        self.dropout1 = nn.Dropout(self.dropout)
        self.dropout2 = nn.Dropout(self.dropout)
        self.dropout3 = nn.Dropout(self.dropout)

    def __call__(self, 
                x: jnp.ndarray,
                training: bool = False) -> tuple:
        """
        Apply the DecoderBlock to input data.

        Args:
            x (jnp.ndarray): Input tensor.
            context (jnp.ndarray): Context tensor.
            mask (jnp.ndarray, optional): Mask tensor. Defaults to None.
            training (bool): Training mode.

        Returns:
            tuple: Output tensor, attention tensor, and cross-attention tensor.
        """
        x = self.norm1(x)
        attended_x, attention1 = self.attention1(x, x, mask=True)
        x = self.dropout1(x, deterministic=not training)
        x += attended_x

        x = self.norm2(x)
        attended_x, attention2 = self.attention2(x, x, mask=True)
        x = self.dropout2(x, deterministic=not training)
        x += attended_x

        x = self.norm3(x)
        output = self.feed_forward(x)
        x = self.dropout3(x, deterministic=not training)
        x += output

        return x, jnp.array(attention1), jnp.array(attention2)

    
class MixtralDecoder(nn.Module):
    """
    Transformer Decoder.

    Args:
        num_layers (int): Number of decoder layers.
        hidden_dim (int): Input dimension.
        num_heads (int): Number of attention heads.
        feedforward_dim (int): Dimension of the feed-forward network.
        dropout (float): Dropout rate.
    """
    num_layers: int
    hidden_dim: int
    num_heads: int
    num_groups: int
    feedforward_dim: int
    dropout: float
    vocab_size: float
    embed_dim: float
    window_size: int
    shift_size: int

    def setup(self):
        self.embedding = nn.Embed(num_embeddings=self.vocab_size, 
                                  features=self.embed_dim)
        
        self.layers = [MixtralDecoderBlock(self.hidden_dim, 
                                    self.num_heads, 
                                    self.feedforward_dim, 
                                    self.dropout,
                                    self.num_groups,
                                    self.window_size,
                                    self.shift_size) for _ in range(self.num_layers)]
        
        self.outputs = nn.Dense(self.vocab_size)
        

    def __call__(self, 
                 x: jnp.ndarray,
                 training: bool = False,
                 drop_last_layer: bool = False) -> tuple:
        """
        Apply the TransformerDecoder to input data.

        Args:
            x (jnp.ndarray): Input tensor.
            context (jnp.ndarray): Context tensor.
            mask (jnp.ndarray, optional): Mask tensor. Defaults to None.
            training (bool): Training mode.

        Returns:
            tuple: Output tensor, list of attention tensors, and list of cross-attention tensors.
            each attention map has dim (num_layers, batch_size, num_heads, seq_length, seq_length)
        """
        attention_maps = []
        x = self.embedding(x)
        cross_attention_maps = []
        for layer in self.layers:
            x, attention, cross_attention = layer(x, training=training)
            attention_maps.append(attention)
            cross_attention_maps.append(cross_attention)

        if not drop_last_layer:
            x = self.outputs(x)

        return x, jnp.array(attention_maps), jnp.array(cross_attention_maps)
    


class Mixtral(nn.Module):
    """
    Args:
        num_layers (int): Number of layers in the encoder and decoder.
        num_heads (int): Number of attention heads in the multi-head attention layers.
        hidden_dim (int): Dimensionality of input embeddings.
        feedforward_dim (int): Dimensionality of the feedforward layers.
        dropout (float): Dropout probability.
        vocab_size (int): Size of the vocabulary.
        embed_dim (int): Dimensionality of token embeddings.
        max_length (int): Maximum length of generated sequences.
        start_token (int): Token ID for the start of sequence.
        end_token (int): Token ID for the end of sequence.
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
    window_size: int
    shift_size: int

    def setup(self):
        """
        Initialize the Mistral model 
        """
        self.decoder = MixtralDecoder(self.num_layers,
                                self.hidden_dim,
                                self.num_heads,
                                self.num_groups,
                                self.feedforward_dim,
                                self.dropout,
                                self.vocab_size,
                                self.embed_dim,
                                self.window_size,
                                self.shift_size)
        
    def __call__(self, 
                 x: jnp.ndarray,
                 training: bool = False,
                 drop_last_layer: bool = False) -> jnp.ndarray:
        
        """ 
        Sequence-to-sequence models use teacher forcing during training and as such, 
        the decoder input is the ground truth sequence.
        """
        return self.decoder(x=x, 
                            training=training,
                            drop_last_layer=drop_last_layer)[0]
    
    def zero_pad(self, arr, max_length):
        """Zero-pad the given array to the specified maximum length along axis=1."""
        current_length = arr.shape[1]
        num_zeros = max_length - current_length

        if num_zeros > 0:
            zeros = jnp.zeros((arr.shape[0], num_zeros), dtype=arr.dtype)
            padded_array = jnp.concatenate([arr, zeros], axis=1)
        else:
            padded_array = arr

        return padded_array
    

    def generate(self, 
                 x: Optional[jnp.ndarray] = None,
                 temperature: float = 1.0,
                 deterministic: bool = False) -> Tuple[jnp.ndarray]:
        """
        Generate sequences either from scratch or continues from the input sequence.

        Args:
            x (jax.numpy.ndarray, optional): Input sequence.
            temperature (float, optional): Temperature for token sampling. Higher values result in more randomness.
            seed (int, optional): Random seed for reproducibility.
            deterministic (bool, optional): If True, selects the most probable next word without random sampling.

        Returns:
            Tuple[jax.numpy.ndarray]: A tuple containing the generated sequence.
        """
        if x is not None:
            assert x.shape[0] == 1, "Batch size must be 1, else use generate_batch()"

        decoder_input = x if x is not None else jnp.array([[self.start_token]])
        output_sequence = []

        # Autoregressive decoding loop
        for _ in range(self.max_length-1):
            decoder_output = self.decoder(self.zero_pad(decoder_input, self.max_length), training=False)[0]
            last_token_logits = decoder_output[:, -1, :]
            scaled_logits = last_token_logits / temperature
            next_token_probabilities = jax.nn.softmax(scaled_logits, axis=-1)

            if deterministic:
                next_token = jnp.argmax(next_token_probabilities, axis=-1)
            else:
                next_token = jax.random.categorical(jax.random.PRNGKey(int(time.time())), next_token_probabilities, axis=-1)

            next_token = next_token[0]
            output_sequence.append(next_token.item())
            decoder_input = jnp.concatenate([decoder_input, jnp.array([[next_token]])], axis=1)

            if next_token.item() == self.end_token or len(output_sequence) == self.max_length:
                break

        return jnp.array(output_sequence)
    

    def generate_batch(self, 
                 x: Optional[jnp.ndarray] = None,
                 temperature: float = 1.0,
                 deterministic: bool = False) -> jnp.ndarray:
        """
        Generate sequences either from scratch or continues from the input sequence in batch.

        Args:
            x (jax.numpy.ndarray, optional): Batch of input sequences.
            temperature (float, optional): Temperature for token sampling. Higher values result in more randomness.
            deterministic (bool, optional): If True, selects the most probable next word without random sampling.

        Returns:
            jax.numpy.ndarray: An array containing the generated sequences for each sample in the batch.
        """

        batch_size = x.shape[0] if x is not None else 1
        decoder_input = x if x is not None else jnp.full((batch_size, 1), self.start_token)
        output_sequences = jnp.zeros((batch_size, self.max_length), dtype=jnp.int32)

        for i in range(self.max_length-1):
            decoder_output = self.decoder(self.zero_pad(decoder_input, self.max_length), training=False)[0]
            last_token_logits = decoder_output[:, -1, :]
            scaled_logits = last_token_logits / temperature
            next_token_probabilities = jax.nn.softmax(scaled_logits, axis=-1)

            if deterministic:
                next_token = jnp.argmax(next_token_probabilities, axis=-1)
            else:
                key = jax.random.PRNGKey(int(time.time()))
                next_token = jax.random.categorical(key, next_token_probabilities, axis=-1)

            output_sequences = output_sequences.at[:, i].set(next_token)
            decoder_input = jnp.concatenate([decoder_input, next_token[:, None]], axis=1)

            if jnp.all(next_token == self.end_token) or len(output_sequences) == self.max_length:
                break

        return output_sequences
    
    
class MistralDataParallelTrainer:
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
                 learning_rate: float = 1e-5,
                 params_path: Optional[str] = None) -> None:
        self.model = model
        self.params = None
        self.params_path = params_path
        self.num_parameters = None
        self.best_val_loss = float("inf")
        self.weights_filename = weights_filename
        self.num_devices = jax.local_device_count()
        self.train_step = jax.pmap(MistralDataParallelTrainer.train_step, axis_name='devices')
        self.evaluation_step = jax.pmap(MistralDataParallelTrainer.evaluation_step, axis_name='devices')
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
        params = self.model.init(rngs, 
                                 jnp.ones(input_shape, dtype=jnp.int32))['params']

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
                   inputs: jnp.ndarray,
                   targets: jnp.ndarray) -> Tuple[Any, jnp.ndarray]:
        """
        Performs a single training step.

        Args:
            state: The current state of the model, including parameters and optimizer state.
            batch: A dictionary containing 'inputs' and 'targets' as keys, representing the input data.

        Returns:
            A tuple of the updated state and the loss value for this step.
        """
        def loss_fn(params):
            logits = state.apply_fn({'params': params}, 
                                    inputs,
                                    training=True,
                                    rngs={'dropout': jax.random.PRNGKey(int(time.time()))})
            return optax.softmax_cross_entropy_with_integer_labels(logits, targets).mean()
        
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
            for inputs, targets in train_loader:
                batch_size = inputs.shape[0]
                batch_size_per_device = batch_size // self.num_devices
                inputs = inputs.reshape((self.num_devices, batch_size_per_device, -1))
                targets = targets.reshape((self.num_devices, batch_size_per_device, -1))
                self.state, loss = self.train_step(state=self.state, 
                                                   inputs=inputs, 
                                                   targets=targets)
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
                        inputs: jnp.ndarray,
                        targets: jnp.ndarray) -> Tuple[Any, jnp.ndarray]:
        """
        Performs a single training step.

        Args:
            state: The current state of the model, including parameters and optimizer state.
            batch: A dictionary containing 'inputs' and 'targets' as keys, representing the input data.

        Returns:
            A tuple of the updated state and the loss value for this step.
        """
        logits = state.apply_fn({'params': state.params}, inputs,  rngs={'dropout': jax.random.PRNGKey(2)})
        return optax.softmax_cross_entropy_with_integer_labels(logits, targets).mean()

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
        """
        Saves the unreplicated model parameters to a file.
        """
        self.params = flax.jax_utils.unreplicate(self.state.params)
        with open(self.weights_filename, 'wb') as f:
            f.write(flax.serialization.to_bytes(self.params))

    def load_params(self, filename: str):
        """
        Loads the model parameters from a file
        """
        with open(filename, 'rb') as f:
            self.params = flax.serialization.from_bytes(self.params, f.read())
        return self.params