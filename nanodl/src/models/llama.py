'''
LlaMA is built upon the transformer architecture, incorporating enhancements inspired by recent advancements in the field of large language models. 
These improvements are drawn from various sources, such as GPT-3, PaLM, and GPT-Neo. Notable modifications include the adoption of pre-normalization for enhanced training stability, 
employing the RMSNorm normalization function. Additionally, the ReLU non-linearity is replaced with the SwiGLU activation function, which is a variant of the GLU activation function.
Absolute positional embeddings are replaced with rotary positional embeddings (RoPE), implemented at each layer of the network. For specific hyper-parameter details, refer to Table 2 in the document.

Example usage:
```
from llama import *

# Dummy data parameters
batch_size = 8
max_length = 51
vocab_size = 1000 
embed_dim = 256 

# Generate data
data = jnp.arange(batch_size * max_length, dtype=jnp.int32).reshape((batch_size, max_length))
dummy_inputs = data[:, :-1]
dummy_targets = data[:, 1:]

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
model = LlaMA(**hyperparams)
rngs = {'params': jax.random.key(0), 'dropout': jax.random.key(1)}
params = model.init(rngs, dummy_inputs)['params']
outputs = model.apply({'params': params}, dummy_inputs, rngs={'dropout': jax.random.PRNGKey(2)})
print(outputs.shape)

# Training on your data
dataloader = [(dummy_inputs, dummy_targets)] * 10
trainer = LlaMADataParallelTrainer(model, dummy_inputs.shape, 'params.pkl')
trainer.train(dataloader, 10, dataloader)
print(trainer.evaluate(dataloader))

# Generate: should always have dims (batch_size, seq_len)
input_sequence = jnp.array([[123, 456], [145, 656]])

params = trainer.load_params('params.pkl')
outputs = model.apply({'params': params},
                      input_sequence, 
                      rngs={'dropout': jax.random.PRNGKey(2)}, 
                      method=model.generate)

print(outputs)
```
'''

import jax
import time
import optax
import pickle
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
    

class RotaryMultiHeadAttention(nn.Module):
    """
    Attention which uses RoPE (Rotary Positional Encoding)
    """
    hidden_dim : int  # Output dimension
    num_heads : int  # Number of parallel heads

    def setup(self):
        # Because the Query is determined from a context, project separately
        self.query_projection = nn.Dense(self.hidden_dim,
                                 kernel_init=nn.initializers.xavier_uniform(),
                                 bias_init=nn.initializers.zeros 
                                )
        self.key_projection = nn.Dense(self.hidden_dim,
                                 kernel_init=nn.initializers.xavier_uniform(),
                                 bias_init=nn.initializers.zeros 
                                )
        self.value_projection = nn.Dense(self.hidden_dim,
                                 kernel_init=nn.initializers.xavier_uniform(),
                                 bias_init=nn.initializers.zeros 
                                )
        self.rope = RotaryPositionalEncoding(self.hidden_dim)
        self.output = nn.Dense(self.hidden_dim,
                               kernel_init=nn.initializers.xavier_uniform(),
                               bias_init=nn.initializers.zeros)


    def __call__(self, 
                 inputs: jnp.ndarray, 
                 context: jnp.ndarray, 
                 mask: jnp.ndarray = None) -> tuple:

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
        query, key = self.rope(query, key) # Encode query and key with RoPE
        context_vectors, attention = self.attention_function(query,key, value, mask=mask)
        outputs = self.output(context_vectors)
        return outputs, attention
    
    def attention_function(self, query, key, value, mask=None):
        input_length = query.shape[1]
        context_length = key.shape[1]
        head_dim = query.shape[-1] // self.num_heads
        dim_key = key.shape[-1]

        # Split queries, keys, and values into heads
        query_heads = jnp.reshape(query, (query.shape[0], self.num_heads, input_length, head_dim))
        key_heads = jnp.reshape(key, (key.shape[0], self.num_heads, context_length, head_dim))
        value_heads = jnp.reshape(value, (value.shape[0], self.num_heads, context_length, head_dim))

        attention_scores = jnp.matmul(query_heads, key_heads.transpose(0, 1, 3, 2)) / jnp.sqrt(dim_key)
        if mask is not None:
            attention_scores = attention_scores * mask

        attention_weights = jax.nn.softmax(attention_scores, axis=-1)
        attended_values = jnp.matmul(attention_weights, value_heads)
        attended_values = jnp.reshape(attended_values, (query.shape[0], input_length, query.shape[-1]))
        return attended_values, attention_weights
    

class SwiGLU(nn.Module):
    """
    Sigmoid-Weighted Linear Unit (SwiGLU) activation function.
    SwiGLU(x) = x * sigmoid(Wx + b)

    Args:
        output_dim (int): Output dimension of the SwiGLU layer.
    """
    @nn.compact
    def __call__(self, x):
        return x * nn.sigmoid(nn.Dense(x.shape[-1])(x))
    

class PositionWiseFFN(nn.Module):
    """
    Position-wise Feed-Forward Network.

    Args:
        num_hiddens (int): Number of hidden units in the feed-forward layers.
        num_outputs (int): Number of output units in the feed-forward layers.
    """
    num_hiddens: int
    num_outputs: int

    def setup(self):
        self.dense1 = nn.Dense(self.num_hiddens, kernel_init=nn.initializers.xavier_uniform())
        self.activation = SwiGLU()
        self.dense2 = nn.Dense(self.num_outputs, kernel_init=nn.initializers.xavier_uniform())

    def __call__(self, X: jnp.ndarray) -> jnp.ndarray:
        """
        Apply the PositionWiseFFN to input data.

        Args:
            X (jnp.ndarray): Input tensor.

        Returns:
            jnp.ndarray: Output tensor after applying the feed-forward network.
        """
        return self.dense2(self.activation(self.dense1(X)))
    

class LlaMAEncoderBlock(nn.Module):
    """
    Transformer Encoder Block.

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

    def setup(self):
        self.attention = RotaryMultiHeadAttention(hidden_dim=self.hidden_dim, 
                                                    num_heads=self.num_heads)
        self.linear = PositionWiseFFN(self.feedforward_dim, self.hidden_dim)
        self.norm1 = nn.RMSNorm(self.dropout)
        self.norm2 = nn.RMSNorm(self.dropout)
        self.dropout1 = nn.Dropout(self.dropout)
        self.dropout2 = nn.Dropout(self.dropout)

    def __call__(self, 
                 x: jnp.ndarray, 
                 mask: jnp.ndarray = None, 
                 training: bool = False) -> tuple:
        """
        Apply the EncoderBlock to input data.

        Args:
            x (jnp.ndarray): Input tensor.
            mask (jnp.ndarray, optional): Mask tensor. Defaults to None.
            training (bool): Training mode.

        Returns:
            tuple: Output tensor and attention tensor.
        """
        x = self.norm1(x)
        attended_x, attention = self.attention(x, x, mask=mask)
        x = self.dropout1(x, deterministic=not training)
        x += attended_x
        x = self.norm2(x)
        fc_out=self.linear(x) 
        x = self.dropout1(fc_out, deterministic=not training) + x
        return x, attention
    

class LlaMAEncoder(nn.Module):
    """
    Transformer Encoder.

    Args:
        num_layers (int): Number of encoder layers.
        hidden_dim (int): Input dimension.
        num_heads (int): Number of attention heads.
        feedforward_dim (int): Dimension of the feed-forward network.
        dropout (float): Dropout rate.
    """
    num_layers: int
    hidden_dim: int
    num_heads: int
    feedforward_dim: int
    dropout: float
    vocab_size: float
    embed_dim: float


    def setup(self):
        self.embedding = nn.Embed(num_embeddings=self.vocab_size, 
                                  features=self.embed_dim)
        
        self.layers = [LlaMAEncoderBlock(self.hidden_dim, 
                                    self.num_heads, 
                                    self.feedforward_dim, 
                                    self.dropout)
                       for _ in range(self.num_layers)]

    def __call__(self, 
                 x: jnp.ndarray, 
                 mask: jnp.ndarray = None, 
                 training: bool = False) -> tuple:
        """
        Apply the TransformerEncoder to input data.

        Args:
            x (jnp.ndarray): Input tensor.
            mask (jnp.ndarray, optional): Mask tensor. Defaults to None.
            training (bool): Training mode.

        Returns:
            tuple: Output tensor and list of attention tensors.
            each attention map has dim (num_layers, batch_size, num_heads, seq_length, seq_length)
        """
        attention_maps = []
        x = self.embedding(x)
        for layer in self.layers:
            x, attention = layer(x, mask=mask, training=training)
            attention_maps.append(attention)
        return x, jnp.array(attention_maps)
    
    
class LlaMADecoderBlock(nn.Module):
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

    def setup(self):
        self.attention1 = RotaryMultiHeadAttention(hidden_dim=self.hidden_dim, num_heads=self.num_heads)
        self.attention2 = RotaryMultiHeadAttention(hidden_dim=self.hidden_dim, num_heads=self.num_heads)
        self.feed_forward = PositionWiseFFN(self.feedforward_dim, self.hidden_dim)
        self.norm1 = nn.RMSNorm(self.dropout)
        self.norm2 = nn.RMSNorm(self.dropout)
        self.norm3 = nn.RMSNorm(self.dropout)
        self.dropout1 = nn.Dropout(self.dropout)
        self.dropout2 = nn.Dropout(self.dropout)
        self.dropout3 = nn.Dropout(self.dropout)

    def causal_mask(self, 
                batch_size: int, 
                destination_dim: int, 
                source_dim: int) -> jnp.ndarray:
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
        idx_source = jnp.arange(destination_dim)[:, None]
        idx_destination = jnp.arange(source_dim)
        mask = idx_source >= idx_destination - source_dim + destination_dim
        mask = mask.astype(jnp.int32) 

        # Expand dimensions to match the required output shape
        mask = mask[None, None, :, :]
        return jnp.broadcast_to(mask, (batch_size, self.num_heads, destination_dim, source_dim))

    def __call__(self, 
                x: jnp.ndarray, 
                context: jnp.ndarray,
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
        mask = self.causal_mask(x.shape[0], x.shape[1], context.shape[1])

        x = self.norm1(x)
        attended_x, attention1 = self.attention1(x, x, mask=mask)
        x = self.dropout1(x, deterministic=not training)
        x += attended_x

        x = self.norm2(x)
        attended_x, attention2 = self.attention2(x, context, mask=mask)
        x = self.dropout2(x, deterministic=not training)
        x += attended_x

        x = self.norm3(x)
        output = self.feed_forward(x)
        x = self.dropout3(x, deterministic=not training)
        x += output

        return x, jnp.array(attention1), jnp.array(attention2)

    
class LlaMADecoder(nn.Module):
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
    feedforward_dim: int
    dropout: float
    vocab_size: float
    embed_dim: float


    def setup(self):
        self.embedding = nn.Embed(num_embeddings=self.vocab_size, 
                                  features=self.embed_dim)
        
        self.layers = [LlaMADecoderBlock(self.hidden_dim, 
                                    self.num_heads, 
                                    self.feedforward_dim, 
                                    self.dropout) for _ in range(self.num_layers)]
        
        self.outputs = nn.Dense(self.vocab_size)
        

    def __call__(self, 
                 x: jnp.ndarray, 
                 context: jnp.ndarray, 
                 training: bool = False) -> tuple:
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
            x, attention, cross_attention = layer(x, context, training=training)
            attention_maps.append(attention)
            cross_attention_maps.append(cross_attention)
        return self.outputs(x), jnp.array(attention_maps), jnp.array(cross_attention_maps)
    


class LlaMA(nn.Module):
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
    hidden_dim: int
    feedforward_dim: int
    dropout: float
    vocab_size: float
    embed_dim: float
    max_length: int
    start_token: int
    end_token: int

    def setup(self):
        """
        Initialize the LlaMA model by setting up the encoder and decoder.
        """
        self.encoder = LlaMAEncoder(self.num_layers,
                                self.hidden_dim,
                                self.num_heads,
                                self.feedforward_dim,
                                self.dropout,
                                self.vocab_size,
                                self.embed_dim)
        
        self.decoder = LlaMADecoder(self.num_layers,
                                self.hidden_dim,
                                self.num_heads,
                                self.feedforward_dim,
                                self.dropout,
                                self.vocab_size,
                                self.embed_dim)
        
    def __call__(self, 
                 x: jnp.ndarray,
                 y: jnp.ndarray,
                 training: bool = False) -> jnp.ndarray:
        
        """ 
        Sequence-to-sequence models use teacher forcing during training and as such, 
        the decoder input is the ground truth sequence.
        """
        z = self.encoder(x=x, training=training)[0]
        return self.decoder(x=y, context=z, training=training)[0]
    

    def generate(self, 
                 x: jnp.ndarray,
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
        # Encode the input sequence
        encoded_sequence = self.encoder(x=x, training=False)[0]

        decoder_input = x if x is not None else jnp.array([[self.start_token]])
        output_sequence = []

        # Autoregressive decoding loop
        for _ in range(self.max_length):
            decoder_output = self.decoder(x=decoder_input,
                                          context=encoded_sequence, 
                                          training=False)[0]
            last_token_logits = decoder_output[:, -1, :]
            scaled_logits = last_token_logits / temperature
            next_token_probabilities = jax.nn.softmax(scaled_logits, axis=-1)

            if deterministic:
                next_token = jnp.argmax(next_token_probabilities, axis=-1)
            else:
                next_token = jax.random.categorical(jax.random.PRNGKey(int(time.time())), next_token_probabilities, axis=-1)

            next_token = next_token[0]
            output_sequence.append(next_token.item())
            print(decoder_input.shape, jnp.array([[next_token]]).shape)
            decoder_input = jnp.concatenate([decoder_input, jnp.array([[next_token]])], axis=1)

            if next_token.item() == self.end_token:
                break

        return tuple(output_sequence)
    

    def generate_batch(self, 
                 x: jnp.ndarray,
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
        # Encode the input sequence
        encoded_sequence = self.encoder(x=x, training=False)[0]

        batch_size = x.shape[0] if x is not None else 1
        decoder_input = x if x is not None else jnp.full((batch_size, 1), self.start_token)
        output_sequences = jnp.zeros((batch_size, self.max_length), dtype=jnp.int32)

        for i in range(self.max_length):
            decoder_output = self.decoder(x=decoder_input,
                                          context=encoded_sequence, 
                                          training=False)[0]
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



class LlaMADataParallelTrainer:
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
                 target_shape: Tuple[int, ...],
                 weights_filename: str,
                 learning_rate: float = 1e-5,
                 params_path: Optional[str] = None) -> None:
        self.model = model
        self.params_path = params_path
        self.num_parameters = None
        self.best_val_loss = float("inf")
        self.weights_filename = weights_filename
        self.num_devices = jax.local_device_count()
        self.train_step = jax.pmap(LlaMADataParallelTrainer.train_step, axis_name='devices')
        self.evaluation_step = jax.pmap(LlaMADataParallelTrainer.evaluation_step, axis_name='devices')
        self.state = self.create_train_state(learning_rate, input_shape, target_shape)
        print(f'Number of accelerators: {self.num_devices}')
    

    def create_train_state(self, 
                           learning_rate: float, 
                           input_shape: Tuple[int, ...],
                           target_shape: Tuple[int, ...]) -> Any:
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
                                 jnp.ones(input_shape, dtype=jnp.int32), 
                                 jnp.ones(target_shape, dtype=jnp.int32))['params']

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
                                    targets,
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
        logits = state.apply_fn({'params': state.params}, inputs, targets,  rngs={'dropout': jax.random.PRNGKey(2)})
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
        Saves the model parameters to a file.
        """
        with open(self.weights_filename, 'wb') as f:
            pickle.dump(self.state.params, f)

    @staticmethod
    def load_params(filename: str) -> Any:
        """
        Loads the model parameters from a file.

        Args:
            filename: The filename of the file containing the parameters.

        Returns:
            The loaded parameters.
        """
        with open(filename, 'rb') as f:
            params = pickle.load(f)
        return params