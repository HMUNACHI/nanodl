'''
Transformers are a groundbreaking class of deep learning models originally introduced in the paper "Attention Is All You Need" by Vaswani et al. 
Their motivation stems from addressing limitations in previous sequence-to-sequence models and enabling more efficient and parallelizable training. 
The key innovation of transformers is the self-attention mechanism, which allows the model to weigh the importance of different parts of the input sequence during processing. 
This architecture has had a profound impact on natural language processing and has been adapted for a wide range of tasks, including machine translation, text generation, image captioning, and more. 
Transformers have become the foundation for various state-of-the-art models, including BERT, GPT, and Transformer, which have achieved remarkable results across multiple domains, showcasing the power of attention-based architectures in deep learning.

Example usage:
```
import jax
import jax.numpy as jnp
from nanodl import ArrayDataset, DataLoader
from nanodl import Transformer, TransformerDataParallelTrainer

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
}

# Initialize model
model = Transformer(**hyperparams)
rngs = jax.random.PRNGKey(0)
rngs, dropout_rng = jax.random.split(rngs)
params = model.init({'params': rngs, 'dropout': dropout_rng}, 
                    dummy_inputs,
                    dummy_targets)['params']

# Call as you would a Jax/Flax model
outputs = model.apply({'params': params}, 
                      dummy_inputs, 
                      dummy_targets,
                      rngs={'dropout': dropout_rng})
print(outputs.shape)

# Training on data
trainer = TransformerDataParallelTrainer(model, 
                                dummy_inputs.shape, 
                                dummy_targets.shape,
                                'params.pkl')

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
from typing import List, Tuple, Any, Optional, Dict, Iterable


class PositionalEncoding(nn.Module):
    """
    Positional Encoding.
    Args:
        num_embeddings (int): Number of embeddings.
        features (int): Number of features in the embeddings.
    """
    num_embeddings: int
    features: int

    def setup(self):
        positional_encoding = jnp.zeros((self.features, self.num_embeddings))
        position = jnp.arange(0, self.features, dtype=jnp.float32)[:, None]
        div_term = jnp.exp(jnp.arange(0, self.num_embeddings, 2) * (-jnp.log(10000.0) / self.num_embeddings))
        positional_encoding = positional_encoding.at[:, 0::2].set(jnp.sin(position * div_term))
        positional_encoding = positional_encoding.at[:, 1::2].set(jnp.cos(position * div_term))
        self.positional_encoding = positional_encoding.T

    def __call__(self, x):
        x = x + self.positional_encoding[:x.shape[1]]
        return x


class TokenAndPositionEmbedding(nn.Module):
    """
    Token and Position Embedding.
    Args:
        max_len (int): Maximum sequence length.
        vocab_size (int): Vocabulary size.
        embed_dim (int): Embedding dimension.
    """
    max_len : int
    vocab_size : int
    embed_dim : int
    learned_position : bool
    
    def setup(self):
        self.token_embeddings = nn.Embed(num_embeddings=self.vocab_size, features=self.embed_dim)

        if self.learned_position:
            self.position_embeddings = nn.Embed(num_embeddings=self.max_len, features=self.embed_dim)
        else:
            self.position_embeddings = PositionalEncoding(num_embeddings=self.max_len, features=self.embed_dim)

    def __call__(self, x):
        x = self.token_embeddings(x)
        if self.learned_position:
            return x + self.position_embeddings(jnp.arange(x.shape[1]))
        else:
            return x + self.position_embeddings(x)
    

class MultiHeadAttention(nn.Module):
    """
    https://arxiv.org/abs/1706.03762 (Vaswani et. al. 2017)
    This involves transforming the input by weighting features by importance.
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
        self.activation = nn.gelu
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
    

class AddNorm(nn.Module):
    """
    Residual connection followed by layer normalization.

    Args:
        dropout (float): Dropout rate for the residual connection.
    """
    dropout: int

    @nn.compact
    def __call__(self, 
                 X: jnp.ndarray, 
                 Y: jnp.ndarray, 
                 training=False) -> jnp.ndarray:
        """
        Apply AddNorm to input tensors.

        Args:
            X (jnp.ndarray): Input tensor X.
            Y (jnp.ndarray): Input tensor Y.
            training (bool): Training mode.

        Returns:
            jnp.ndarray: Output tensor after applying AddNorm.
        """
        return nn.LayerNorm()(
            nn.Dropout(self.dropout)(Y, deterministic=not training) + X)
    

class TransformerEncoderBlock(nn.Module):
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
        self.attention = MultiHeadAttention(hidden_dim=self.hidden_dim,
                                            num_heads=self.num_heads)
        self.linear = PositionWiseFFN(self.feedforward_dim, self.hidden_dim)
        self.add_norm1 = AddNorm(self.dropout)
        self.add_norm2 = AddNorm(self.dropout)

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
        attended_x, attention = self.attention(x, x, mask=mask)
        x = self.add_norm1(x, attended_x, training)
        linear_output = self.linear(x)
        x = self.add_norm2(x, linear_output, training)
        return x, attention
    
    
class TransformerEncoder(nn.Module):
    """
    Transformer Encoder.

    Args:
        num_layers (int): Number of encoder layers.
        hidden_dim(int): Input dimension.
        num_heads (int): Number of attention heads.
        feedforward_dim (int): Dimension of the feed-forward network.
        dropout (float): Dropout rate.
    """
    num_layers: int
    hidden_dim: int
    num_heads: int
    feedforward_dim: int
    dropout: float
    max_len : int
    vocab_size : int
    embed_dim : int
    learned_position : bool = True


    def setup(self):
        self.embedding = TokenAndPositionEmbedding(self.max_len,
                                                   self.vocab_size,
                                                   self.embed_dim,
                                                   self.learned_position)
        
        self.layers = [TransformerEncoderBlock(self.hidden_dim, 
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
    

class TransformerDecoderBlock(nn.Module):
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
        self.attention1 = MultiHeadAttention(hidden_dim=self.hidden_dim, num_heads=self.num_heads)
        self.attention2 = MultiHeadAttention(hidden_dim=self.hidden_dim, num_heads=self.num_heads)
        self.feed_forward = PositionWiseFFN(self.feedforward_dim, self.hidden_dim)
        self.add_norm1 = AddNorm(self.dropout)
        self.add_norm2 = AddNorm(self.dropout)
        self.add_norm3 = AddNorm(self.dropout)

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

        attended_x, attention1 = self.attention1(x, x)
        x = self.add_norm1(x, attended_x, training)

        attended_x, attention2 = self.attention2(x, context, mask=mask)
        x = self.add_norm2(x, attended_x, training)

        linear_output = self.feed_forward(x)
        x = self.add_norm3(x, linear_output, training)
        
        return x, jnp.array(attention1), jnp.array(attention2)
    

class TransformerDecoder(nn.Module):
    """
    Transformer Decoder.

    Args:
        num_layers (int): Number of encoder layers.
        hidden_dim(int): Input dimension.
        num_heads (int): Number of attention heads.
        feedforward_dim (int): Dimension of the feed-forward network.
        dropout (float): Dropout rate.
    """
    num_layers: int
    hidden_dim: int
    num_heads: int
    feedforward_dim: int
    dropout: float
    max_len : int
    vocab_size : int
    embed_dim : int
    learned_position : bool = True


    def setup(self):
        self.embedding = TokenAndPositionEmbedding(self.max_len,
                                                   self.vocab_size,
                                                   self.embed_dim,
                                                   self.learned_position)
        
        self.layers = [TransformerDecoderBlock(self.hidden_dim, 
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
    
    
class Transformer(nn.Module):
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
        Initialize the Transformer model by setting up the encoder and decoder.
        """
        self.encoder = TransformerEncoder(
            hidden_dim=self.hidden_dim,
            num_heads=self.num_heads,
            num_layers=self.num_layers,
            feedforward_dim=self.feedforward_dim,
            dropout=self.dropout,
            max_len=self.max_length,
            vocab_size=self.vocab_size,
            embed_dim=self.embed_dim,
        )
        
        self.decoder = TransformerDecoder(
            hidden_dim=self.hidden_dim,
            num_heads=self.num_heads,
            num_layers=self.num_layers,
            feedforward_dim=self.feedforward_dim,
            dropout=self.dropout,
            max_len=self.max_length,
            vocab_size=self.vocab_size,
            embed_dim=self.embed_dim,
        )
        
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
        encoded_sequence = self.encoder(x=x, training=False)[0]
        decoder_input = jnp.array([[self.start_token]])
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
            decoder_input = jnp.concatenate([decoder_input, jnp.array([[next_token]])], axis=1)

            if next_token.item() == self.end_token:
                break

        return jnp.array(output_sequence)
    

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
        encoded_sequence = self.encoder(x=x, training=False)[0]
        batch_size = x.shape[0] if x is not None else 1
        decoder_input = jnp.full((batch_size, 1), self.start_token)
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



class TransformerDataParallelTrainer:
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
        self.params = None
        self.params_path = params_path
        self.num_parameters = None
        self.best_val_loss = float("inf")
        self.weights_filename = weights_filename
        self.num_devices = jax.local_device_count()
        self.train_step = jax.pmap(TransformerDataParallelTrainer.train_step, axis_name='devices')
        self.evaluation_step = jax.pmap(TransformerDataParallelTrainer.evaluation_step, axis_name='devices')
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