'''
T5, which stands for Text-to-Text Transfer Transformer, is an influential deep learning architecture introduced by Google Research. 
Its motivation stems from the idea of unifying various natural language processing tasks into a single framework to achieve greater model simplicity and efficiency. 
T5 reimagines tasks as text-to-text problems, where both inputs and outputs are represented as text. 
This consistent formulation allows T5 to perform an astonishingly wide range of tasks, 
from translation and summarization to question-answering and document classification, by adjusting the input and output formats accordingly.
The architecture is roughly equivalent to the original Transformer proposed by
Vaswani et al. (2017) with the exception of removing the Layer Norm bias, placing the layer
normalization outside the residual path, and using a different position embedding scheme.

Note: The layer normalisation is left as is, the modification 
'''

import jax
import jax.numpy as jnp
import flax.linen as nn
from jax.nn import softmax


class T5(nn.Module):
    """
    Transformer-based Text-to-Text Transfer Transformer (T5) model for sequence generation.

    Args:
        num_layers (int): Number of layers in the encoder and decoder.
        input_dim (int): Dimensionality of input embeddings.
        num_heads (int): Number of attention heads in the multi-head attention layers.
        feedforward_dim (int): Dimensionality of the feedforward layers.
        dropout (float): Dropout probability.
        vocab_size (int): Size of the vocabulary.
        embed_dim (int): Dimensionality of token embeddings.
        max_length (int): Maximum length of generated sequences.
        start_token (int): Token ID for the start of sequence.
        end_token (int): Token ID for the end of sequence.
    """

    def setup(self):
        """
        Initialize the T5 model by setting up the encoder and decoder.
        """
        self.encoder = Encoder(self.num_layers,
                                self.input_dim,
                                self.num_heads,
                                self.feedforward_dim,
                                self.dropout,
                                self.vocab_size,
                                self.embed_dim)
        
        self.decoder = Decoder(self.num_layers,
                                self.input_dim,
                                self.num_heads,
                                self.feedforward_dim,
                                self.dropout,
                                self.vocab_size,
                                self.embed_dim)

    def __call__(self, 
                 x: jnp.ndarray, 
                 temperature: float = 1.0,
                 training: bool = True) -> tuple:
        """
        Generate sequences using the T5 model.

        Args:
            x (jax.numpy.ndarray): Input sequence.
            temperature (float, optional): Temperature for token sampling. Higher values result in more randomness.
            training (bool, optional): Whether the model is in training mode.

        Returns:
            tuple: A tuple containing the generated sequence.
        """
        
        # Encode the input sequence
        encoded_sequence = self.encoder(x=x, training=training)

        # Initialize the decoding input with a special token
        decoder_input = jnp.array([[self.start_token]])

        # Initialize the output sequence
        output_sequence = []

        # Autoregressive decoding loop
        for _ in range(self.max_length):
            # Generate the next token
            decoder_output = self.decoder(x=decoder_input, 
                                          context=encoded_sequence, 
                                          training=training) 
            
            # Apply temperature scaling to the logits
            scaled_logits = decoder_output / temperature

            next_token_probabilities = jax.nn.softmax(scaled_logits, axis=-1)
            
            # Sample the next token from the distribution
            next_token = jax.random.categorical(jax.random.PRNGKey(0), next_token_probabilities, 1)[0]

            # Append the generated token to the output sequence
            output_sequence.append(next_token.item())

            # Use the generated token as the input for the next step
            decoder_input = jnp.expand_dims(next_token, axis=1)

            # Check if the end token is generated
            if next_token.item() == self.end_token:
                break

        return output_sequence, next_token_probabilities


class Encoder(nn.Module):
    """
    Transformer Encoder.

    Args:
        num_layers (int): Number of encoder layers.
        input_dim (int): Input dimension.
        num_heads (int): Number of attention heads.
        feedforward_dim (int): Dimension of the feed-forward network.
        dropout (float): Dropout rate.
    """
    num_layers: int
    input_dim: int
    num_heads: int
    feedforward_dim: int
    dropout: float
    vocab_size: float
    embed_dim: float


    def setup(self):
        self.embedding = nn.Embed(num_embeddings=self.vocab_size, 
                                  features=self.embed_dim)
        
        self.layers = [EncoderBlock(self.input_dim, 
                                    self.num_heads, 
                                    self.feedforward_dim, 
                                    self.dropout)
                       for _ in range(self.num_layers)]

    def __call__(self, 
                 x: jnp.ndarray, 
                 mask: jnp.ndarray = None, 
                 training: bool = True) -> tuple:
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
    

class EncoderBlock(nn.Module):
    """
    Transformer Encoder Block.

    Args:
        input_dim (int): Input dimension.
        num_heads (int): Number of attention heads.
        feedforward_dim (int): Dimension of the feed-forward network.
        dropout (float): Dropout rate.
    """
    input_dim: int
    num_heads: int
    feedforward_dim: int
    dropout: float

    def setup(self):
        self.attention = RelativeMultiHeadAttention(hidden_dim=self.input_dim, 
                                                    num_heads=self.num_heads)
        self.linear = PositionWiseFFN(self.feedforward_dim, self.input_dim)
        self.add_norm1 = AddNorm(self.dropout)
        self.add_norm2 = AddNorm(self.dropout)

    def __call__(self, 
                 x: jnp.ndarray, 
                 mask: jnp.ndarray = None, 
                 training: bool = True) -> tuple:
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


class Decoder(nn.Module):
    """
    Transformer Decoder.

    Args:
        num_layers (int): Number of decoder layers.
        input_dim (int): Input dimension.
        num_heads (int): Number of attention heads.
        feedforward_dim (int): Dimension of the feed-forward network.
        dropout (float): Dropout rate.
    """
    num_layers: int
    input_dim: int
    num_heads: int
    feedforward_dim: int
    dropout: float
    vocab_size: float
    embed_dim: float


    def setup(self):
        self.embedding = nn.Embed(num_embeddings=self.vocab_size, 
                                  features=self.embed_dim)
        
        self.layers = [DecoderBlock(self.input_dim, 
                                    self.num_heads, 
                                    self.feedforward_dim, 
                                    self.dropout) for _ in range(self.num_layers)]
        
        self.outputs = nn.Dense(self.vocab_size)
        

    def __call__(self, 
                 x: jnp.ndarray, 
                 context: jnp.ndarray, 
                 mask: jnp.ndarray = None, 
                 training: bool = True) -> tuple:
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
    

class DecoderBlock(nn.Module):
    """
    Transformer Decoder Block.

    Args:
        input_dim (int): Input dimension.
        num_heads (int): Number of attention heads.
        feedforward_dim (int): Dimension of the feed-forward network.
        dropout (float): Dropout rate.
    """
    input_dim: int
    num_heads: int
    feedforward_dim: int
    dropout: float

    def setup(self):
        self.attention1 = RelativeMultiHeadAttention(hidden_dim=self.input_dim, num_heads=self.num_heads)
        self.attention2 = RelativeMultiHeadAttention(hidden_dim=self.input_dim, num_heads=self.num_heads)
        self.feed_forward = PositionWiseFFN(self.feedforward_dim, self.input_dim)
        self.add_norm1 = AddNorm(self.dropout)
        self.add_norm2 = AddNorm(self.dropout)
        self.add_norm3 = AddNorm(self.dropout)

    def causal_mask(self, 
                    batch_size: int, 
                    destination_dim: int, 
                    source_dim: int) -> jnp.ndarray:
        """
        Generate a causal mask for self-attention.

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
        mask = mask.reshape((1, destination_dim, source_dim))
        concatenator = jnp.concatenate([jnp.array([batch_size]), 
                                        jnp.array([self.num_heads]), 
                                        jnp.array([1, 1], dtype=jnp.int32)], 0)

        return jnp.tile(mask, concatenator)

    def __call__(self, 
                x: jnp.ndarray, 
                context: jnp.ndarray, 
                mask: jnp.ndarray = None, 
                training: bool = True) -> tuple:
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

        attended_x, attention1 = self.attention1(x, x, mask=mask)
        x = self.add_norm1(x, attended_x, training)

        attended_x, attention2 = self.attention2(x, context, mask=mask)
        x = self.add_norm2(x, attended_x, training)

        linear_output = self.feed_forward(x)
        x = self.add_norm3(x, linear_output, training)
        
        return x, jnp.array(attention1), jnp.array(attention2)
    


class RelativeMultiHeadAttention(nn.Module):
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
                 mask: jnp.ndarray = None, 
                 clip: int = 3) -> tuple:

        """
        Args:
            inputs: inputs ((batch_size, seq_len, dims))
            context: optional - context ((batch_size, seq_len, dims))
            clip: the k value at which to clip the relative position by
            Mask: optional - masks where reqions to ignore are flipped to os
                  regions to attend to are 1s (batch_size, seq_len, dims)

        Return: outputs (batch_size, seq_len, seq_len)
                attention matrixes (batch_size, heads, seq_len, seq_len)
        """
        query = self.query_projection(inputs)
        key = self.key_projection(context)
        value = self.value_projection(context)

        query_relative_positions = jnp.expand_dims(jnp.arange(query.shape[2]), axis=0) 
        query_relative_positions -= jnp.expand_dims(jnp.arange(query.shape[1]), axis=1)
        query_relative_positions = jnp.where(query_relative_positions < clip, query_relative_positions, clip)
        query_relative_positions = jnp.where(query_relative_positions > -clip, query_relative_positions, -clip)
        query += query_relative_positions

        value_relative_positions = jnp.expand_dims(jnp.arange(value.shape[2]), axis=0) 
        value_relative_positions -= jnp.expand_dims(jnp.arange(value.shape[1]), axis=1)
        value_relative_positions = jnp.where(value_relative_positions < clip, value_relative_positions, clip)
        value_relative_positions = jnp.where(value_relative_positions > -clip, value_relative_positions, -clip)
        value += value_relative_positions
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
        self.activation = GEGLU()
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
    

class GEGLU(nn.Module):
    """
    Gated GLU (Gated Linear Unit).
    GEGLU(x) = x * 0.5 * gate * (1 + tanh(gate * 0.7978845608 * (1 + 0.044715 * (gate**2))))

    Args:
        output_dim (int): Output dimension of the GLU layer.
    """
    output_dim: int

    def setup(self):
        self.dense = nn.Dense(self.output_dim * 2,
                              kernel_init=nn.initializers.xavier_uniform())

    def __call__(self, inputs):
        x = self.dense(inputs)
        x, gate = x[..., : self.output_dim], x[..., self.output_dim :]
        tanh_res = jnp.tanh(gate * 0.7978845608 * (1 + 0.044715 * (gate**2)))
        return x * 0.5 * gate * (1 + tanh_res)