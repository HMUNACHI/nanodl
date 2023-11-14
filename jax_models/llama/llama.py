'''
LlaMA is built upon the transformer architecture, incorporating enhancements inspired by recent advancements in the field of large language models. 
These improvements are drawn from various sources, such as GPT-3, PaLM, and GPT-Neo. Notable modifications include the adoption of pre-normalization for enhanced training stability, 
employing the RMSNorm normalization function. Additionally, the ReLU non-linearity is replaced with the SwiGLU activation function for improved performance, with a dimension change from 4d to 2/3/4d. 
Absolute positional embeddings are replaced with rotary positional embeddings (RoPE), implemented at each layer of the network. For specific hyper-parameter details, refer to Table 2 in the document.

Note: This implementation uses GLU which is closely related to SwiGLU
'''

import jax
import jax.numpy as jnp
import flax.linen as nn


class LlaMA(nn.Module):
    """
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
        self.attention = RotaryMultiHeadAttention(hidden_dim=self.input_dim, 
                                                    num_heads=self.num_heads)
        self.linear = PositionWiseFFN(self.feedforward_dim, self.input_dim)
        self.norm1 = nn.RMSNorm(self.dropout)
        self.norm2 = nn.RMSNorm(self.dropout)
        self.dropout1 = nn.Dropout(self.dropout)
        self.dropout2 = nn.Dropout(self.dropout)

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
        x = self.norm1(x)
        attended_x, attention = self.attention(x, x, mask=mask)
        x = self.dropout1(x, deterministic=not training)
        x += attended_x
        x = self.norm2(x)
        fc_out=self.linear(x) 
        x = self.dropout1(fc_out, deterministic=not training) + x
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
        self.attention1 = RotaryMultiHeadAttention(hidden_dim=self.input_dim, num_heads=self.num_heads)
        self.attention2 = RotaryMultiHeadAttention(hidden_dim=self.input_dim, num_heads=self.num_heads)
        self.feed_forward = PositionWiseFFN(self.feedforward_dim, self.input_dim)
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
        x += attended_x

        return x, jnp.array(attention1), jnp.array(attention2)
    

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
        self.activation = nn.glu()
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