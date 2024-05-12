import time
from typing import Any, Iterable, Optional, Tuple

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import optax
from flax.training import train_state


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
    Combines token embeddings with positional encodings for input sequences in a transformer model.

    This module embeds tokens using learned embeddings and adds positional encodings. The positional encodings can either be learned or fixed (sine and cosine functions based) depending on the `learned_position` flag.

    Attributes:
        max_len (int): Maximum length of the input sequences.
        vocab_size (int): Size of the vocabulary.
        embed_dim (int): Dimension of the embeddings.
        learned_position (bool): Flag to use learned positional embeddings instead of fixed positional encodings.
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

    This module applies two linear transformations with a ReLU activation in between, as per the original transformer model design. It is applied to each position separately and identically.

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
        self.dense2 = nn.Dense(
            self.num_outputs, kernel_init=nn.initializers.xavier_uniform()
        )

    def __call__(self, X: jnp.ndarray) -> jnp.ndarray:
        return self.dense2(nn.gelu(self.dense1(X)))


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


class EncoderBlock(nn.Module):
    """
    Represents a single block in the transformer encoder.

    Each encoder block consists of a multi-head self-attention layer and a position-wise feed-forward network. Both sublayers have residual connections and are followed by layer normalization.

    Attributes:
        hidden_dim (int): Dimensionality of the input and output features.
        num_heads (int): Number of attention heads.
        feedforward_dim (int): Dimension of the feed-forward network.
        dropout (float): Dropout rate.
    """

    hidden_dim: int
    num_heads: int
    feedforward_dim: int
    dropout: float

    def setup(self):
        self.attention = SelfMultiHeadAttention(
            hidden_dim=self.hidden_dim, num_heads=self.num_heads
        )
        self.ff = PositionWiseFFN(self.feedforward_dim, self.hidden_dim)
        self.add_norm1 = AddNorm(self.dropout)
        self.add_norm2 = AddNorm(self.dropout)

    def __call__(
        self, x: jnp.ndarray, mask: jnp.ndarray = None, training: bool = False
    ) -> tuple:
        attended_x, attention = self.attention(x, mask=mask)
        x = self.add_norm1(x, attended_x, training)
        ff_output = self.ff(x)
        x = self.add_norm2(x, ff_output, training)
        return x, attention


class TextEncoder(nn.Module):
    """
    Implements a transformer encoder for text.

    This module combines an embedding layer (with optional learned positional encodings) with multiple encoder blocks to process sequences of text.

    Attributes:
        num_layers (int): Number of encoder blocks in the transformer.
        hidden_dim (int): Dimensionality of the input and output features.
        num_heads (int): Number of attention heads.
        feedforward_dim (int): Dimension of the feed-forward network.
        dropout (float): Dropout rate.
        max_len (int): Maximum length of the input sequences.
        vocab_size (int): Size of the vocabulary.
        embed_dim (int): Dimension of the embeddings.
        learned_position (bool): Flag to use learned positional embeddings instead of fixed positional encodings.
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
            EncoderBlock(
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


class PatchEmbedding(nn.Module):
    """
    Implements patch embedding for vision transformers.

    This module extracts patches from input images, flattens them, and projects them to a specified embedding dimension. Optionally, learned position embeddings can be added to the patch embeddings.

    Attributes:
        patch_size (tuple): Size (height, width) of the patches to extract from input images.
        embed_dim (int): Dimension of the embeddings for the patches.

    """

    patch_size: Tuple[int, int]
    embed_dim: int

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.embed_dim)(self.extract_patches(x))
        return x + nn.Embed(num_embeddings=x.shape[1], features=x.shape[2])(
            jnp.arange(x.shape[1])
        )

    def extract_patches(self, images: jnp.ndarray) -> jnp.ndarray:
        if len(images.shape) != 4:
            raise ValueError("Input images should have shape (batch_size, H, W, C)")

        batch_size, h, w, c = images.shape
        ph, pw = self.patch_size

        if h % ph != 0 or w % pw != 0:
            raise ValueError("Image dimensions must be divisible by patch size.")

        # Calculate the number of patches in each dimension
        num_patches_h = h // ph
        num_patches_w = w // pw

        # Reshape the images into patches and flatten each patch
        patches = jnp.reshape(
            images, (batch_size, num_patches_h, ph, num_patches_w, pw, c)
        )
        patches = jnp.transpose(patches, (0, 1, 3, 2, 4, 5))
        patches = jnp.reshape(patches, (batch_size, -1, ph * pw * c))
        return patches


class ImageEncoder(nn.Module):
    """
    Implements a vision transformer (ViT) encoder for image processing.

    This module applies patch embedding to input images and then processes the resulting sequence of embedded patches through multiple transformer encoder blocks.

    Attributes:
        patch_size (tuple): Size of the patches (height, width) to be extracted from input images.
        num_layers (int): Number of transformer encoder blocks.
        hidden_dim (int): Dimensionality of the input and output features for the transformer encoder.
        num_heads (int): Number of attention heads in the transformer encoder.
        feedforward_dim (int): Dimension of the feed-forward network in the transformer encoder.
        dropout (float): Dropout rate for regularization.

    """

    patch_size: Tuple[int, int]
    num_layers: int
    hidden_dim: int
    num_heads: int
    feedforward_dim: int
    dropout: float

    def setup(self):
        self.embedding = PatchEmbedding(self.patch_size, self.feedforward_dim)

        self.layers = [
            EncoderBlock(
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


class CLIP(nn.Module):
    """
    CLIP (Contrastive Language-Image Pretraining) is designed to understand and connect vision and language.
    Its motivation arises from the need to bridge the gap between textual and visual information processing in AI.
    CLIP's architecture is based on a vision-language transformer,
    which is pretrained on a large corpus of text and images from the internet,
    allowing it to learn associations between text and visuals.
    Unlike traditional models that are pretrained on single-modal data, CLIP can perform a wide range of tasks,
    including image classification, zero-shot object recognition, and even generating textual descriptions for images.
    CLIP's versatility and performance stem from its ability to encode and compare text and image representations directly,
    enabling it to generalize well across various vision and language tasks while minimizing the need for task-specific fine-tuning.

    Args:
    - embed_dim (int): Dimension of the shared embedding space.
    - dropout (float): Dropout rate for model layers.
    - n_outputs (int): Number of output classes.
    - num_heads (int): Number of attention heads in the transformer layers.
    - feedforward_dim (int): Dimension of the feedforward network in transformer layers.
    - num_layers_text (int): Number of transformer layers for text encoding.
    - hidden_dim_text (int): Input dimension for text data.
    - image_patch_size (int): Size of image patches.
    - hidden_dim_image (int): Input dimension for image data.
    - num_layers_images (int): Number of transformer layers for image encoding.

    Methods:
    - get_attention_maps(texts, images): Computes attention maps for text and images.
    - encode_text(texts): Encodes text data using the text encoder.
    - encode_image(images): Encodes image data using the image encoder.
    - embed_text(texts): Embeds text data into the shared embedding space.
    - embed_image(images): Embeds image data into the shared embedding space.

    Note:
        Text input shape: (batch_size, max_length, embed_dim)
        Image input shape: (batch_size, height, width, channels)
        Image shape after patch embedding: (batch_size, sequence_length, embed_dim)
        This image sequence length can be calculated with (height * width) / (patch_height * patch_width)

    Example Usage:
    ```
    import jax
    import jax.numpy as jnp
    from nanodl import ArrayDataset, DataLoader
    from nanodl import CLIP, CLIPDataParallelTrainer

    # Dummy data parameters
    batch_size = 8
    max_length = 50
    vocab_size = 1000
    embed_dim = 256
    patch_size = (16, 16)

    # Generate dummy text and image data
    dummy_texts = jnp.ones((batch_size, max_length), dtype=jnp.int32)
    dummy_images = jnp.ones((batch_size, 224, 224, 3))
    dataset = ArrayDataset(dummy_texts, dummy_images)
    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            shuffle=True,
                            drop_last=False)

    # CLIP model parameters
    clip_params = {
        "dropout": 0.1,
        "num_heads": 2,
        "feedforward_dim": embed_dim,
        "num_layers_text": 1,
        "hidden_dim_text": embed_dim,
        "image_patch_size": patch_size,
        "hidden_dim_image": embed_dim,
        "num_layers_images": 1,
        "max_len": max_length,
        "vocab_size": vocab_size,
        "embed_dim": embed_dim
    }

    # Initialize CLIP model
    clip_model = CLIP(**clip_params)
    rng = jax.random.PRNGKey(0)
    params = clip_model.init(rng, dummy_texts, dummy_images)['params']
    loss = clip_model.apply({'params': params}, dummy_texts, dummy_images)

    # Training on your data
    trainer = CLIPDataParallelTrainer(clip_model,
                                    dummy_texts.shape,
                                    dummy_images.shape, 'params.pkl')
    trainer.train(dataloader, 2)

    # Sample encodings
    image_encodings = clip_model.apply({'params': params},
                                    images = dummy_images,
                                    method=clip_model.encode_image)
    print(image_encodings.shape)

    # Sample embeddings
    image_embeddings = clip_model.apply({'params': params},
                                    images = dummy_images,
                                    method=clip_model.embed_image)
    print(image_embeddings.shape)
    ```
    """

    dropout: float
    num_heads: int
    feedforward_dim: int
    num_layers_text: int
    hidden_dim_text: int
    image_patch_size: int
    hidden_dim_image: int
    num_layers_images: int
    max_len: int
    vocab_size: int
    embed_dim: int

    def setup(self):
        """
        Initializes the model components and parameters.
        """
        self.text_encoder = TextEncoder(
            hidden_dim=self.hidden_dim_text,
            num_heads=self.num_heads,
            num_layers=self.num_layers_text,
            feedforward_dim=self.feedforward_dim,
            dropout=self.dropout,
            max_len=self.max_len,
            vocab_size=self.vocab_size,
            embed_dim=self.embed_dim,
        )
        self.image_encoder = ImageEncoder(
            patch_size=self.image_patch_size,
            num_layers=self.num_layers_images,
            hidden_dim=self.hidden_dim_image,
            num_heads=self.num_heads,
            feedforward_dim=self.feedforward_dim,
            dropout=self.dropout,
        )
        self.text_pooler = nn.Dense(self.embed_dim)
        self.image_pooler = nn.Dense(self.embed_dim)
        self.temperature = self.param("temperature", nn.initializers.zeros, ())

    def __call__(
        self, texts: jnp.ndarray, images: jnp.ndarray, training: bool = False
    ) -> Tuple[jnp.ndarray, jnp.ndarray, float]:

        text_latents, _ = self.text_encoder(texts, training=training)
        image_latents, _ = self.image_encoder(images, training=training)
        text_embedding = self.text_pooler(jnp.mean(text_latents, axis=1))
        image_embedding = self.image_pooler(jnp.mean(image_latents, axis=1))
        return self.clip_loss(text_embedding, image_embedding)

    def clip_loss(
        self, text_embeddings: jnp.ndarray, image_embeddings: jnp.ndarray
    ) -> float:

        def l2_normalise(x):
            return x / jnp.linalg.norm(x, axis=-1, keepdims=True)

        def cross_entropy(preds, targets):
            return (-targets * jax.nn.log_softmax(preds)).sum(axis=1).mean()

        text_embeddings = l2_normalise(text_embeddings)
        image_embeddings = l2_normalise(image_embeddings)
        similarity_matrix = (
            image_embeddings @ text_embeddings.T / (self.temperature + 0.00001)
        )
        labels = jnp.arange(similarity_matrix.shape[0])
        image_loss = cross_entropy(similarity_matrix, labels)
        text_loss = cross_entropy(similarity_matrix.T, labels)

        return (image_loss + text_loss) / 2

    def get_attention_maps(
        self, texts: jnp.ndarray, images: jnp.ndarray
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:

        _, text_attention = self.text_encoder(texts, training=False)
        _, image_attention = self.image_encoder(images, training=False)
        return text_attention, image_attention

    def encode_text(self, texts: jnp.ndarray) -> jnp.ndarray:

        return self.text_encoder(texts)[0]

    def encode_image(self, images: jnp.ndarray) -> jnp.ndarray:

        return self.image_encoder(images)[0]

    def embed_text(self, texts: jnp.ndarray) -> jnp.ndarray:

        return self.text_pooler(jnp.mean(self.text_encoder(texts)[0], axis=1))

    def embed_image(self, images: jnp.ndarray) -> jnp.ndarray:

        return self.image_pooler(jnp.mean(self.image_encoder(images)[0], axis=1))


class CLIPDataParallelTrainer:
    """
    Trainer class using data parallelism with JAX.
    This trainer leverages JAX's `pmap` for parallel training across multiple devices (GPUs/TPUs).
    It handles the model training loop, including gradient computation, parameter updates, and evaluation.

    Attributes:
        model (Any): The model to be trained.
        text_input_shape (Tuple[int, ...]): The shape of the text input tensor.
        image_input_shape (Tuple[int, ...]): The shape of the image input tensor.
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
        text_input_shape: Tuple[int, ...],
        image_input_shape: Tuple[int, ...],
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
            CLIPDataParallelTrainer.train_step, axis_name="devices"
        )
        self.evaluation_step = jax.pmap(
            CLIPDataParallelTrainer.evaluation_step, axis_name="devices"
        )
        self.state = self.create_train_state(
            learning_rate, text_input_shape, image_input_shape
        )
        print(f"Number of accelerators: {self.num_devices}")

    def create_train_state(
        self,
        learning_rate: float,
        text_input_shape: Tuple[int, ...],
        image_input_shape: Tuple[int, ...],
    ) -> Any:
        rng = jax.random.PRNGKey(0)
        params = self.model.init(
            rng,
            jnp.ones(text_input_shape, dtype=jnp.int32),
            jnp.ones(image_input_shape),
        )["params"]

        if self.params_path is not None:
            params = self.load_params(self.params_path)

        self.num_parameters = sum(
            param.size for param in jax.tree_util.tree_leaves(params)
        )
        state = train_state.TrainState.create(
            apply_fn=self.model.apply, params=params, tx=optax.adam(learning_rate)
        )
        return jax.device_put_replicated(state, jax.local_devices())

    @staticmethod
    def train_step(
        state: Any, texts: jnp.ndarray, images: jnp.ndarray
    ) -> Tuple[Any, jnp.ndarray]:

        grad_fn = jax.value_and_grad(
            lambda params: state.apply_fn(
                {"params": params},
                texts,
                images,
                training=True,
                rngs={"dropout": jax.random.PRNGKey(int(time.time()))},
            )
        )
        loss, grads = grad_fn(state.params)
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
            for texts, images in train_loader:
                batch_size = texts.shape[0]
                batch_size_per_device = batch_size // self.num_devices
                texts = texts.reshape(
                    (self.num_devices, batch_size_per_device, texts.shape[1])
                )
                images = images.reshape(
                    (
                        self.num_devices,
                        batch_size_per_device,
                        images.shape[1],
                        images.shape[2],
                        images.shape[3],
                    )
                )
                self.state, loss = self.train_step(self.state, texts, images)
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
        return

    @staticmethod
    def evaluation_step(
        state: Any, texts: jnp.ndarray, images: jnp.ndarray
    ) -> Tuple[Any, jnp.ndarray]:

        forward_fn = lambda params: state.apply_fn({"params": params}, texts, images)
        return forward_fn(state.params)

    def evaluate(self, test_loader: Iterable[Tuple[jnp.ndarray, jnp.ndarray]]) -> None:

        total_loss = 0.0
        count = 0
        for texts, images in test_loader:
            batch_size = texts.shape[0]
            batch_size_per_device = batch_size // self.num_devices
            texts = texts.reshape((self.num_devices, batch_size_per_device, -1))
            images = images.reshape((self.num_devices, batch_size_per_device, -1))
            loss = self.evaluation_step(self.state, texts, images)
            total_loss += jnp.mean(loss)
            count += 1

        return total_loss / count

    def save_params(self) -> None:
        self.params = flax.jax_utils.unreplicate(self.state.params)
        with open(self.weights_filename, "wb") as f:
            f.write(flax.serialization.to_bytes(self.params))

    def load_params(self, filename: str):
        with open(filename, "rb") as f:
            self.params = flax.serialization.from_bytes(self.params, f.read())
        return self.params
