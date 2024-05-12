import time
from typing import Any, Iterable, List, Optional, Tuple

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import optax
from einops import rearrange
from flax.training import train_state


class PatchEmbedding(nn.Module):
    """
    Implements patch embedding for vision transformers.

    This module utilises a 2D conv layer to project patches of from the image to a specified embedding dimension.

    Attributes:
        image_size (int): Size of square image.
        patch_size (int): Size of square patches from image.
        embed_dim (int): Dimension of the embeddings for the patches.

    """

    image_size: int
    patch_size: int
    embed_dim: int
    num_channels: int

    def setup(self):
        self.num_patches = (self.image_size**2) // (self.patch_size**2)

        # Use sliding window from conv layer implementation to avoid "splitting" the image.
        self.proj = nn.Conv(
            features=self.embed_dim,
            kernel_size=(self.patch_size, self.patch_size),
            strides=self.patch_size,
            padding="VALID",
        )

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = self.proj(x)
        x = jnp.reshape(
            x, (x.shape[0], -1, self.embed_dim)
        )  # (batch_size, num_patches, embed_dim)
        return x


class PositionalEmbedding(nn.Module):
    """
    Implements Learnt Positional Embedding.

    This module adds a learnt vector to the patch embeddings to introduce a notion of temporal / spatial dependence.

    Attributes:
        embed_dim (int): Patch embedding dimensions.
        num_patches (int): Number of patches in an image which is dependent on the patch size.

    """

    embed_dim: int
    num_patches: int

    def setup(self):
        self.embedding = nn.Embed(
            num_embeddings=self.num_patches, features=self.embed_dim
        )

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        positions = jnp.arange(x.shape[1])[jnp.newaxis, :].repeat(x.shape[0], axis=0)
        embed = self.embedding(positions)
        x = x + embed
        return x


class MultiHeadedAttention(nn.Module):
    """
    Implements the multi-head attention mechanism as described in "Attention is All You Need" by Vaswani et al 2017.

    This module splits the input into multiple heads, applies scaled dot-product attention independently on each head, and then concatenates the results. It allows the model to jointly attend to information from different representation subspaces at different positions.

    Attributes:
        embed_dim (int): Dimensionality of the input and output features.
        num_heads (int): Number of attention heads.

    """

    embed_dim: int
    num_heads: int

    def setup(self):
        self.attn_proj = nn.Dense(
            3 * self.embed_dim,
            kernel_init=nn.initializers.xavier_uniform(),
            bias_init=nn.initializers.zeros,
        )
        self.out_proj = nn.Dense(
            self.embed_dim,
            kernel_init=nn.initializers.xavier_uniform(),
            bias_init=nn.initializers.zeros,
        )

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        qkv = self.attn_proj(x)
        query, key, value = jnp.array_split(qkv, 3, axis=-1)
        query = jnp.reshape(query, (query.shape[0], query.shape[1], self.num_heads, -1))
        key = jnp.reshape(key, (key.shape[0], key.shape[1], self.num_heads, -1))
        value = jnp.reshape(value, (value.shape[0], value.shape[1], self.num_heads, -1))
        query = jnp.permute_dims(query, (0, 2, 1, 3))
        key = jnp.permute_dims(key, (0, 2, 1, 3))
        value = jnp.permute_dims(value, (0, 2, 1, 3))
        attn_weights = jnp.matmul(query, key.transpose(0, 1, 3, 2)) / (
            self.embed_dim**0.5
        )
        attn_weights = nn.softmax(attn_weights, -1)
        attn = jnp.matmul(attn_weights, value)
        attn = jnp.reshape(attn, (query.shape[0], -1, self.embed_dim))
        attn = self.out_proj(attn)
        return attn, attn_weights


class TransformerEncoderBlock(nn.Module):
    """
    Implements a Transformer Encoder Block.

    The transformer encoder block is composed of an attention block and a feedforward block. The sublayers have residual connections followed by a Layer Norm.

    Attributes:
        embed_dim (int): Dimensionality of the input and output features.
        num_heads (int): Number of attention heads.
        feed_forward_dim (int): Dimension of the feed-forward network.
        dropout_p (float): Dropout rate.

    """

    embed_dim: int
    num_heads: int
    feed_forward_dim: int
    dropout_p: float

    def setup(self):
        self.norm1 = nn.LayerNorm()
        self.norm2 = nn.LayerNorm()

        self.ff = nn.Sequential(
            [
                nn.Dense(self.feed_forward_dim),
                lambda x: nn.gelu(x),
                nn.Dense(self.embed_dim),
            ]
        )

        self.attn = MultiHeadedAttention(
            embed_dim=self.embed_dim,
            num_heads=self.embed_dim,
        )

        self.dropout = nn.Dropout(self.dropout_p)

    def __call__(self, x: jnp.ndarray, training: bool) -> jnp.ndarray:
        x_, attn_weights = self.attn(self.norm1(x))
        x = x + x_
        x = self.dropout(x, deterministic=not training)
        x = x + self.ff(self.norm2(x))
        x = self.dropout(x, deterministic=not training)
        return x, attn_weights


class TransformerEncoder(nn.Module):
    """
    Implements a Transformer Encoder Block.

    The transformer encoder block is composed of an attention block and a feedforward block. The sublayers have residual connections followed by a Layer Norm.

    Attributes:
        dropout (int): dropout probability.
        num_heads (int): Number of attention heads.
        embed_dim (int): Dimensionality of inputs and outputs.
        num_layers (int): Number of encoder blocks.
        feed_forward_dim (int): Dimension of the feed-forward network.

    """

    dropout: float
    num_heads: int
    embed_dim: int
    num_layers: int
    feed_forward_dim: int

    def setup(self):
        self.layers = [
            TransformerEncoderBlock(
                embed_dim=self.embed_dim,
                num_heads=self.num_heads,
                feed_forward_dim=self.feed_forward_dim,
                dropout_p=self.dropout,
            )
            for _ in range(self.num_layers)
        ]

    def __call__(self, x: jnp.ndarray, training: bool) -> jnp.ndarray:
        attn_maps = []
        for layer in self.layers:
            x, attn_weights = layer(x, training=training)
            attn_maps.append(attn_weights)
        return x, jnp.array(attn_maps)


class IJEPA(nn.Module):
    """
    Implements the IJEPA architecture for non-generative self-supervised learning.
    Ref: "Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture" by Mahmoud Assran et al.

    This module consists of three ViTs / Transformer Encoders; A context and target encoder and an embedding predictor.
    The embedding predictor is trained to predict the outputs of the target encoder given the outputs of the context encoder.

    Attributes:
        image_size (int): Image size. Assuming image is a square image.
        num_channels (int): Number of image channels.
        patch_size (int): Patch size for ViTs. Assuming patch size is a square and image is a square image.
        embed_dim (int): Embedding dimensions for ViTs.
        num_heads (int): Number of transformer encoder heads for context and target encoders.
        dropout_p (float): Dropout probability.
        predictor_num_heads (int): Number of transformer encoder heads for embedding predictor.
        share_patch_embedding (bool): Whether or not to share the patch embeddings across the context and target encoders.

    Example usage:
        ```py
        import jax
        import jax.numpy as jnp
        from nanodl import ArrayDataset, DataLoader
        from nanodl import IJEPA, IJEPADataSampler, IJEPADataParallelTrainer

        # Dummy data parameters
        batch_size = 8
        embed_dim = 256
        patch_size = 16
        image_size = 256
        M=4

        num_patches = (256 * 256) // (patch_size * patch_size)

        # Generate data
        dummy_inputs = jnp.ones((batch_size, image_size, image_size, 3))
        dummy_context_masks = jnp.zeros((batch_size, M, num_patches, embed_dim))
        dummy_target_masks = jnp.zeros((batch_size, M, num_patches, embed_dim))

        key = jax.random.PRNGKey(10)

        # Create dataset and dataloader
        dataset = ArrayDataset(dummy_inputs)

        dataloader = DataLoader(dataset,
                                batch_size=batch_size,
                                shuffle=True,
                                drop_last=False)

        data_sampler = IJEPADataSampler(
            image_size=img_size,
            patch_size=patch_size
        )

        # model parameters
        hyperparams = {
            "image_size": 256
            "num_channels": 3
            "patch_size": patch_size,
            "embed_dim": embed_dim,
            "num_heads": 4,
            "num_layers": 4,
            "dropout_p": 0.1,
            "predictor_num_heads": 4,
            "predictor_bottleneck": 128,
            "predictor_num_layers": 2
        }

        # Initialize model
        model = IJEPA(**hyperparams)
        rngs = {'params': jax.random.key(0), 'dropout': jax.random.key(1)}
        params = model.init(rngs, dummy_inputs, dummy_context_masks, dummy_target_masks)['params']

        outputs, _ = model.apply(
            {'params': params},
            dummy_inputs,
            dummy_context_mask,
            dummy_target_mask,
            rngs=rngs
        )

        print(outputs.shape)

        # Training on your data
        trainer = IJEPADataParallelTrainer(model, dummy_inputs.shape, 'params.pkl', data_sampler=data_sampler)
        trainer.train(dataloader, 10, dataloader)
        ```
    """

    image_size: int
    num_channels: int
    patch_size: int
    embed_dim: int
    num_heads: int
    num_layers: int
    dropout_p: float
    predictor_num_heads: int
    predictor_bottleneck: int
    predictor_num_layers: int
    share_patch_embedding: bool = True

    def setup(self):
        self.num_patches = (self.image_size**2) // (self.patch_size**2)

        self.feed_forward_dim = self.embed_dim * 4
        self.predictor_feed_forward_dim = self.predictor_bottleneck * 4

        create_patch_embedding = lambda: PatchEmbedding(
            image_size=self.image_size,
            patch_size=self.patch_size,
            embed_dim=self.embed_dim,
            num_channels=self.num_channels,
        )

        if (
            self.share_patch_embedding
        ):  # We could have the context and target decoder share the patch emebddings
            patch_embedding = create_patch_embedding()
            self.patch_embedding = {
                "context": patch_embedding,
                "target": patch_embedding,
            }

        else:  # Or have them learn different patch embeddings
            self.patch_embedding = {
                "context": create_patch_embedding(),
                "target": create_patch_embedding(),
            }

        # because the positional embedding is constant, doesn't need to be shared.
        self.positional_embedding = PositionalEmbedding(
            embed_dim=self.embed_dim, num_patches=self.num_patches
        )

        self.context_encoder = TransformerEncoder(
            dropout=self.dropout_p,
            num_heads=self.num_heads,
            embed_dim=self.embed_dim,
            num_layers=self.num_layers,
            feed_forward_dim=self.feed_forward_dim,
        )

        self.target_encoder = TransformerEncoder(
            dropout=self.dropout_p,
            num_heads=self.num_heads,
            embed_dim=self.embed_dim,
            num_layers=self.num_layers,
            feed_forward_dim=self.feed_forward_dim,
        )

        self.embedding_predictor = TransformerEncoder(
            dropout=self.dropout_p,
            num_heads=self.predictor_num_heads,
            embed_dim=self.predictor_bottleneck,
            num_layers=self.predictor_num_layers,
            feed_forward_dim=self.predictor_feed_forward_dim,
        )

        self.to_predictor_embed = nn.Dense(self.predictor_bottleneck)
        self.to_encoder_embed = nn.Dense(self.embed_dim)

    def __call__(
        self,
        x: jnp.ndarray,
        context_mask: jnp.ndarray,
        target_mask: jnp.ndarray,
        training: bool = False,
    ) -> Tuple[
        List[Tuple[jnp.ndarray, jnp.ndarray]],
        List[Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]],
    ]:
        x_context = self.patch_embedding["context"](x)
        x_context = self.positional_embedding(x_context)
        x_target = self.patch_embedding["target"](x)
        x_target = self.positional_embedding(x_target)

        outputs = []
        attn_weights = []

        for m in range(context_mask.shape[1]):
            context, context_attn_weights = self.context_encoder(
                x_context, training=training
            )
            context = context * jnp.expand_dims(
                context_mask[:, m], -1
            )  # (N, num_patches, E)
            target, target_attn_weights = self.target_encoder(
                x_target, training=training
            )
            target = target * jnp.expand_dims(
                target_mask[:, m], -1
            )  # (N, num_patches, E)

            predicted_embeddings, embed_attn_weights = self.embedding_predictor(
                self.to_predictor_embed(context), training=training
            )

            predicted_embeddings = self.to_encoder_embed(predicted_embeddings)
            predicted_embeddings = predicted_embeddings * jnp.expand_dims(
                target_mask[:, m], -1
            )
            outputs.append((predicted_embeddings, target))
            attn_weights.append(
                (context_attn_weights, target_attn_weights, embed_attn_weights)
            )

        return (outputs, attn_weights)


class IJEPADataSampler:
    """
    Implements a data sampler for the IJEPA model.

    The data sampler is used to sample data for the IJEPA model. 
    It samples the scale of the target block using a uniform random distribution and scales it within the target scale range. 
    Also samples the scale of the context using a uniform random distribution and scales it within the context scale range.

    Attributes:
        image_size (int): The size of the image.
        patch_size (int): The size of the patches into which the image is divided.
        M (int): The number of patches.
        context_scale_range (tuple): The range of scales for the context.
        target_scale_range (tuple): The range of scales for the target.
        target_aspect_ratio_range (tuple): The range of aspect ratios for the target.
        h (int): The height of the image divided by the patch size.
        w (int): The width of the image divided by the patch size.
        to_scale (function): A function to scale a value within a specified range.
        random_key (int): A seed for generating random numbers.
    """
    to_scale: Any = lambda self, x, a, b: (b - a) * x + a
    random_key: int = 0
    random_key = jax.random.PRNGKey(random_key)

    def __init__(
        self,
        image_size: int = 256,
        patch_size: int = 16,
        M: int = 4,
        context_scale_range: tuple = (0.85, 1),
        target_scale_range: tuple = (0.15, 0.2),
        target_aspect_ratio_range: tuple = (0.75, 1.5),
    ):

        self.image_size = image_size
        self.patch_size = patch_size
        self.M = M
        self.context_scale_range = context_scale_range
        self.target_scale_range = target_scale_range
        self.target_aspect_ratio_range = target_aspect_ratio_range

        self.h = image_size // patch_size
        self.w = image_size // patch_size

    def sample_target_block_scale(self) -> Tuple[int, int]:
        scale = self.to_scale(
            jax.random.uniform(self.random_key),
            self.target_scale_range[0],
            self.target_scale_range[1],
        )

        context_scale = self.to_scale(
            jax.random.uniform(self.random_key),
            self.context_scale_range[0],
            self.context_scale_range[1],
        )

        aspect_ratio = self.to_scale(
            jax.random.uniform(self.random_key),
            self.target_aspect_ratio_range[0],
            self.target_aspect_ratio_range[1],
        )

        target_mask_scale = int(self.h * self.w * scale * context_scale)

        target_h = int((target_mask_scale * aspect_ratio) ** 0.5)
        target_w = int((target_mask_scale / aspect_ratio) ** 0.5)

        if target_h >= self.h:
            target_h -= target_h - self.h - 1
        if target_w >= self.w:
            target_w -= target_w - self.w - 1

        return target_h, target_w

    def sample_context_target_blocks(
        self, h: int, w: int
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        context_mask = jnp.ones((self.M, self.image_size, self.image_size))
        target_mask = jnp.zeros((self.M, self.image_size, self.image_size))

        for m in range(self.M):
            top = jax.random.randint(self.random_key, (), 0, self.h - h)
            left = jax.random.randint(self.random_key, (), 0, self.w - w)

            context_mask = context_mask.at[
                m,
                top * self.patch_size : (top + h) * self.patch_size,
                left * self.patch_size : (left + w) * self.patch_size,
            ].set(0)

            target_mask = target_mask.at[
                m,
                top * self.patch_size : (top + h) * self.patch_size,
                left * self.patch_size : (left + w) * self.patch_size,
            ].set(1)

        context_mask = rearrange(
            context_mask,
            "m (p1 h) (p2 w) -> m (h w) (p1 p2)",
            p1=self.patch_size,
            p2=self.patch_size,
        )
        target_mask = rearrange(
            target_mask,
            "m (p1 h) (p2 w) -> m (h w) (p1 p2)",
            p1=self.patch_size,
            p2=self.patch_size,
        )

        context_mask = jnp.any(context_mask == 1, axis=-1)
        target_mask = jnp.any(target_mask == 0, axis=-1)

        return context_mask, target_mask

    def __call__(self) -> Tuple[jnp.ndarray, jnp.ndarray]:
        h, w = self.sample_target_block_scale()
        context_mask, target_mask = self.sample_context_target_blocks(h, w)

        return context_mask, target_mask


class IJEPADataParallelTrainer:
    """
    Implements a parallel trainer for the IJEPA model.

    The IJEPADataParallelTrainer is used to train the IJEPA model in parallel. 
    
    Attributes:
        model (Any): The model to be trained.
        input_shape (Tuple[int, ...]): The shape of the input data.
        weights_filename (str): The filename of the weights of the model.
        data_sampler (IJEPADataSampler): The data sampler used to sample data for training.
        learning_rate (float): The learning rate for training. Default is 1e-4.
        params_path (str, optional): The path to the parameters of the model. Default is None.
        params (Any): The parameters of the model. Initialized as None.
        num_parameters (int): The number of parameters in the model. Initialized as None.
        best_val_loss (float): The best validation loss achieved during training. Initialized as infinity.
        num_devices (int): The number of devices used for training.
        train_step (function): The function used to perform a training step.
        evaluation_step (function): The function used to perform an evaluation step.
        state (Any): The state of the model during training.
    """
    def __init__(
        self,
        model: Any,
        input_shape: Tuple[int, ...],
        weights_filename: str,
        data_sampler: IJEPADataSampler,
        learning_rate: float = 1e-4,
        params_path: Optional[str] = None,
    ) -> None:

        self.model = model
        self.params = None
        self.params_path = params_path
        self.num_parameters = None
        self.best_val_loss = float("inf")
        self.weights_filename = weights_filename
        self.data_sampler = data_sampler
        self.num_devices = jax.local_device_count()
        self.train_step = jax.pmap(
            IJEPADataParallelTrainer.train_step, axis_name="devices"
        )
        self.evaluation_step = jax.pmap(
            IJEPADataParallelTrainer.evaluation_step, axis_name="devices"
        )
        self.state = self.create_train_state(learning_rate, input_shape)
        print(f"Number of accelerators: {self.num_devices}")

    def create_train_state(
        self, learning_rate: float, input_shape: Tuple[int, ...]
    ) -> Any:

        rngs = {"params": jax.random.key(0), "dropout": jax.random.key(1)}
        context_mask, target_mask = self.data_sampler()
        context_mask = jnp.repeat(context_mask[jnp.newaxis], input_shape[0], axis=0)
        target_mask = jnp.repeat(target_mask[jnp.newaxis], input_shape[0], axis=0)
        params = self.model.init(
            rngs, jnp.ones(input_shape), context_mask, target_mask
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
        state: Any,
        images: jnp.ndarray,
        context_mask: jnp.ndarray,
        target_mask: jnp.ndarray,
    ) -> Tuple[Any, jnp.ndarray]:

        def loss_fn(params):
            outputs, _ = state.apply_fn(
                {"params": params},
                images,
                context_mask=context_mask,
                target_mask=target_mask,
                training=True,
                rngs={"dropout": jax.random.PRNGKey(int(time.time()))},
            )

            losses = jnp.array(
                [
                    jnp.mean(jnp.square(outputs[i][0] - outputs[i][1]))
                    for i in range(len(outputs))
                ]
            )

            return jnp.mean(losses)

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
            for images in train_loader:
                images = images[0] if len(images) == 1 else images

                batch_size = images.shape[0]
                batch_size_per_device = batch_size // self.num_devices
                images = images.reshape(
                    (
                        self.num_devices,
                        batch_size_per_device,
                        images.shape[1],
                        images.shape[2],
                        images.shape[3],
                    )
                )

                context_mask, target_mask = self.data_sampler()

                context_mask = jnp.repeat(context_mask[jnp.newaxis], batch_size, axis=0)
                target_mask = jnp.repeat(target_mask[jnp.newaxis], batch_size, axis=0)

                context_mask = context_mask.reshape(
                    (
                        self.num_devices,
                        batch_size_per_device,
                        context_mask.shape[1],
                        context_mask.shape[2],
                    )
                )
                target_mask = target_mask.reshape(
                    (
                        self.num_devices,
                        batch_size_per_device,
                        target_mask.shape[1],
                        target_mask.shape[2],
                    )
                )

                self.state, loss = self.train_step(
                    state=self.state,
                    images=images,
                    context_mask=context_mask,
                    target_mask=target_mask,
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
        state: Any,
        images: jnp.ndarray,
        context_mask: jnp.ndarray,
        target_mask: jnp.ndarray,
    ) -> Tuple[Any, jnp.ndarray]:
        outputs, _ = state.apply_fn(
            {"params": state.params},
            images,
            context_mask=context_mask,
            target_mask=target_mask,
            rngs={"dropout": jax.random.PRNGKey(int(time.time()))},
        )

        losses = jnp.array(
            [
                jnp.mean(jnp.square(outputs[i][0] - outputs[i][1]))
                for i in range(len(outputs))
            ]
        )

        return jnp.mean(losses)

    def evaluate(self, test_loader: Iterable[Tuple[jnp.ndarray, jnp.ndarray]]) -> None:

        total_loss = 0.0
        count = 0
        for images in test_loader:
            images = images[0] if len(images) == 1 else images

            batch_size = images.shape[0]
            batch_size_per_device = batch_size // self.num_devices
            images = images.reshape(
                (
                    self.num_devices,
                    batch_size_per_device,
                    images.shape[1],
                    images.shape[2],
                    images.shape[3],
                )
            )

            context_mask, target_mask = self.data_sampler()

            context_mask = jnp.repeat(context_mask[jnp.newaxis], batch_size, axis=0)
            target_mask = jnp.repeat(target_mask[jnp.newaxis], batch_size, axis=0)

            context_mask = context_mask.reshape(
                (
                    self.num_devices,
                    batch_size_per_device,
                    context_mask.shape[1],
                    context_mask.shape[2],
                )
            )

            target_mask = target_mask.reshape(
                (
                    self.num_devices,
                    batch_size_per_device,
                    target_mask.shape[1],
                    target_mask.shape[2],
                )
            )

            loss = self.evaluation_step(
                state=self.state,
                images=images,
                context_mask=context_mask,
                target_mask=target_mask,
            )

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
