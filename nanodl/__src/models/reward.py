import time
from typing import Any, Iterable, Optional, Tuple

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import optax
from flax.training import train_state


class RewardModel(nn.Module):
    """
    The RewardModel estimates the reward or value of a given input sequence,
    typically used in reinforcement learning frameworks for natural language processing tasks.
    It uses the last hidden state of a transformer-based model to generate a scalar reward prediction,
    guiding the agent's behavior by evaluating the desirability or utility of its generated outputs.

    Args:
        model (nn.Module): The neural network model to be used.
        dim (int): The dimension of the input data.
        dropout (float): The dropout rate for the model, a value between 0 and 1. 

    Example:
        ```python
        from nanodl import ArrayDataset, DataLoader
        from nanodl import Gemma, RewardModel, RewardDataParallelTrainer

        # Generate dummy data
        batch_size = 8
        max_length = 10

        # Replace with actual tokenised data
        dummy_chosen = jnp.ones((101, max_length), dtype=jnp.int32)
        dummy_rejected = jnp.zeros((101, max_length), dtype=jnp.int32)

        # Create dataset and dataloader
        dataset = ArrayDataset(dummy_chosen, dummy_rejected)
        dataloader = DataLoader(dataset,
                                batch_size=batch_size,
                                shuffle=True,
                                drop_last=False)

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

        # Initialize reward model from Gemma
        model = Gemma(**hyperparams)
        reward_model = RewardModel(model, dim=hyperparams['hidden_dim'], dropout=0.1)

        # Train the reward model
        trainer = RewardDataParallelTrainer(reward_model, dummy_chosen.shape, 'reward_model_weights.pkl')
        trainer.train(dataloader, 5, dataloader)
        params = trainer.load_params('reward_model_weights.pkl')

        # Call as you would a regular Flax model
        rngs = jax.random.PRNGKey(0)
        rngs, dropout_rng = jax.random.split(rngs)
        rewards = reward_model.apply({'params': params},
                            dummy_chosen,
                            rngs={'dropout': dropout_rng})

        print(rewards.shape)
        ```
    """

    model: nn.Module
    dim: int
    dropout: float

    @nn.compact
    def __call__(self, x: jnp.ndarray, training: bool = False):

        x = self.model(x, training=training, drop_last_layer=True)
        x = nn.Dropout(rate=self.dropout)(x, deterministic=not training)
        x = nn.Dense(1)(x)
        return nn.sigmoid(x)[:, -1, 0]


class RewardDataParallelTrainer:
    """
    Trainer class using data parallelism with JAX.
    This trainer leverages JAX's `pmap` for parallel training across multiple devices (GPUs/TPUs).
    It handles the model training loop, including gradient computation, parameter updates, and evaluation.

    Attributes:
        model (Any): The model to be trained.
        input_shape (Tuple[int, ...]): The shape of the input tensor.
        weights_filename (str): Filename where the trained model weights will be saved.
        learning_rate (float): Learning rate for the optimizer.
        params_path (Optional[str]): Path to pre-trained reward model parameters for initializing the REWARD model, if available.
        model_params_path (Optional[str]): Path to pre-trained backbone model parameters for initializing the BACKBONE model, if available.

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
        model_params_path: Optional[str] = None,
    ) -> None:

        self.model = model
        self.params = None
        self.params_path = params_path
        self.model_params_path = model_params_path
        self.num_parameters = None
        self.best_val_loss = float("inf")
        self.weights_filename = weights_filename
        self.num_devices = jax.local_device_count()
        self.train_step = jax.pmap(
            RewardDataParallelTrainer.train_step, axis_name="devices"
        )
        self.evaluation_step = jax.pmap(
            RewardDataParallelTrainer.evaluation_step, axis_name="devices"
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

        if self.model_params_path is not None:
            model_params = self.load_params(self.model_params_path)
            params = self.merge_params(model_params, params)

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
        state: Any, chosen: jnp.ndarray, rejected: jnp.ndarray
    ) -> Tuple[Any, jnp.ndarray]:

        def loss_fn(params):
            chosen_rewards = state.apply_fn(
                {"params": params},
                chosen,
                training=True,
                rngs={"dropout": jax.random.PRNGKey(int(time.time()))},
            )

            rejected_rewards = state.apply_fn(
                {"params": params},
                rejected,
                training=True,
                rngs={"dropout": jax.random.PRNGKey(int(time.time()))},
            )

            return -jnp.log(jax.nn.sigmoid(chosen_rewards - rejected_rewards)).mean()

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
            for chosen, rejected in train_loader:
                batch_size = chosen.shape[0]
                batch_size_per_device = batch_size // self.num_devices
                chosen = chosen.reshape((self.num_devices, batch_size_per_device, -1))
                rejected = rejected.reshape(
                    (self.num_devices, batch_size_per_device, -1)
                )
                self.state, loss = self.train_step(
                    state=self.state, chosen=chosen, rejected=rejected
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
        state: Any, chosen: jnp.ndarray, rejected: jnp.ndarray
    ) -> Tuple[Any, jnp.ndarray]:
        chosen_rewards = state.apply_fn(
            {"params": state.params}, chosen, rngs={"dropout": jax.random.PRNGKey(2)}
        )
        rejected_rewards = state.apply_fn(
            {"params": state.params}, rejected, rngs={"dropout": jax.random.PRNGKey(2)}
        )
        return -jnp.log(jax.nn.sigmoid(chosen_rewards - rejected_rewards)).mean()

    def evaluate(self, test_loader: Iterable[Tuple[jnp.ndarray, jnp.ndarray]]) -> None:

        total_loss = 0.0
        count = 0
        for chosen, rejected in test_loader:
            batch_size = chosen.shape[0]
            batch_size_per_device = batch_size // self.num_devices
            chosen = chosen.reshape((self.num_devices, batch_size_per_device, -1))
            rejected = rejected.reshape((self.num_devices, batch_size_per_device, -1))
            loss = self.evaluation_step(self.state, chosen, rejected)
            total_loss += jnp.mean(loss)
            count += 1

        mean_loss = total_loss / count
        return mean_loss

    def merge_params(untrained_params, trained_params):
        updated_untrained_params = jax.tree_map(
            lambda untrained, trained: (
                trained if untrained.shape == trained.shape else untrained
            ),
            untrained_params,
            trained_params,
        )
        return updated_untrained_params

    def save_params(self) -> None:
        self.params = flax.jax_utils.unreplicate(self.state.params)
        with open(self.weights_filename, "wb") as f:
            f.write(flax.serialization.to_bytes(self.params))

    def load_params(self, filename: str):
        with open(filename, "rb") as f:
            self.params = flax.serialization.from_bytes(self.params, f.read())
        return self.params
