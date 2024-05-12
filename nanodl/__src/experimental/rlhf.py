import copy
import time
from typing import Any, Iterable, Optional, Tuple

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import optax
from flax.training import train_state


class RLHF(nn.Module):
    policy_network: Any
    reference: bool = False

    def setup(self) -> None:
        self.dense1 = nn.Dense(256)
        self.dense2 = nn.Dense(256)
        self.dense3 = nn.Dense(1)

    def __call__(
        self, x: jnp.ndarray, training: bool = False
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:

        logits = self.policy_network(x, training=training)
        log_probs = logits - jax.scipy.special.logsumexp(logits, axis=-1, keepdims=True)
        probs = jnp.exp(log_probs)
        rng = jax.random.PRNGKey(int(time.time()))
        action = jax.random.categorical(rng, log_probs, axis=-1)
        entropy = -jnp.sum(probs * log_probs, axis=-1)
        action_log_probs = jnp.take_along_axis(log_probs, action[:, None], axis=-1)
        value = self.get_value(x) if not self.reference else None
        return action, action_log_probs, entropy, value

    def get_value(self, x: jnp.ndarray, training: bool = False) -> jnp.ndarray:
        hidden = self.policy_network(x, training=training, drop_last_layer=True)
        hidden = nn.relu(self.dense1(hidden))
        hidden = nn.relu(self.dense2(hidden))
        value = nn.tanh(self.dense3(hidden))
        return value

    def generate(self, x: jnp.ndarray) -> jnp.ndarray:
        return self.policy_network.generate(x)

    def generate_batch(self, x: jnp.ndarray) -> jnp.ndarray:
        return self.policy_network.generate_batch(x)


class PPODataParallelTrainer:
    def __init__(
        self,
        rlhf_main: Any,
        rlhf_ref: Any,
        reward_model: Any,
        input_shape: Tuple[int, ...],
        weights_filename: str,
        gamma: float = 0.99,
        beta: float = 0.2,
        lam: float = 0.95,
        ent_coef: float = 0.01,
        vf_coef: float = 0.5,
        learning_rate: float = 1e-4,
        params_path: Optional[str] = None,
        sft_params_path: Optional[str] = None,
        reward_params_path: Optional[str] = None,
    ) -> None:

        self.rlhf_main = rlhf_main
        self.reward_model = reward_model
        self.rlhf_ref = rlhf_ref

        self.gamma = gamma
        self.lam = lam
        self.beta = beta
        self.epsilon = 1.0e-8
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef

        self.params = None
        self.ref_params = None
        self.params_path = params_path
        self.sft_params = self.load_params(sft_params_path)

        rngs = {"params": jax.random.key(0), "dropout": jax.random.key(1)}
        reward_params = self.reward_model.init(
            rngs, jnp.ones(input_shape, dtype=jnp.int32)
        )["params"]
        self.reward_params = self.load_params(reward_params_path, params=reward_params)

        self.num_parameters = None
        self.best_val_loss = float("inf")
        self.weights_filename = weights_filename
        self.num_devices = jax.local_device_count()
        self.train_step = jax.pmap(
            PPODataParallelTrainer.train_step, axis_name="devices"
        )
        self.state = self.create_train_state(learning_rate, input_shape)
        print(f"Number of accelerators: {self.num_devices}")

    def create_train_state(
        self, learning_rate: float, input_shape: Tuple[int, ...]
    ) -> Any:

        rngs = {"params": jax.random.key(0), "dropout": jax.random.key(1)}
        params = self.rlhf_main.init(rngs, jnp.ones(input_shape, dtype=jnp.int32))[
            "params"
        ]
        params["policy_network"]["decoder"] = self.sft_params["decoder"]
        self.ref_params = copy.deepcopy(params)

        if self.params_path is not None:
            params = self.load_params(self.params_path)

        self.num_parameters = sum(
            param.size for param in jax.tree_util.tree_leaves(params)
        )
        print(f"Number of parameters: {self.num_parameters}")
        state = train_state.TrainState.create(
            apply_fn=self.rlhf_main.apply, params=params, tx=optax.adam(learning_rate)
        )

        return jax.device_put_replicated(state, jax.local_devices())

    def compute_agent_objective(
        self, model_logits, sft_logits, reward_score, gamma, beta
    ):
        ratio = nn.log_softmax(model_logits, axis=-1) - nn.log_softmax(
            sft_logits, axis=-1
        )
        left = jnp.mean(reward_score - beta * ratio.mean(axis=-1))
        right = gamma * nn.log_softmax(model_logits, axis=-1).mean(axis=-1)
        return left + right

    def advantage_and_return(self, rewards, values):
        rewards = jnp.expand_dims(rewards, axis=0)
        values = jnp.expand_dims(values, axis=0)

        gen_len = rewards.shape[1]
        lastgaelam = 0
        advantages_reversed = []

        for t in reversed(range(gen_len)):
            nextvalues = values[:, t + 1] if t < gen_len - 1 else 0.0
            delta = rewards[:, t] + self.gamma * nextvalues - values[:, t]
            lastgaelam = delta + self.gamma * self.lam * lastgaelam
            advantages_reversed.append(lastgaelam)

        # Reversing and stacking to create the correct shape for advantages
        advantages = jnp.vstack(advantages_reversed[::-1]).T
        returns = advantages + values
        advantages = jnp.squeeze(advantages, axis=0)
        returns = jnp.squeeze(returns, axis=0)
        return advantages, returns

    def calculate_loss(self, logprobs, values, entropies, ref_logprobs, rewards):
        ratio = jnp.exp(logprobs - ref_logprobs)
        clipped_ratio = jnp.clip(ratio, 1 - self.epsilon, 1 + self.epsilon)
        advantages, returns = self.advantage_and_return(rewards, values)
        value_loss = jnp.square(values - returns).mean()
        pg_loss_1 = advantages * ratio
        pg_loss_2 = advantages * clipped_ratio
        pg_loss = jnp.minimum(pg_loss_1, pg_loss_2).mean()
        loss = pg_loss - self.ent_coef * entropies.mean() + self.vf_coef * value_loss
        return loss

    def get_ref_log_probs(self, inputs: jnp.ndarray) -> jnp.ndarray:
        return self.rlhf_ref.apply(
            {"params": self.ref_params},
            inputs,
            training=True,
            rngs={"dropout": jax.random.PRNGKey(int(time.time()))},
        )

    def get_rewards(self, inputs: jnp.ndarray) -> jnp.ndarray:
        responses = self.rlhf_main.apply(
            {"params": self.params},
            inputs,
            rngs={"dropout": jax.random.PRNGKey(int(time.time()))},
            method=self.rlhf_main.generate_batch,
        )
        return self.reward_model.apply(
            {"params": self.reward_params},
            responses,
            training=False,
            rngs={"dropout": jax.random.PRNGKey(int(time.time()))},
        )

    def train_step(
        self,
        state: Any,
        inputs: jnp.ndarray,
        ref_log_probs: jnp.ndarray,
        rewards: jnp.ndarray,
    ) -> Tuple[Any, jnp.ndarray]:

        def loss_fn(params):
            _, action_log_probs, entropy, value = state.apply_fn(
                {"params": params},
                inputs,
                training=True,
                rngs={"dropout": jax.random.PRNGKey(int(time.time()))},
            )

            return self.calculate_loss(
                action_log_probs, value, entropy, ref_log_probs, rewards
            )

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
            for inputs in train_loader:
                inputs = inputs[0]
                ref_log_probs = self.get_ref_log_probs(inputs)
                rewards = self.get_rewards(inputs)
                batch_size = inputs.shape[0]
                batch_size_per_device = batch_size // self.num_devices
                inputs = inputs.reshape((self.num_devices, batch_size_per_device, -1))
                ref_log_probs = ref_log_probs.reshape(
                    (self.num_devices, batch_size_per_device, -1)
                )
                rewards = rewards.reshape((self.num_devices, batch_size_per_device, -1))
                self.state, loss = self.train_step(
                    state=self.state,
                    inputs=inputs,
                    ref_log_probs=ref_log_probs,
                    rewards=rewards,
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

    def merge_params(self, untrained_params, trained_params):
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

    def load_params(self, filename: str, params=None):
        with open(filename, "rb") as f:
            params = self.params if params is None else params
            self.params = flax.serialization.from_bytes(params, f.read())
        return self.params


# from nanodl import ArrayDataset, DataLoader
# from nanodl import Gemma, GemmaDataParallelTrainer
# from nanodl import RewardModel, RewardDataParallelTrainer
# # from nanodl import RLHF, PPODataParallelTrainer

# batch_size = 8
# max_length = 10
# model_params_path = 'base_params.pkl'
# rlhf_params_path = 'rlhf_params.pkl'
# reward_params_path = 'reward_params.pkl'

# # model parameters
# hyperparams = {
#     'num_layers': 1,
#     'hidden_dim': 128,
#     'num_heads': 2,
#     'feedforward_dim': 128,
#     'dropout': 0.1,
#     'vocab_size': 200,
#     'embed_dim': 128,
#     'max_length': max_length,
#     'start_token': 0,
#     'end_token': 50,
#     'num_groups': 2,
# }

# print('Step 1: Pretraining')
# # Replace with actual tokenised data
# data = jnp.ones((101, max_length), dtype=jnp.int32)
# dummy_inputs = data[:, :-1]
# dummy_targets = data[:, 1:]
# dataset = ArrayDataset(dummy_inputs, dummy_targets)
# dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)
# model = Gemma(**hyperparams)
# # trainer = GemmaDataParallelTrainer(model, dummy_inputs.shape, model_params_path)
# # trainer.train(train_loader=dataloader, num_epochs=2, val_loader=dataloader)

# print('\nStep 2: Superfised Fine-Tuning')
# # Replace with actual tokenised data
# dummy_prompt = jnp.ones((101, max_length), dtype=jnp.int32)
# dummy_chosen = jnp.ones((101, max_length), dtype=jnp.int32)
# dummy_rejected = jnp.zeros((101, max_length), dtype=jnp.int32)
# # dataset = ArrayDataset(dummy_prompt, dummy_chosen)
# # dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)
# # model = Gemma(**hyperparams)
# # trainer = GemmaDataParallelTrainer(model, dummy_prompt.shape, model_params_path)
# # trainer.train(train_loader=dataloader, num_epochs=2, val_loader=dataloader)

# print('\nStep 3: Train a reward model')
# dataset = ArrayDataset(dummy_chosen, dummy_rejected)
# dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)
# reward_model = RewardModel(Gemma(**hyperparams), dim=hyperparams['hidden_dim'], dropout=0.1)
# # trainer = RewardDataParallelTrainer(reward_model, dummy_chosen.shape, reward_params_path)
# # trainer.train(dataloader, 2, dataloader)

# print('\nStep 4: Train the RLHF model via PPO, using a reference model and the reward model.')
# rlhf_model = RLHF(model)
# rlhf_ref = RLHF(model, reference=True)
# dataset = ArrayDataset(dummy_chosen)
# dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)
# trainer = PPODataParallelTrainer(rlhf_model,
#                                  rlhf_ref,
#                                  reward_model,
#                                  dummy_inputs.shape,
#                                  rlhf_params_path,
#                                  sft_params_path=model_params_path,
#                                  reward_params_path=reward_params_path)

# trainer.train(dataloader, 2)
