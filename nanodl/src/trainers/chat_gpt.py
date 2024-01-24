'''
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
```
from gpt3 import *

# Dummy data parameters
batch_size = 8
max_length = 51
vocab_size = 1000 
embed_dim = 256 

# Generate dummy data
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
model = GPT3(**hyperparams)
rngs = {'params': jax.random.key(0), 'dropout': jax.random.key(1)}
params = model.init(rngs, dummy_inputs)['params']
outputs = model.apply({'params': params}, dummy_inputs, rngs={'dropout': jax.random.PRNGKey(2)})
print(outputs.shape)

# Training on your data
dataloader = [(dummy_inputs, dummy_targets)] * 10
trainer = GPT3DataParallelTrainer(model, dummy_inputs.shape, 'params.pkl')
trainer.train(dataloader, num_epochs=2)
print(trainer.evaluate(dataloader))

# Generate: should always have dims (batch_size, seq_len)
start_tokens = jnp.array([[123, 456], [145, 656]])

params = trainer.load_params('params.pkl')
outputs = model.apply({'params': params},
                      start_tokens, 
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
from typing import List, Tuple, Any, Optional, Iterable 


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
    

class MixtureOfExperts(nn.Module):
    """
    Mixture of Experts Layer.

    This layer consists of multiple expert feed-forward networks and a gating mechanism
    to determine the contribution of each expert based on the input.

    Attributes:
        num_experts (int): Number of experts in the mixture.
        num_hiddens (int): Number of hidden units in each expert.
        num_outputs (int): Number of output units in the final layer after combining expert outputs.

    Args:
        num_experts (int): Number of experts.
        num_hiddens (int): Number of hidden units in each expert network.
        num_outputs (int): Number of output units in the final layer.
    """
    num_experts: int
    num_hiddens: int
    num_outputs: int

    def setup(self):
        self.experts = [nn.Dense(self.num_hiddens, 
                                kernel_init=nn.initializers.xavier_uniform()) for _ in range(self.num_experts)
                                ]
        self.gate = nn.Dense(self.num_experts, 
                            kernel_init=nn.initializers.xavier_uniform()
                            )
        self.dense_final = nn.Dense(self.num_outputs, 
                                    kernel_init=nn.initializers.xavier_uniform()
                                    )
        self.activation = GEGLU(self.num_hiddens)

    def __call__(self, X: jnp.ndarray) -> jnp.ndarray:
        """
        Forward pass through the Mixture of Experts layer.

        Args:
            X (jnp.ndarray): Input tensor.

        Returns:
            jnp.ndarray: Output tensor after processing through the MoE layer.
        """
        gating_weights = nn.softmax(self.gate(X), axis=-1)
        expert_outputs = jnp.stack([expert(X) for expert in self.experts], axis=2)
        gating_weights = gating_weights[..., None]
        mixed_expert_output = jnp.sum(gating_weights * expert_outputs, axis=2)
        return self.dense_final(self.activation(mixed_expert_output))
    

class MixtureOfExpertMLP(nn.Module):
    """
    Position-wise Feed-Forward Network with Mixture of Experts.

    Args:
        num_hiddens (int): Number of hidden units in each expert.
        num_outputs (int): Number of output units in the final layer.
        num_experts (int): Number of experts in the MoE layer.
    """
    num_hiddens: int
    num_outputs: int
    num_experts: int

    def setup(self):
        self.moe_layer = MixtureOfExperts(num_experts=self.num_experts, 
                                        num_hiddens=self.num_hiddens, 
                                        num_outputs=self.num_outputs)

    def __call__(self, X: jnp.ndarray) -> jnp.ndarray:
        return self.moe_layer(X)


class RLHF(nn.Module):

    num_layers: int
    hidden_dim: int
    num_expert: int
    n_possible_rewards: int
    decoder: nn.Module

    """
    Decoder-only model from OpenAI's GPT-3 paper: https://arxiv.org/abs/2005.14165

    Args:
        num_layers (int): Number of layers in the encoder and decoder.
        num_heads (int): Number of attention heads in the multi-head attention layers.
        feedforward_dim (int): Dimensionality of the feedforward layers.
        dropout (float): Dropout probability.
        vocab_size (int): Size of the vocabulary.
        embed_dim (int): Dimensionality of embeddings.
        max_length (int): Maximum length of generated sequences.
        start_token (int): Token ID for the start of sequence.
        end_token (int): Token ID for the end of sequence.
    """

    def setup(self):
        self.reward_head = MixtureOfExpertMLP(num_hiddens=self.hidden_dim, 
                                              num_experts=self.num_expert, 
                                              num_outputs=self.n_possible_rewards)
        
        self.policy_head = MixtureOfExpertMLP(num_hiddens=self.hidden_dim, 
                                              num_experts=self.num_expert, 
                                              num_outputs=1)
        
        
    def __call__(self, 
                 x: jnp.ndarray,
                 training: bool = True) -> jnp.ndarray:
        
        """ 
        Causal models are trained differently, the outputs are just the inputs shifted by 1
        While the generation is autoregressve, hence a different function for that
        """
        z = self.decoder(x=x, training=training, drop_last_layer=True)[0]
        rewards = self.reward_head(z) 
        rewards = jax.nn.softmax(rewards, axis=-1)
        policy_logits = self.policy_head(z)
        return rewards, policy_logits


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
                next_token = jax.random.categorical(jax.random.PRNGKey(int(time.time())), next_token_probabilities, axis=-1)

            next_token = next_token[0]
            output_sequence.append(next_token.item())
            print(decoder_input.shape, jnp.array([[next_token]]).shape)
            decoder_input = jnp.concatenate([decoder_input, jnp.array([[next_token]])], axis=1)

            if next_token.item() == self.end_token:
                break

        return tuple(output_sequence)
    

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

        for i in range(self.max_length):
            decoder_output = self.decoder(decoder_input, training=False)[0]
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
    


class RewardDataParallelTrainer:
    """
    A class for training a reward model in a data-parallel fashion.

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
                 params: Any,
                 input_shape: Tuple[int, ...],
                 weights_filename: str,
                 learning_rate: float = 1e-5,
                 params_path: Optional[str] = None) -> None:
        self.model = model
        self.params_path = params_path
        self.num_parameters = None
        self.best_val_loss = float("inf")
        self.weights_filename = weights_filename
        self.num_devices = jax.local_device_count()
        self.train_step = jax.pmap(RewardDataParallelTrainer.train_step, axis_name='devices')
        self.evaluation_step = jax.pmap(RewardDataParallelTrainer.evaluation_step, axis_name='devices')
        self.state = self.create_train_state(params, learning_rate, input_shape)
        print(f'Number of accelerators: {self.num_devices}')
    

    def create_train_state(self, 
                           pretrained_params: Any,
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
        params = self.model.init(rngs, jnp.ones(input_shape, dtype=jnp.int32))['params']
        params['self.decoder'] = pretrained_params['self.decoder']

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
                   chosen: jnp.ndarray,
                   rejected: jnp.ndarray) -> Tuple[Any, jnp.ndarray]:
        """
        Performs a single training step.

        Args:
            state: The current state of the model, including parameters and optimizer state.
            batch: A dictionary containing 'inputs' and 'targets' as keys, representing the input data.

        Returns:
            A tuple of the updated state and the loss value for this step.
        """
        def loss_fn(params):
            reward_chosen = state.apply_fn({'params': params}, 
                                    chosen, 
                                    training=True,
                                    rngs={'dropout': jax.random.PRNGKey(int(time.time()))})
            
            reward_rejected = state.apply_fn({'params': params}, 
                                    rejected, 
                                    training=True,
                                    rngs={'dropout': jax.random.PRNGKey(int(time.time()))})
            return -jnp.mean(jnp.log(jnp.sigmoid(reward_chosen - reward_rejected)))
        
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
            for chosen, rejected in train_loader:
                batch_size = chosen.shape[0]
                batch_size_per_device = batch_size // self.num_devices
                chosen = chosen.reshape((self.num_devices, batch_size_per_device, -1))
                rejected = rejected.reshape((self.num_devices, batch_size_per_device, -1))
                self.state, loss = self.train_step(state=self.state, 
                                                   chosen=chosen, 
                                                   rejected=rejected)
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
                        chosen: jnp.ndarray,
                        rejected: jnp.ndarray) -> Tuple[Any, jnp.ndarray]:
        """
        Performs a single training step.

        Args:
            state: The current state of the model, including parameters and optimizer state.
            batch: A dictionary containing 'inputs' and 'targets' as keys, representing the input data.

        Returns:
            A tuple of the updated state and the loss value for this step.
        """
        reward_chosen = state.apply_fn({'params': state.params}, 
                                    chosen, 
                                    training=False,
                                    rngs={'dropout': jax.random.PRNGKey(int(time.time()))})
            
        reward_rejected = state.apply_fn({'params': state.params}, 
                                    rejected, 
                                    training=False,
                                    rngs={'dropout': jax.random.PRNGKey(int(time.time()))})
        return -jnp.mean(jnp.log(jnp.sigmoid(reward_chosen - reward_rejected)))

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