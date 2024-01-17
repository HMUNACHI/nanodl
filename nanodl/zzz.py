from gpt import *
from lamda import *

# Dummy data parameters
batch_size = 8
max_length = 51
vocab_size = 1000 
embed_dim = 256 

# Generate dummy text and image data
data = jnp.arange(batch_size * max_length, dtype=jnp.int32).reshape((batch_size, max_length))
dummy_inputs = data[:, :-1]
dummy_targets = data[:, 1:]

# CLIP model parameters
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

# Initialize CLIP model
model = LaMDA(**hyperparams)
rngs = {'params': jax.random.key(0), 'dropout': jax.random.key(1)}
params = model.init(rngs, dummy_inputs)['params']
outputs = model.apply({'params': params}, dummy_inputs, rngs={'dropout': jax.random.PRNGKey(2)})
print(outputs.shape)

# Training on your data
dataloader = [(dummy_inputs, dummy_targets)] * 10
trainer = GPTDataParallelTrainer(model, dummy_inputs.shape, 'params.pkl')
trainer.train(dataloader, 10, dataloader)
print(trainer.evaluate(dataloader))