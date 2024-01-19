from whisper import *

# Dummy data parameters
batch_size = 8
max_length = 50
embed_dim = 256 
vocab_size = 1000 

# Generate dummy text and image data
dummy_targets = jnp.arange(batch_size * max_length, dtype=jnp.int32).reshape((batch_size, max_length))
dummy_inputs = jnp.ones((batch_size, max_length, embed_dim))

# model parameters
hyperparams = {
    'num_layers': 1,
    'hidden_dim': 256,
    'num_heads': 2,
    'feedforward_dim': 256,
    'dropout': 0.1,
    'vocab_size': 1000,
    'embed_dim': embed_dim,
    'max_length': max_length,
    'start_token': 0,
    'end_token': 50,
}

# Initialize model
model = Whisper(**hyperparams)
rngs = {'params': jax.random.key(0), 'dropout': jax.random.key(1)}
params = model.init(rngs, dummy_inputs, dummy_targets)['params']
outputs = model.apply({'params': params}, dummy_inputs, dummy_targets, rngs=rngs)
print(outputs.shape)

# Training on your data
dataloader = [(dummy_inputs, dummy_targets)] * 10
trainer = WhisperDataParallelTrainer(model, dummy_inputs.shape, dummy_targets.shape, 'params.pkl')
trainer.train(dataloader, 10, dataloader)
print(trainer.evaluate(dataloader))