from mixer import *

# Dummy data parameters
batch_size = 8
max_length = 50 
n_outputs = 5  
embed_dim = 256  
patch_size = (16, 16)  

# Generate dummy text and image data
dummy_inputs = jnp.ones((batch_size, 224, 224, 3))
key = jax.random.PRNGKey(10)
dummy_labels = jax.random.randint(key, shape=(batch_size,), minval=0, maxval=n_outputs-1)

# model parameters
hyperparams = {
    "dropout": 0.1,
    "num_heads": 2,
    "feedforward_dim": embed_dim,
    "patch_size": patch_size,
    "hidden_dim": embed_dim,
    "num_layers": 4,
    "n_outputs": n_outputs
}

# Initialize model
model = Mixer(**hyperparams)
rngs = {'params': jax.random.key(0), 'dropout': jax.random.key(1)}
params = model.init(rngs, dummy_inputs)['params']
outputs = model.apply({'params': params}, dummy_inputs, rngs=rngs)[0]
print(outputs.shape)

# Training on your data
dataloader = [(dummy_inputs, dummy_labels)] * 10
trainer = MixerDataParallelTrainer(model, dummy_inputs.shape, 'params.pkl')
trainer.train(dataloader, 10, dataloader)
print(trainer.evaluate(dataloader))