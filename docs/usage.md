Usage
=====

Installation
------------

You will need Python 3.9 or later, and working [JAX](https://github.com/google/jax/blob/main/README.md)
installation, [FLAX](https://github.com/google/flax/blob/main/README.md)
installation, [OPTAX](https://github.com/google-deepmind/optax/blob/main/README.md)
installation (with GPU support for running training, without can only support creations).
Models can be designed and tested on CPUs but trainers are all Distributed Data-Parallel which would require a GPU with 1 to N GPUS/TPUS. For CPU-only version of JAX:

```
pip install --upgrade pip # To support manylinux2010 wheels.
pip install jax, flax, optax
```

Then, install nanodl from PyPi:

```
pip install nanodl
```

Creating a GPT Model
----------------

```py
import jax
import jax.numpy as jnp
from nanodl import ArrayDataset, DataLoader
from nanodl import GPT4, GPTDataParallelTrainer

# Generate dummy data
batch_size = 8
max_length = 10

# Replace with actual tokenised data
data = jnp.ones((101, max_length), dtype=jnp.int32)

# Shift to create next-token prediction dataset
dummy_inputs = data[:, :-1]
dummy_targets = data[:, 1:]

# Create dataset and dataloader
dataset = ArrayDataset(dummy_inputs, dummy_targets)
dataloader = DataLoader(dataset, 
                        batch_size=batch_size, 
                        shuffle=True, 
                        drop_last=False)

# How to loop through dataloader
for batch in dataloader:
    x, y = batch
    print(x.shape, y.shape)
    break

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
model = GPT4(**hyperparams)
rngs = jax.random.PRNGKey(0)
rngs, dropout_rng = jax.random.split(rngs)
params = model.init({'params': rngs, 'dropout': dropout_rng}, dummy_inputs)['params']

# Call as you would a Jax/Flax model
outputs = model.apply({'params': params}, 
                      dummy_inputs, 
                      rngs={'dropout': dropout_rng})
print(outputs.shape)

# Training on data
trainer = GPTDataParallelTrainer(model, dummy_inputs.shape, 'params.pkl')
trainer.train(train_loader=dataloader, 
              num_epochs=2, 
              val_loader=dataloader)

print(trainer.evaluate(dataloader))

# Generating from a start token
start_tokens = jnp.array([[123, 456]])

# Remember to load the trained parameters 
params = trainer.load_params('params.pkl')
outputs = model.apply({'params': params},
                      start_tokens,
                      rngs={'dropout': jax.random.PRNGKey(2)}, 
                      method=model.generate)
print(outputs) 
```

Creating a Diffusion model
----------------

```py
import jax
import jax.numpy as jnp
from nanodl import ArrayDataset, DataLoader
from nanodl import DiffusionModel, DiffusionDataParallelTrainer

image_size = 32
block_depth = 2
batch_size = 8
widths = [32, 64, 128]
key = jax.random.PRNGKey(0)
input_shape = (101, image_size, image_size, 3)
images = jax.random.normal(key, input_shape)

# Use your own images
dataset = ArrayDataset(images) 
dataloader = DataLoader(dataset, 
                        batch_size=batch_size, 
                        shuffle=True, 
                        drop_last=False) 

# Create diffusion model
diffusion_model = DiffusionModel(image_size, widths, block_depth)
params = diffusion_model.init(key, images)
pred_noises, pred_images = diffusion_model.apply(params, images)
print(pred_noises.shape, pred_images.shape)

# Training on your data
# Note: saved params are often different from training weights, use the saved params for generation
trainer = DiffusionDataParallelTrainer(diffusion_model, 
                                       input_shape=images.shape, 
                                       weights_filename='params.pkl', 
                                       learning_rate=1e-4)
trainer.train(dataloader, 10, dataloader)
print(trainer.evaluate(dataloader))

# Generate some samples
params = trainer.load_params('params.pkl')
generated_images = diffusion_model.apply({'params': params}, 
                                         num_images=5, 
                                         diffusion_steps=5, 
                                         method=diffusion_model.generate)
print(generated_images.shape)
```

Creating a Whisper TTS model
----------------

```py
import jax
import jax.numpy as jnp
from nanodl import ArrayDataset, DataLoader
from nanodl import Whisper, WhisperDataParallelTrainer

# Dummy data parameters
batch_size = 8
max_length = 50
embed_dim = 256 
vocab_size = 1000 

# Generate data: replace with actual tokenised/quantised data
dummy_targets = jnp.ones((101, max_length), dtype=jnp.int32)
dummy_inputs = jnp.ones((101, max_length, embed_dim))

dataset = ArrayDataset(dummy_inputs, 
                       dummy_targets)

dataloader = DataLoader(dataset, 
                        batch_size=batch_size, 
                        shuffle=True, 
                        drop_last=False)

# How to loop through dataloader
for batch in dataloader:
    x, y = batch
    print(x.shape, y.shape)
    break

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
trainer = WhisperDataParallelTrainer(model, 
                                     dummy_inputs.shape, 
                                     dummy_targets.shape, 
                                     'params.pkl')
trainer.train(dataloader, 2, dataloader)

# Sample inference
params = trainer.load_params('params.pkl')

# for more than one sample, use model.generate_batch
transcripts = model.apply({'params': params}, 
                          dummy_inputs[:1], 
                          rngs=rngs, 
                          method=model.generate)

print(transcripts)
```

Creating an Accelerated PCA
----------------

```py
import jax
from nanodl import PCA

data = jax.random.normal(jax.random.key(0), (1000, 10))
pca = PCA(n_components=2)
pca.fit(data)
transformed_data = pca.transform(data)
original_data = pca.inverse_transform(transformed_data)
X_sampled = pca.sample(n_samples=1000, key=None)
print(X_sampled.shape, original_data.shape, transformed_data.shape)
```
GPU/TPU-accelerated versions of many models on SKLearn like NaiveBayesClassifier, Linear Regression, KMeans are available on NanoDL.

Using an individual module
----------------

Each contituent layer can be used in your own model.

```py
from nanodl import GraphAttetnionLayer

class GAT(nn.Module):
    nfeat: int
    nhid: int
    nclass: int
    dropout_rate: float
    alpha: float
    nheads: int

    @nn.compact
    def __call__(self, 
                 x: jnp.ndarray, 
                 adj: jnp.ndarray, 
                 training: bool = False) -> jnp.ndarray:
        heads = [GraphAttentionLayer(self.nfeat, 
                                     self.nhid, 
                                     dropout_rate=self.dropout_rate, 
                                     alpha=self.alpha, concat=True) for _ in range(self.nheads)]
        
        x = jnp.concatenate([head(x, adj, training) for head in heads], axis=1)
        x = nn.Dropout(rate=self.dropout_rate, 
                       deterministic=not training)(x)

        out_att = GraphAttentionLayer(self.nhid * self.nheads, 
                                      self.nclass, 
                                      dropout_rate=self.dropout_rate, 
                                      alpha=self.alpha, concat=False)
        
        return out_att(x, adj, training)
```

With this, you could for example create a transformer model with T5 Encoder and LlaMa2 Decoder!