<p align="center">
  <img src="assets/logo.jpg" alt="Alt text"/>
</p>

# A Jax-based library for designing and training transformer models from scratch.

![[License](https://img.shields.io/github/license/hmunachi/nanodl?style=flat-square) 
![Read the Docs](https://img.shields.io/readthedocs/nanodl)
![Discord](https://img.shields.io/discord/1222217369816928286)
![Stars](https://img.shields.io/github/stars/hmunachi/nanodl?style=social) 
![Forks](https://img.shields.io/github/forks/hmunachi/nanodl?style=social) 
![Issues](https://img.shields.io/github/issues/hmunachi/nanodl?style=flat-square) 
![LinkedIn](https://img.shields.io/badge/-LinkedIn-blue?style=flat-square&logo=linkedin&logoColor=white)(https://www.linkedin.com//company/80434055)
![Twitter](https://img.shields.io/twitter/follow/hmunachii?style=social)](https://twitter.com/hmunachii)


[**Overview**](#overview)
| [**Quick install**](#quick-install)
| [**What does NanoDL look like?**](#what-does-nanodl-look-like)
| [**Documentation**](https://nanodl.readthedocs.io/)

Author: [Henry Ndubuaku](https://www.linkedin.com/in/henry-ndubuaku-7b6350b8/)

## Overview
Developing and training transformer-based models is typically resource-intensive and time-consuming and AI/ML experts frequently need to build smaller-scale versions of these models for specific problems. Jax, a low-resource yet powerful framework, accelerates the development of neural networks, but existing resources for transformer development in Jax are limited. NanoDL addresses this challenge with the following features:

- A wide array of blocks and layers, facilitating the creation of customised transformer models from scratch.
- An extensive selection of models like Gemma, LlaMa2, Mistral, Mixtral, GPT3, GPT4 (inferred), T5, Whisper, ViT, Mixers, GAT, CLIP, and more, catering to a variety of tasks and applications.
- Data-parallel distributed trainers includding RLHF so developers can efficiently train large-scale models on multiple GPUs or TPUs, without the need for manual training loops.
- Dataloaders, making the process of data handling for Jax/Flax more straightforward and effective.
- Custom layers not found in Flax/Jax, such as RoPE, GQA, MQA, and SWin attention, allowing for more flexible model development.
- GPU/TPU-accelerated classical ML models like PCA, KMeans, Regression, Gaussian Processes etc., akin to SciKit Learn on GPU.
- Modular design so users can blend elements from various models, such as GPT, Mixtral, and LlaMa2, to craft unique hybrid transformer models.
- True random number generators in Jax which do not need the verbose code.
- A range of advanced algorithms for NLP and computer vision tasks, such as Gaussian Blur, BLEU, Tokenizer etc.
- Each model is contained in a single file with no external dependencies, so the source code can also be easily used. 
- True random number generators in Jax which do not need the verbose code (examples shown in next sections).

Feedback on any of our discussion, issue and pull request threads are welcomed! Please report any feature requests, issues, questions or concerns in the [discussion forum](https://github.com/hmunachi/nanodl/discussions), or just let us know what you're working on! In case you want to reach out directly, we're at ndubuakuhenry@gmail.com.


There are experimental features (like MAMBA architecture and RLHF) in the repo which are not available via the package, pending tests.

## Quick install

You will need Python 3.9 or later, and working [JAX](https://github.com/google/jax/blob/main/README.md)
installation, [FLAX](https://github.com/google/flax/blob/main/README.md)
installation, [OPTAX](https://github.com/google-deepmind/optax/blob/main/README.md)
installation (with GPU support for running training, without can only support creations).
Models can be designed and tested on CPUs but trainers are all Distributed Data-Parallel which would require a GPU with 1 to N GPUS/TPUS. For CPU-only version of JAX:

```
pip install --upgrade pip # To support manylinux2010 wheels.
pip install jax flax optax
```

Then, install nanodl from PyPi:

```
pip install nanodl
```

## What does nanodl look like?

We provide various example usages of the nanodl API.

```py
import jax
import jax.numpy as jnp
from nanodl import time_rng_key
from nanodl import ArrayDataset, DataLoader
from nanodl import GPT4, GPTDataParallelTrainer

# Generate dummy data
batch_size = 8
max_length = 10

# Replace with actual list of tokenised texts
data = jnp.ones((101, max_length), dtype=jnp.int32)

# Shift to create next-token prediction dataset
dummy_inputs, dummy_targets = data[:, :-1], data[:, 1:]

# Create dataset and dataloader
dataset = ArrayDataset(dummy_inputs, dummy_targets)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)

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

# Initialize inferred GPT4 model 
model = GPT4(**hyperparams)
params = model.init(
    {'params': time_rng_key(), 
     'dropout': time_rng_key()
     }, 
    dummy_inputs)['params']

# Training on data
trainer = GPTDataParallelTrainer(model, dummy_inputs.shape, 'params.pkl')
trainer.train(train_loader=dataloader, num_epochs=2, val_loader=dataloader)

# Generating from a start token
start_tokens = jnp.array([[123, 456]])

# Remember to load the trained parameters 
params = trainer.load_params('params.pkl')
outputs = model.apply({'params': params},
                      start_tokens,
                      rngs={'dropout': time_rng_key()}, 
                      method=model.generate)
```

Vision example

```py
import jax
import jax.numpy as jnp
from nanodl import time_rng_key
from nanodl import ArrayDataset, DataLoader
from nanodl import DiffusionModel, DiffusionDataParallelTrainer

image_size = 32
block_depth = 2
batch_size = 8
widths = [32, 64, 128]
input_shape = (101, image_size, image_size, 3)
images = jax.random.normal(time_rng_key(), input_shape)

# Use your own images
dataset = ArrayDataset(images) 
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False) 

# Create diffusion model
diffusion_model = DiffusionModel(image_size, widths, block_depth)
params = diffusion_model.init(key, images)
pred_noises, pred_images = diffusion_model.apply(params, images)
print(pred_noises.shape, pred_images.shape)

# Training on your data
trainer = DiffusionDataParallelTrainer(diffusion_model, 
                                       input_shape=images.shape, 
                                       weights_filename='params.pkl', 
                                       learning_rate=1e-4)
trainer.train(dataloader, 10, dataloader)

# Generate some samples
params = trainer.load_params('params.pkl')
generated_images = diffusion_model.apply({'params': params}, 
                                         num_images=5, 
                                         diffusion_steps=5, 
                                         method=diffusion_model.generate)
```

Audio example

```py
import jax
import jax.numpy as jnp
from nanodl import time_rng_key
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

dataset = ArrayDataset(dummy_inputs, dummy_targets)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)

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
rngs = {'params': time_rng_key(), 'dropout': time_rng_key()}
params = model.init(rngs, dummy_inputs, dummy_targets)['params']

# Training on your data
trainer = WhisperDataParallelTrainer(model, 
                                     dummy_inputs.shape, 
                                     dummy_targets.shape, 
                                     'params.pkl')
trainer.train(dataloader, 2, dataloader)

# Sample inference
params = trainer.load_params('params.pkl')

# for more than one sample, often use model.generate_batch
transcripts = model.apply({'params': params}, 
                          dummy_inputs[:1], 
                          rngs=rngs, 
                          method=model.generate)
```

Reward Model example for RLHF

```py
import jax
import jax.numpy as jnp
from nanodl import time_rng_key
from nanodl import ArrayDataset, DataLoader
from nanodl import Mistral, RewardModel, RewardDataParallelTrainer

# Generate dummy data
batch_size = 8
max_length = 10

# Replace with actual tokenised data
dummy_chosen = jnp.ones((101, max_length), dtype=jnp.int32)
dummy_rejected = jnp.zeros((101, max_length), dtype=jnp.int32)

# Create dataset and dataloader
dataset = ArrayDataset(dummy_chosen, dummy_rejected)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)

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
    'window_size': 5,
    'shift_size': 2
}

# Initialize reward model from Mistral
model = Mistral(**hyperparams)
reward_model = RewardModel(model, dim=hyperparams['hidden_dim'], dropout=0.1)

# Train the reward model
trainer = RewardDataParallelTrainer(reward_model, dummy_chosen.shape, 'reward_model_weights.pkl')
trainer.train(dataloader, 5, dataloader)
params = trainer.load_params('reward_model_weights.pkl')

# Call as you would a regular Flax model
rewards = reward_model.apply({'params': params}, 
                    dummy_chosen, 
                    rngs={'dropout': time_rng_key()})
```

PCA example

```py
import jax
from nanodl import PCA

# Use actual data
data = jax.random.normal(jax.random.key(0), (1000, 10))

# Initialise and train PCA model
pca = PCA(n_components=2)
pca.fit(data)

# Get PCA transforms
transformed_data = pca.transform(data)

# Get reverse transforms
original_data = pca.inverse_transform(transformed_data)

# Sample from the distribution
X_sampled = pca.sample(n_samples=1000, key=None)
```

NanoDL provides random module which abstracts away Jax's intricacies.
It generates truly random variables by using the current timestamp as seed.

```py
import jax 

# Jax example
key = jax.random.PRNGKey(0) 
jax_array = jax.random.uniform(key, shape=(3, 3))

# NanoDL example
jax_array = nanodl.uniform(shape=(3, 3))

# For reproducability, use seed
jax_array = nanodl.uniform(shape=(3, 3), seed=0)
```

This is the first iteration of this project, roughness is expected, and contributions are therefore highly encouraged! 

- Make your changes without changing the design patterns
- Write tests for your changes if necessary
- Install locally with `pip install -e .`
- Run tests with `python -m unittest discover -s tests`
- Then submit a pull request.

Contributions can be made in various forms:

- Writing documentation.
- Fixing bugs.
- Implementing papers.
- Writing high-coverage tests.
- Optimizing existing codes.
- Experimenting and submitting real-world examples to the examples section.
- Reporting bugs.
- Responding to reported issues.

To follow up or share thoughts, follow [here](https://forms.gle/vwveb9SKdPYywHx9A)

## Sponsorships

The name "NanoDL" stands for Nano Deep Learning. Models are exploding in size, therefore gate-keeping 
experts and companies with limited resources from building flexible models without prohibitive costs.
Following the success of Phi models, the long-term goal is to build and train nano versions of all available models,
while ensuring they compete with the original models in performance, with total 
number of parameters not exceeding 1B. Trained weights will be made available via this library.
Any form of sponsorship, funding, grants or contribution will help with training resources.
You can sponsor via the user profile tag or reach out via ndubuakuhenry@gmail.com.

## Citing nanodl

To cite this repository:

```
@software{nanodl2024github,
  author = {Henry Ndubuaku},
  title = {NanoDL: A Jax-based library for designing and training transformer models from scratch.},
  url = {http://github.com/hmunachi/nanodl},
  version = {1.0.1dev},
  year = {2024},
}
```
