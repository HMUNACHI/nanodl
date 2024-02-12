## Overview

Developing and training transformer-based models is typically resource-intensive and time-consuming and AI/ML experts frequently need to build smaller-scale versions of these models for specific problems. Jax, a low-resource yet powerful framework, accelerates the development of neural networks, but existing resources for transformer development in Jax are limited. NanoDL addresses this challenge with the following features:

- A wide array of blocks and layers, facilitating the creation of customised transformer models from scratch.
- An extensive selection of models like LlaMa2, Mistral, Mixtral, GPT3, GPT4 (inferred), T5, Whisper, ViT, Mixers, GAT, CLIP, and more, catering to a variety of tasks and applications.
- Data-parallel distributed trainers so developers can efficiently train large-scale models on multiple GPUs or TPUs, without the need for manual training loops.
- Dataloaders, making the process of data handling for Jax/Flax more straightforward and effective.
- Custom layers not found in Flax/Jax, such as RoPE, GQA, MQA, and SWin attention, allowing for more flexible model development.
- GPU/TPU-accelerated classical ML models like PCA, KMeans, Regression, Gaussian Processes etc., akin to SciKit Learn on GPU.
- Modular design so users can blend elements from various models, such as GPT, Mixtral, and LlaMa2, to craft unique hybrid transformer models.
- A range of advanced algorithms for NLP and computer vision tasks, such as Gaussian Blur, BLEU etc.
- Each model is contained in a single file with no external dependencies, so the source code can also be easily used. 

Feedback on any of our discussion, issue and pull request threads are welcomed! Please report any feature requests, issues, questions or concerns in the [discussion forum](https://github.com/hmunachi/nanodl/discussions), or just let us know what you're working on! In case you want to reach out directly, we're at ndubuakuhenry@gmail.com.

# Contribution

This is the first iteration of this project, roughness is expected, contributions are therefore highly encouraged! Follow the recommended steps:

- Raise the issue/discussion to get second opinions
- Fork the repository
- Create a branch
- Make your changes without ruining the design patterns
- Write tests for your changes if necessary
- Install locally with `pip install -e .`
- Run tests with `python -m unittest discover -s tests`
- Then submit a pull request from branch.

Contributions can be made in various forms:

- Writing documentation.
- Fixing bugs.
- Implementing papers.
- Writing high-coverage tests.
- OPtimizing existing codes.
- Experimenting and submitting real-world examples to the examples section.
- Reporting bugs.
- Responding to reported issues.

Coming features include:
- Reinforcement Learning With Human Feedback (RLHF).
- Tokenizers.
- Code optimisations.

To follow up or share thoughts, follow [here](https://forms.gle/vwveb9SKdPYywHx9A)

## Sponsorships

The name "NanoDL" stands for Nano Deep Learning. Models are exploding in size, therefore gate-keeping 
experts and companies with limited resources from building flexible models without prohibitive costs.
Following the success of Phi models, the long-term goal is to build and train nano versions of all available models,
while ensuring they compete with the original models in performance, with total 
number of parameters not exceeding 1B. Trained weights will be made available via this library.
Any form of sponsorship, funding, grants or contribution will help with training resources.
You can sponsor via the provided button, or reach out via ndubuakuhenry@gmail.com.