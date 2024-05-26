__version__ = "1.2.5.dev1"

from nanodl.__src.classical.bayes import NaiveBayesClassifier
from nanodl.__src.classical.clustering import GaussianMixtureModel, KMeans
from nanodl.__src.classical.dimensionality_reduction import PCA
from nanodl.__src.classical.regression import (
    GaussianProcess,
    LinearRegression,
    LogisticRegression,
)
from nanodl.__src.experimental.gat import GAT, GraphAttentionLayer
from nanodl.__src.models.attention import (
    GatedMultiHeadAttention,
    HierarchicalMultiHeadAttention,
    LocalMultiHeadAttention,
    MultiQueryAttention,
    RotaryMultiHeadAttention,
)
from nanodl.__src.models.clip import (
    CLIP,
    CLIPDataParallelTrainer,
    ImageEncoder,
    SelfMultiHeadAttention,
    TextEncoder,
)
from nanodl.__src.models.diffusion import (
    DiffusionDataParallelTrainer,
    DiffusionModel,
    UNet,
    UNetDownBlock,
    UNetResidualBlock,
    UNetUpBlock,
)
from nanodl.__src.models.gemma import (
    Gemma,
    GemmaDataParallelTrainer,
    GemmaDecoder,
    GemmaDecoderBlock,
)
from nanodl.__src.models.gpt import (
    GPT3,
    GPT4,
    GPT3Block,
    GPT3Decoder,
    GPT4Block,
    GPT4Decoder,
    GPTDataParallelTrainer,
    PositionWiseFFN,
)
from nanodl.__src.models.ijepa import IJEPA, IJEPADataParallelTrainer, IJEPADataSampler
from nanodl.__src.models.lamda import (
    LaMDA,
    LaMDABlock,
    LaMDADataParallelTrainer,
    LaMDADecoder,
    RelativeMultiHeadAttention,
)
from nanodl.__src.models.llama import (
    GroupedRotaryMultiHeadAttention,
    Llama3,
    Llama3Decoder,
    Llama3DecoderBlock,
    LlamaDataParallelTrainer,
    RotaryPositionalEncoding,
)
from nanodl.__src.models.mistral import (
    GroupedRotaryShiftedWindowMultiHeadAttention,
    Mistral,
    MistralDataParallelTrainer,
    MistralDecoder,
    MistralDecoderBlock,
    Mixtral,
    MixtralDecoder,
    MixtralDecoderBlock,
)
from nanodl.__src.models.mixer import (
    Mixer,
    MixerBlock,
    MixerDataParallelTrainer,
    MixerEncoder,
)
from nanodl.__src.models.reward import RewardDataParallelTrainer, RewardModel
from nanodl.__src.models.t5 import (
    T5,
    T5DataParallelTrainer,
    T5Decoder,
    T5DecoderBlock,
    T5Encoder,
    T5EncoderBlock,
)
from nanodl.__src.models.transformer import (
    AddNorm,
    MultiHeadAttention,
    PositionalEncoding,
    PositionWiseFFN,
    TokenAndPositionEmbedding,
    Transformer,
    TransformerDataParallelTrainer,
    TransformerDecoderBlock,
    TransformerEncoder,
)
from nanodl.__src.models.vit import (
    PatchEmbedding,
    ViT,
    ViTBlock,
    ViTDataParallelTrainer,
    ViTEncoder,
)
from nanodl.__src.models.whisper import (
    Whisper,
    WhisperDataParallelTrainer,
    WhisperSpeechEncoder,
    WhisperSpeechEncoderBlock,
)
from nanodl.__src.utils.data import ArrayDataset, DataLoader, Dataset
from nanodl.__src.utils.ml import (
    batch_cosine_similarities,
    batch_pearsonr,
    classification_scores,
    count_parameters,
    entropy,
    gini_impurity,
    hamming,
    jaccard,
    kl_divergence,
    mean_reciprocal_rank,
    zero_pad_sequences,
)
from nanodl.__src.utils.nlp import (
    bleu,
    cider_score,
    meteor,
    perplexity,
    rouge,
    word_error_rate,
)
from nanodl.__src.utils.random import *
from nanodl.__src.utils.vision import (
    adjust_brightness,
    adjust_contrast,
    flip_image,
    gaussian_blur,
    normalize_images,
    random_crop,
    random_flip_image,
    sobel_edge_detection,
)

__all__ = [
    # Sklearn GPU
    "NaiveBayesClassifier",
    "PCA",
    "KMeans",
    "GaussianMixtureModel",
    "LinearRegression",
    "LogisticRegression",
    "GaussianProcess",
    # Models
    "IJEPA",
    "IJEPADataParallelTrainer",
    "IJEPADataSampler",
    "Gemma",
    "GemmaDataParallelTrainer",
    "GemmaDecoder",
    "GemmaDecoderBlock",
    "GAT",
    "GraphAttentionLayer",
    "T5",
    "T5DataParallelTrainer",
    "T5Encoder",
    "T5Decoder",
    "T5EncoderBlock",
    "T5DecoderBlock",
    "ViT",
    "ViTDataParallelTrainer",
    "ViTBlock",
    "ViTEncoder",
    "PatchEmbedding",
    "CLIP",
    "CLIPDataParallelTrainer",
    "ImageEncoder",
    "TextEncoder",
    "SelfMultiHeadAttention",
    "LaMDA",
    "LaMDADataParallelTrainer",
    "LaMDABlock",
    "LaMDADecoder",
    "RelativeMultiHeadAttention",
    "Mixer",
    "MixerDataParallelTrainer",
    "MixerBlock",
    "MixerEncoder",
    "Llama3",
    "LlamaDataParallelTrainer",
    "RotaryPositionalEncoding",
    "Llama3Decoder",
    "Llama3DecoderBlock",
    "GroupedRotaryMultiHeadAttention",
    "GPT3",
    "GPT4",
    "GPTDataParallelTrainer",
    "GPT3Block",
    "GPT4Block",
    "GPT3Decoder",
    "GPT4Decoder",
    "PositionWiseFFN",
    "Mistral",
    "MistralDataParallelTrainer",
    "MistralDecoder",
    "MistralDecoderBlock",
    "GroupedRotaryShiftedWindowMultiHeadAttention",
    "Mixtral",
    "MixtralDecoder",
    "MixtralDecoderBlock",
    "Whisper",
    "WhisperDataParallelTrainer",
    "WhisperSpeechEncoder",
    "WhisperSpeechEncoderBlock",
    "RewardModel",
    "RewardDataParallelTrainer",
    "DiffusionModel",
    "DiffusionDataParallelTrainer",
    "UNet",
    "UNetDownBlock",
    "UNetUpBlock",
    "UNetResidualBlock",
    "Transformer",
    "TransformerDataParallelTrainer",
    "TransformerEncoder",
    "TransformerDecoderBlock",
    "PositionalEncoding",
    "PositionWiseFFN",
    "TokenAndPositionEmbedding",
    "MultiHeadAttention",
    "AddNorm",
    # Utilities
    "Dataset",
    "ArrayDataset",
    "DataLoader",
    "batch_cosine_similarities",
    "batch_pearsonr",
    "classification_scores",
    "count_parameters",
    "entropy",
    "gini_impurity",
    "hamming",
    "jaccard",
    "kl_divergence",
    "mean_reciprocal_rank",
    "zero_pad_sequences",
    "bleu",
    "cider_score",
    "meteor",
    "perplexity",
    "rouge",
    "word_error_rate",
    "adjust_brightness",
    "adjust_contrast",
    "flip_image",
    "gaussian_blur",
    "normalize_images",
    "random_crop",
    "random_flip_image",
    "sobel_edge_detection",
    "MultiQueryAttention",
    "LocalMultiHeadAttention",
    "HierarchicalMultiHeadAttention",
    "GatedMultiHeadAttention",
    "RotaryMultiHeadAttention",
    # Random
    "time_rng_key",
    "uniform",
    "normal",
    "bernoulli",
    "categorical",
    "randint",
    "permutation",
    "gumbel",
    "choice",
    "bits",
    "exponential",
    "triangular",
    "truncated_normal",
    "poisson",
    "geometric",
    "gamma",
    "chisquare",
]

import importlib
import sys


def check_library_installed(lib_name):
    try:
        return importlib.import_module(lib_name)
    except ImportError:
        raise ImportError(f"{lib_name} is not installed or improperly installed.")


def test_flax(flax):
    model = flax.linen.Dense(features=10)


def test_jax(jax):
    arr = jax.numpy.array([1, 2, 3])
    result = jax.numpy.sum(arr)


def test_optax(optax):
    optimizer = optax.sgd(learning_rate=0.1)


def test_einops(einops):
    arr = einops.rearrange([1, 2, 3], "a b c -> b a c")


def main():
    try:
        flax = check_library_installed("flax")
        jax = check_library_installed("jax")
        optax = check_library_installed("optax")
        einops = check_library_installed("einops")

        test_flax(flax)
        test_jax(jax)
        test_optax(optax)

    except ImportError as e:
        print(e)
        sys.exit(1)
    except Exception as e:
        print(f"An error occurred while verifying Jax/Flax/Optax installation: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
