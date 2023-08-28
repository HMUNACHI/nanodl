"""
Implementations of various activation functions as classes
"""

from jax import jit
import flax.linen as nn
import jax.numpy as jnp
from functools import partial

class Relu:
    """
    Rectified Linear Unit (ReLU) activation function.
    ReLU(x) = max(0, x)
    https://arxiv.org/abs/1502.01852

    Args:
        x (jax.numpy.ndarray): Input tensor.

    Returns:
        jax.numpy.ndarray: Output tensor after applying ReLU activation.
    """
    @partial(jit, static_argnums=(0,))
    def __call__(self, x):
        return jnp.where(x > 0, x, 0.0)
    

class LeakyRelu:
    """
    Leaky Rectified Linear Unit (Leaky ReLU) activation function.
    LeakyReLU(x) = x if x > 0, else alpha * x, where alpha is a small constant.
    https://ai.stanford.edu/~amaas/papers/relu_hybrid_icml2013_final.pdf

    Args:
        x (jax.numpy.ndarray): Input tensor.
        alpha (float, optional): Slope for negative values. Defaults to 0.1.

    Returns:
        jax.numpy.ndarray: Output tensor after applying Leaky ReLU activation.
    """
    @partial(jit, static_argnums=(0,))
    def __call__(self, x, alpha=0.1):
        return jnp.where(x > 0, x, alpha * x)
    

class ELU:
    """
    Exponential Linear Unit (ELU) activation function.
    ELU(x) = x if x > 0, else alpha * (exp(x) - 1), where alpha is a constant.
    https://arxiv.org/abs/1511.07289

    Args:
        x (jax.numpy.ndarray): Input tensor.
        alpha (float, optional): Slope for negative values. Defaults to 1.0.
        lambda_ (float, optional): Constant multiplier for the negative range. Defaults to 1.0.

    Returns:
        jax.numpy.ndarray: Output tensor after applying ELU activation.
    """
    @partial(jit, static_argnums=(0,))
    def __call__(self, x, alpha=1.0, lambda_=1.0):
        return jnp.where(x > 0, x, alpha * (jnp.exp(x) - 1))
    

class SELU:
    """
    Scaled Exponential Linear Unit (SELU) activation function.
    SELU(x) = lambda * x if x > 0, else lambda * (alpha * exp(x) - alpha),
    where lambda and alpha are constants.
    https://arxiv.org/abs/1706.02515

    Args:
        x (jax.numpy.ndarray): Input tensor.
        alpha (float, optional): Slope for negative values. Defaults to 1.67.
        lambda_ (float, optional): Constant multiplier for the positive range. Defaults to 1.05.

    Returns:
        jax.numpy.ndarray: Output tensor after applying SELU activation.
    """
    @partial(jit, static_argnums=(0,))
    def __call__(self, x, alpha=1.67, lambda_=1.05):
        return lambda_ * jnp.where(x > 0, x, alpha * jnp.exp(x) - alpha)


class TanH:
    """
    Hyperbolic Tangent (TanH) activation function.
    TanH(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))

    Args:
        x (jax.numpy.ndarray): Input tensor.

    Returns:
        jax.numpy.ndarray: Output tensor after applying TanH activation.
    """
    @partial(jit, static_argnums=(0,))
    def __call__(self, x):
        return (jnp.exp(x) - jnp.exp(-x)) / (jnp.exp(x) + jnp.exp(-x))
    

class Sigmoid:
    """
    Sigmoid activation function.
    Sigmoid(x) = 1 / (1 + exp(-x))

    Args:
        x (jax.numpy.ndarray): Input tensor.

    Returns:
        jax.numpy.ndarray: Output tensor after applying Sigmoid activation.
    """
    @partial(jit, static_argnums=(0,))
    def __call__(self, x):
        return 1 / (1 + jnp.exp(-x))
    

class Softplus:
    """
    Softplus activation function.
    Softplus(x) = log(1 + exp(x))

    Args:
        x (jax.numpy.ndarray): Input tensor.

    Returns:
        jax.numpy.ndarray: Output tensor after applying Softplus activation.
    """
    @partial(jit, static_argnums=(0,))
    def __call__(self, x):
        return jnp.log(1 + jnp.exp(x))


class Softmax:
    """
    Softmax activation function.
    Softmax(x) = exp(x) / sum(exp(x))
    http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.385.7422

    Args:
        x (jax.numpy.ndarray): Input tensor.

    Returns:
        jax.numpy.ndarray: Output tensor after applying Softmax activation.
    """
    @partial(jit, static_argnums=(0,))
    def __call__(self, x):
        return jnp.exp(x) / jnp.exp(x).sum()
    

class Swish:
    """
    Swish activation function.
    Swish(x) = x * Sigmoid(x)
    https://arxiv.org/abs/1710.05941

    Args:
        x (jax.numpy.ndarray): Input tensor.

    Returns:
        jax.numpy.ndarray: Output tensor after applying Swish activation.
    """
    def __init__(self):
        self.sigmoid = Sigmoid()

    @partial(jit, static_argnums=(0,))
    def __call__(self, x):
        return x * self.sigmoid(x)
    

class Mish:
    """
    Mish activation function.
    Mish(x) = x * tanh(log(1 + exp(x)))
    https://arxiv.org/abs/1908.08681

    Args:
        x (jax.numpy.ndarray): Input tensor.

    Returns:
        jax.numpy.ndarray: Output tensor after applying Mish activation.
    """
    def __init__(self):
        self.tanH = TanH()

    @partial(jit, static_argnums=(0,))
    def __call__(self, x):
        return x * self.tanH(jnp.log(1 + jnp.exp(x)))


class GELU:
    """
    Gaussian Error Linear Unit (GELU) activation function.
    GELU(x) = 0.5 * x * (1 + tanh(0.7978845608 * (x + 0.044715 * x**3)))
    https://arxiv.org/abs/1606.08415

    Args:
        x (jax.numpy.ndarray): Input tensor.

    Returns:
        jax.numpy.ndarray: Output tensor after applying GELU activation.
    """
    def __init__(self):
        self.tanH = TanH()

    @partial(jit, static_argnums=(0,))
    def __call__(self, x):
        tanh_res = self.tanH(0.7978845608 * (x + 0.044715 * x**3))
        return 0.5 * x * (1 + tanh_res)


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