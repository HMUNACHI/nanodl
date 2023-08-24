"""Implementations of various activation functions as classes"""

from jax import jit
import flax.linen as nn
from functools import partial
from jax.numpy import exp, log, where

class Relu:
    @partial(jit, static_argnums=(0,))
    def __call__(self, x):
        return where(x > 0, x, 0.0)
    

class LeakyRelu:
    @partial(jit, static_argnums=(0,))
    def __call__(self, x, alpha=0.1):
        return where(x>0, x, alpha*x)
    

class ELU:
    @partial(jit, static_argnums=(0,))
    def __call__(self, x, alpha=0.1, lambda_=1.05):
        return where(x > 0, x, alpha * (exp(x) - 1))
    

class SELU:
    @partial(jit, static_argnums=(0,))
    def __call__(self, x, alpha=1.67, lambda_=1.05):
        return lambda_ * where(x > 0, x, alpha * exp(x) - alpha)


class TanH:
    @partial(jit, static_argnums=(0,))
    def __call__(self, x):
        return (exp(x) - exp(-x))/(exp(x) + exp(-x))
    

class Sigmoid:
    @partial(jit, static_argnums=(0,))
    def __call__(self, x):
        return 1 / (1 + exp(-x))
    

class Softplus:
    @partial(jit, static_argnums=(0,))
    def __call__(self, x):
        return log(1 + exp(x))


class Softmax:
    @partial(jit, static_argnums=(0,))
    def __call__(self, x):
        return exp(x) / exp(x).sum()
    

class Swish:
    def __init__(self):
        self.sigmoid = Sigmoid()

    @partial(jit, static_argnums=(0,))
    def __call__(self, x):
        return x * self.sigmoid(x)
    

class Mish:
    def __init__(self):
        self.tanH = TanH()

    @partial(jit, static_argnums=(0,))
    def __call__(self, x):
        return x * self.tanH(log(1 + exp(x)))


class GELU:
    def __init__(self):
        self.tanH = TanH()

    @partial(jit, static_argnums=(0,))
    def __call__(self, x):
        tanh_res = self.tanH(0.7978845608 * (x + 0.044715 * x**3))
        return 0.5 * x * (1 + tanh_res)


class GEGLU(nn.Module):
    """
    Gated GLU (Gated Linear Unit).

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