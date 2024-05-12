import time
from typing import Any, Tuple, Union

import jax
import jax.numpy as jnp
from jax import random


def time_rng_key(seed=None) -> jnp.ndarray:
    """Generate a JAX random key based on the current UNIX timestamp.

    Returns:
        jnp.ndarray: A JAX random key.
    """
    key = int(time.time()) if seed is None else seed
    return random.PRNGKey(key)


def uniform(
    shape: Tuple[int, ...],
    minval: Any = 0.0,
    maxval: Any = 1.0,
    seed=None,
    dtype: Any = jnp.float32,
) -> jnp.ndarray:
    """Generate a tensor of uniform random values.

    Args:
        shape (Tuple[int, ...]): The shape of the output tensor.
        dtype (Any, optional): The data type of the output tensor. Defaults to jnp.float32.
        minval (Any, optional): The lower bound of the uniform distribution. Defaults to 0.0.
        maxval (Any, optional): The upper bound of the uniform distribution. Defaults to 1.0.

    Returns:
        jnp.ndarray: A tensor of uniform random values.
    """
    return random.uniform(
        time_rng_key(seed), shape, dtype=dtype, minval=minval, maxval=maxval
    )


def normal(shape: Tuple[int, ...], dtype: Any = jnp.float32, seed=None) -> jnp.ndarray:
    """Generate a tensor of normal random values.

    Args:
        shape (Tuple[int, ...]): The shape of the output tensor.
        dtype (Any, optional): The data type of the output tensor. Defaults to jnp.float32.

    Returns:
        jnp.ndarray: A tensor of normal random values.
    """
    return random.normal(time_rng_key(seed), shape, dtype=dtype)


def bernoulli(p: float, shape: Tuple[int, ...] = (), seed=None) -> jnp.ndarray:
    """Generate random boolean values with a given probability.

    Args:
        p (float): Probability of sampling a True value.
        shape (Tuple[int, ...], optional): The shape of the output tensor. Defaults to ().

    Returns:
        jnp.ndarray: A tensor of boolean values.
    """
    return random.bernoulli(time_rng_key(seed), p, shape)


def categorical(
    logits: jnp.ndarray, axis: int = -1, shape: Tuple[int, ...] = (), seed=None
) -> jnp.ndarray:
    """Draw samples from a categorical distribution.

    Args:
        logits (jnp.ndarray): The unnormalized log probabilities of the categories.
        axis (int, optional): The axis along which the categorical distribution is applied. Defaults to -1.
        shape (Tuple[int, ...], optional): The shape of the output tensor. Defaults to ().

    Returns:
        jnp.ndarray: The sampled indices with the specified shape.
    """
    return random.categorical(time_rng_key(seed), logits, axis=axis, shape=shape)


def randint(
    shape: Tuple[int, ...], minval: int, maxval: int, dtype: str = "int32", seed=None
) -> jnp.ndarray:
    """Generate random integers between minval (inclusive) and maxval (exclusive).

    Args:
        shape (Tuple[int, ...]): The shape of the output tensor.
        minval (int): The lower bound of the random integers, inclusive.
        maxval (int): The upper bound of the random integers, exclusive.
        dtype (str, optional): The data type of the output tensor. Defaults to 'int32'.

    Returns:
        jnp.ndarray: A tensor of random integers.
    """
    return random.randint(time_rng_key(seed), shape, minval, maxval, dtype=dtype)


def permutation(x: Union[int, jnp.ndarray], axis: int = 0, seed=None) -> jnp.ndarray:
    """Randomly permute a sequence, or return a permuted range.

    Args:
        x (Union[int, jnp.ndarray]): If x is an integer, permute range(x). If x is an array, permute its elements.
        axis (int, optional): The axis along which to permute if x is an array. Defaults to 0.

    Returns:
        jnp.ndarray: The permuted sequence or array.
    """
    if isinstance(x, int):
        arr = jax.numpy.arange(x)
        return random.permutation(time_rng_key(seed), arr, axis=axis)
    else:
        return random.permutation(time_rng_key(seed), x, axis=axis)


def gumbel(shape: Tuple[int, ...], dtype: Any = jnp.float32, seed=None) -> jnp.ndarray:
    """Draw samples from a Gumbel distribution.

    Args:
        shape (Tuple[int, ...]): The shape of the output tensor.
        dtype (Any, optional): The data type of the output tensor. Defaults to jnp.float32.

    Returns:
        jnp.ndarray: A tensor of samples from a Gumbel distribution.
    """
    return random.gumbel(time_rng_key(seed), shape, dtype=dtype)


def choice(
    a: Union[int, jnp.ndarray],
    shape: Tuple[int, ...] = (),
    replace: bool = True,
    p: Union[None, jnp.ndarray] = None,
    axis: int = 0,
    seed=None,
) -> jnp.ndarray:
    """Randomly choose elements from a given 1-D array.

    Args:
        a (Union[int, jnp.ndarray]): If an int, the random sample is generated as if a were jnp.arange(a).
        shape (Tuple[int, ...], optional): The shape of the output tensor. Defaults to ().
        replace (bool, optional): Whether the sample is with or without replacement. Defaults to True.
        p (Union[None, jnp.ndarray], optional): The probabilities associated with each entry in a. Defaults to None.
        axis (int, optional): The axis along which to choose if a is an array. Defaults to 0.

    Returns:
        jnp.ndarray: The randomly chosen elements.
    """
    if isinstance(a, int):
        a = jnp.arange(a)
    return random.choice(
        time_rng_key(seed), a, shape=shape, replace=replace, p=p, axis=axis
    )


def bits(shape: Tuple[int, ...], dtype: Any = jnp.uint32, seed=None) -> jnp.ndarray:
    """Generate random bits.

    Args:
        shape (Tuple[int, ...]): The shape of the output tensor.
        dtype (Any, optional): The data type of the output tensor, typically an unsigned integer type. Defaults to jnp.uint32.

    Returns:
        jnp.ndarray: A tensor of random bits.
    """
    return random.bits(time_rng_key(seed), shape, dtype=dtype)


def exponential(
    shape: Tuple[int, ...], dtype: Any = jnp.float32, seed=None
) -> jnp.ndarray:
    """Draw samples from an exponential distribution.

    Args:
        shape (Tuple[int, ...]): The shape of the output tensor.
        dtype (Any, optional): The data type of the output tensor. Defaults to jnp.float32.

    Returns:
        jnp.ndarray: A tensor of samples from an exponential distribution.
    """
    return random.exponential(time_rng_key(seed), shape, dtype=dtype)


def triangular(
    left: float, right: float, mode: float, shape: Tuple[int, ...] = (), seed=None
) -> jnp.ndarray:
    """Draw samples from a triangular distribution.

    Args:
        left (float): The lower limit of the distribution.
        right (float): The upper limit of the distribution.
        mode (float): The mode (peak) of the distribution.
        shape (Tuple[int, ...], optional): The shape of the output tensor. Defaults to ().

    Returns:
        jnp.ndarray: A tensor of samples from a triangular distribution.
    """
    return random.triangular(time_rng_key(seed), left, right, mode, shape)


def truncated_normal(
    lower: float,
    upper: float,
    shape: Tuple[int, ...] = (),
    dtype: Any = jnp.float32,
    seed=None,
) -> jnp.ndarray:
    """Draw samples from a truncated normal distribution.

    Args:
        lower (float): The lower bound of the distribution.
        upper (float): The upper bound of the distribution.
        shape (Tuple[int, ...], optional): The shape of the output tensor. Defaults to ().
        dtype (Any, optional): The data type of the output tensor. Defaults to jnp.float32.

    Returns:
        jnp.ndarray: A tensor of samples from a truncated normal distribution.
    """
    return random.truncated_normal(time_rng_key(seed), lower, upper, shape, dtype)


def poisson(
    lam: float, shape: Tuple[int, ...] = (), dtype: Any = jnp.int32, seed=None
) -> jnp.ndarray:
    """Draw samples from a Poisson distribution.

    Args:
        lam (float): The expectation of interval (lambda parameter).
        shape (Tuple[int, ...], optional): The shape of the output tensor. Defaults to ().
        dtype (Any, optional): The data type of the output tensor. Defaults to jnp.int32.

    Returns:
        jnp.ndarray: A tensor of samples from a Poisson distribution.
    """
    return random.poisson(time_rng_key(seed), lam, shape=shape, dtype=dtype)


def geometric(
    p: float, shape: Tuple[int, ...] = (), dtype: Any = jnp.int32, seed=None
) -> jnp.ndarray:
    """Draw samples from a geometric distribution.

    Args:
        p (float): The probability of success of an individual trial.
        shape (Tuple[int, ...], optional): The shape of the output tensor. Defaults to ().
        dtype (Any, optional): The data type of the output tensor. Defaults to jnp.int32.

    Returns:
        jnp.ndarray: A tensor of samples from a geometric distribution.
    """
    return random.geometric(time_rng_key(seed), p, shape=shape, dtype=dtype)


def gamma(
    a: float, shape: Tuple[int, ...] = (), dtype: Any = jnp.float32, seed=None
) -> jnp.ndarray:
    """Draw samples from a gamma distribution.

    Args:
        a (float): The shape parameter of the gamma distribution.
        shape (Tuple[int, ...], optional): The shape of the output tensor. Defaults to ().
        dtype (Any, optional): The data type of the output tensor. Defaults to jnp.float32.

    Returns:
        jnp.ndarray: A tensor of samples from a gamma distribution.
    """
    return random.gamma(time_rng_key(seed), a, shape=shape, dtype=dtype)


def chisquare(
    df: float, shape: Tuple[int, ...] = (), dtype: Any = jnp.float32, seed=None
) -> jnp.ndarray:
    """Draw samples from a chi-square distribution.

    Args:
        df (float): The degrees of freedom.
        shape (Tuple[int, ...], optional): The shape of the output tensor. Defaults to ().
        dtype (Any, optional): The data type of the output tensor. Defaults to jnp.float32.

    Returns:
        jnp.ndarray: A tensor of samples from a chi-square distribution.
    """
    return random.chisquare(time_rng_key(seed), df, shape=shape, dtype=dtype)
