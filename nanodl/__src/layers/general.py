import jax
import time
import jax.numpy as jnp
from jax import random

def dropout(x: jnp.ndarray, 
            rate: float, 
            training: bool = False) -> jnp.ndarray:
    """Apply dropout to input tensor.

    Args:
        x (jnp.ndarray): Input tensor.
        rate (float): Dropout rate, must be between 0 and 1.
        training (bool, optional): Whether to apply dropout. 
        If False, returns input tensor unchanged. Defaults to False.

    Raises:
        ValueError: If dropout rate is not in [0, 1).

    Returns:
        jnp.ndarray: Tensor after applying dropout.
    """
    if not training:
        return x

    if not 0 <= rate < 1:
        raise ValueError("Dropout rate must be in the range [0, 1).")
    
    if rate == 0:
        return x

    keep_prob = 1 - rate
    mask = jax.random.bernoulli(random.PRNGKey(int(time.time())), keep_prob, x.shape)
    return jax.lax.select(mask, x / keep_prob, jnp.zeros_like(x))