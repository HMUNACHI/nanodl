import jax
import math
import jax.numpy as jnp


def newton_raphson(f, initial_guesses, tol=1e-5, max_iter=1000):
    """
    Newton-Raphson method for finding multiple roots of a function using JAX.

    Args:
        f (function): Function for which to find the roots.
        initial_guesses (array): Array of initial guesses for the roots.
        tol (float, optional): Tolerance for convergence. Default is 1e-5.
        max_iter (int, optional): Maximum number of iterations. Default is 1000.

    Returns:
        list: Approximations of the roots.
    """
    roots = []
    df = jax.grad(f)  # Automatic differentiation to find the derivative

    for x0 in initial_guesses:
        x = x0
        for _ in range(max_iter):
            fx = f(x)
            dfx = df(x)
            if jnp.abs(fx) < tol:
                roots.append(x)
                break
            x = x - fx / dfx

    # Remove duplicates and sort roots
    roots = jnp.array(roots)
    roots = jnp.unique(jnp.around(roots, decimals=int(-jnp.log10(tol))))
    return roots.tolist()