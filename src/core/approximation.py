""" 
Implementations of numerical methods
Jax can easily take the with just jax.grad(function)"""

import jax
import math
import jax.numpy as jnp

def derivative(f, x, h=0.0001):
    return (f(x+h) - f(x)) / h

def integral(f, a, b, h=0.0001):
    x = a
    y_prime = 0
    while x <= b:
        y_prime += h * f(x)
        x += h
    return y_prime


def newton_raphson(f, x0, max_iter=100, tol=1e-6):
    """
    Apply the Newton-Raphson method to find the root of a function.

    Args:
        f (function): The function for which to find the root.
        x0 (float): Initial guess for the root.
        max_iter (int, optional): Maximum number of iterations. Defaults to 100.
        tol (float, optional): Tolerance for convergence. The iteration stops when the
            absolute difference between consecutive x values is less than this value. Defaults to 1e-6.

    Returns:
        float: Approximated root of the function.
    """
    df_dx = jax.grad(f)  # Compute the derivative of f using JAX's autograd

    x = x0
    for _ in range(max_iter):
        x_new = x - f(x) / df_dx(x)
        if jnp.abs(x_new - x) < tol:
            return x_new
        x = x_new

    return x



def secant_method(f, x0, x1, max_iter=100, tol=1e-6):
    """
    Apply the Secant method to find the root of a function.

    Args:
        f (function): The function for which to find the root.
        x0 (float): First initial guess for the root.
        x1 (float): Second initial guess for the root.
        max_iter (int, optional): Maximum number of iterations. Defaults to 100.
        tol (float, optional): Tolerance for convergence. The iteration stops when the
            absolute difference between consecutive x values is less than this value. Defaults to 1e-6.

    Returns:
        float: Approximated root of the function.
    """
    df_dx = jax.grad(f)  # Compute the derivative of f using JAX's autograd

    x_curr = x1
    x_prev = x0
    for _ in range(max_iter):
        f_curr = f(x_curr)
        f_prev = f(x_prev)
        delta_x = (x_curr - x_prev) * f_curr / (f_curr - f_prev)
        x_new = x_curr - delta_x

        if jnp.abs(delta_x) < tol:
            return x_new

        x_prev = x_curr
        x_curr = x_new

    return x_curr



def taylor_series(f, x0, n_terms):
    """
    Compute the Taylor series expansion of a function at a given point.

    Args:
        f (function): The function to expand.
        x0 (float): The point around which to compute the expansion.
        n_terms (int): Number of terms in the Taylor series.

    Returns:
        function: A function representing the Taylor series expansion.
    """
    df_dx = jax.grad(f)  # Compute the derivative of f using JAX's autograd

    def taylor_term(x, term_idx):
        return df_dx(x0) * (x - x0)**term_idx / math.factorial(term_idx)

    def taylor_sum(x):
        return sum(taylor_term(x, i) for i in range(n_terms))

    return taylor_sum