""" 
Implementations of numerical methods
Jax can easily take the with just jax.grad(function)"""

import jax
import math
import jax.numpy as jnp

def derivative(f, x, h=0.0001):
    """
    Calculate the numerical derivative of a function at a specific point.

    Args:
        f (callable): The function for which to calculate the derivative.
        x (float): The point at which to calculate the derivative.
        h (float, optional): The step size for numerical differentiation.

    Returns:
        float: The numerical derivative of the function at the given point.
    """
    return (f(x + h) - f(x)) / h

def integral(f, a, b, h=0.0001):
    """
    Calculate the numerical integral of a function over a specified interval.

    Args:
        f (callable): The function to be integrated.
        a (float): The lower limit of the integration interval.
        b (float): The upper limit of the integration interval.
        h (float, optional): The step size for numerical integration.

    Returns:
        float: The numerical integral of the function over the specified interval.
    """
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


def maclaurin_expansion(x, n_terms=10):
    """
    Calculate the Maclaurin series expansion of the exponential function at a point x.

    Args:
        x (float): The point at which to evaluate the Maclaurin series.
        n_terms (int, optional): The number of terms in the expansion.

    Returns:
        float: The value of the Maclaurin series expansion at the given point.
    """
    result = 0.0
    for n in range(n_terms):
        result += (x ** n) / math.factorial(n)
    return result


def eigen_decomposition_from_scratch(matrix, num_iterations=100):
    """
    Perform eigen decomposition on a square matrix using power iteration.

    Args:
        matrix (ndarray): Square matrix to decompose.
        num_iterations (int, optional): Number of power iterations for each eigenvector.

    Returns:
        ndarray: Matrix containing eigenvectors as columns.
        ndarray: Array of eigenvalues.
    """
    def power_iteration(A, num_iterations):
        n = A.shape[0]
        x = jnp.ones(n)
        for _ in range(num_iterations):
            x = jnp.dot(A, x)
            x /= jnp.linalg.norm(x)
        return x

    def deflate(A, eigenvector, eigenvalue):
        return A - eigenvalue * jnp.outer(eigenvector, eigenvector)

    n = matrix.shape[0]
    eigenvectors = []
    eigenvalues = []

    for _ in range(n):
        eigenvector = power_iteration(matrix, num_iterations)
        eigenvalue = jnp.dot(eigenvector, jnp.dot(matrix, eigenvector))
        
        eigenvectors.append(eigenvector)
        eigenvalues.append(eigenvalue)

        matrix = deflate(matrix, eigenvector, eigenvalue)

    return jnp.column_stack(eigenvectors), jnp.array(eigenvalues)


def svd_jacobi(A, tol=1e-6, max_iter=1000):
    """
    Compute Singular Value Decomposition (SVD) of a matrix using the Jacobi method.

    Args:
        A (ndarray): Input matrix for SVD.
        tol (float, optional): Tolerance for convergence.
        max_iter (int, optional): Maximum number of iterations.

    Returns:
        ndarray: U matrix in SVD.
        ndarray: Singular values.
        ndarray: V transpose matrix in SVD.
    """
    m, n = A.shape
    U = jnp.eye(m)
    V = jnp.eye(n)

    converged = False
    num_iter = 0

    while not converged and num_iter < max_iter:
        converged = True

        for i in range(m - 1):
            for j in range(i + 1, m):
                B = A.at[i, i].set(A[i, i] - A[j, j]).at[j, j].set(A[j, j] - A[i, i])\
                     .at[i, j].set(A[i, j] + A[j, i]).at[j, i].set(A[i, j] + A[j, i])

                norm_B = jnp.linalg.norm(B)
                if norm_B > tol:
                    Q, _ = jnp.linalg.qr(B)
                    A = jnp.dot(Q.T, A).dot(Q)
                    U = jnp.dot(U, Q)
                    V = jnp.dot(V, Q)
                    converged = False

        num_iter += 1

    singular_values = jnp.sqrt(jnp.diag(A))
    return U, singular_values, V.T


def low_rank_factorization(X, rank, num_iterations=100, learning_rate=0.1):
    """
    Perform low rank factorization on a given matrix X.

    Args:
        X (ndarray): Input matrix to factorize.
        rank (int): Rank of the factorized matrices.
        num_iterations (int, optional): Number of optimization iterations.
        learning_rate (float, optional): Learning rate for optimization.

    Returns:
        ndarray: Factorized matrix U.
        ndarray: Factorized matrix V.
    """
    m, n = X.shape
    U = jax.random.normal(jax.random.PRNGKey(0), (m, rank))
    V = jax.random.normal(jax.random.PRNGKey(1), (n, rank))

    for _ in range(num_iterations):
        UV = jnp.dot(U, V.T)
        error = X - UV
        grad_U = -jnp.dot(error, V)
        grad_V = -jnp.dot(error.T, U)
        U -= learning_rate * grad_U
        V -= learning_rate * grad_V

    return U, V


def monte_carlo_pi(num_samples):
    """
    Estimate the value of π using Monte Carlo simulation.

    Args:
        num_samples (int): Number of random samples.

    Returns:
        float: Estimated value of π.
        
    Note: can be incredibly slow
    """
    key = jax.random.PRNGKey(0)
    points_inside_circle = 0

    for _ in range(num_samples):
        x = jax.random.uniform(key)
        y = jax.random.uniform(key)
        distance = x**2 + y**2
        if distance <= 1.0:
            points_inside_circle += 1

    estimated_pi = 4.0 * points_inside_circle / num_samples
    return estimated_pi