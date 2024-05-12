import unittest

import jax.numpy as jnp

from nanodl import (
    bernoulli,
    bits,
    categorical,
    chisquare,
    choice,
    exponential,
    gamma,
    geometric,
    gumbel,
    normal,
    permutation,
    poisson,
    randint,
    time_rng_key,
    triangular,
    truncated_normal,
    uniform,
)


class TestRandomFunctions(unittest.TestCase):

    def test_time_rng_key(self):
        key1 = time_rng_key(seed=42)
        key2 = time_rng_key(seed=42)
        self.assertTrue(
            jnp.array_equal(key1, key2), "Keys should be equal for the same seed"
        )

    def test_uniform(self):
        result = uniform((2, 3))
        self.assertEqual(result.shape, (2, 3))
        self.assertEqual(result.dtype, jnp.float32)

    def test_normal(self):
        result = normal((4, 5), seed=42)
        self.assertEqual(result.shape, (4, 5))
        self.assertEqual(result.dtype, jnp.float32)

    def test_bernoulli(self):
        result = bernoulli(0.5, (10,), seed=42)
        self.assertEqual(result.shape, (10,))
        self.assertEqual(result.dtype, jnp.bool_)

    def test_categorical(self):
        logits = jnp.array([0.1, 0.2, 0.7])
        result = categorical(logits, shape=(5,), seed=42)
        self.assertEqual(result.shape, (5,))

    def test_randint(self):
        result = randint((3, 3), 0, 10, seed=42)
        self.assertEqual(result.shape, (3, 3))
        self.assertEqual(result.dtype, jnp.int32)

    def test_permutation(self):
        arr = jnp.arange(10)
        result = permutation(arr, seed=42)
        self.assertEqual(result.shape, arr.shape)
        self.assertNotEqual(jnp.all(result == arr), True)

    def test_gumbel(self):
        result = gumbel((2, 2), seed=42)
        self.assertEqual(result.shape, (2, 2))
        self.assertEqual(result.dtype, jnp.float32)

    def test_choice(self):
        result = choice(5, shape=(3,), seed=42)
        self.assertEqual(result.shape, (3,))

    def test_bits(self):
        result = bits((2, 2), seed=42)
        self.assertEqual(result.shape, (2, 2))
        self.assertEqual(result.dtype, jnp.uint32)

    def test_exponential(self):
        result = exponential((2, 2), seed=42)
        self.assertEqual(result.shape, (2, 2))
        self.assertEqual(result.dtype, jnp.float32)

    def test_triangular(self):
        result = triangular(0, 1, 0.5, (2, 2), seed=42)
        self.assertEqual(result.shape, (2, 2))
        self.assertEqual(result.dtype, jnp.float32)

    def test_truncated_normal(self):
        result = truncated_normal(0, 1, (2, 2), seed=42)
        self.assertEqual(result.shape, (2, 2))
        self.assertEqual(result.dtype, jnp.float32)

    def test_poisson(self):
        result = poisson(3, (2, 2), seed=42)
        self.assertEqual(result.shape, (2, 2))
        self.assertEqual(result.dtype, jnp.int32)

    def test_geometric(self):
        result = geometric(0.5, (2, 2), seed=42)
        self.assertEqual(result.shape, (2, 2))
        self.assertEqual(result.dtype, jnp.int32)

    def test_gamma(self):
        result = gamma(2, (2, 2), seed=42)
        self.assertEqual(result.shape, (2, 2))
        self.assertEqual(result.dtype, jnp.float32)

    def test_chisquare(self):
        result = chisquare(2, (2, 2), seed=42)
        self.assertEqual(result.shape, (2, 2))
        self.assertEqual(result.dtype, jnp.float32)


if __name__ == "__main__":
    unittest.main()
