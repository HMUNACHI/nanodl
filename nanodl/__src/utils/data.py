import collections
from dataclasses import dataclass
from typing import Iterator

import jax
import jax.numpy as jnp

# This script modifies the JAX DataLoader from the following repository:
# JAX DataLoader by Birkhoff G. (https://birkhoffg.github.io/jax-dataloader/)
# Accessed on [Date you accessed the repository, e.g., February 4, 2024]
# This DataLoader implementation is used for efficient data loading in JAX-based machine learning projects.


class Dataset:
    """
    A PyTorch-like Dataset class for JAX.

    This is a base class for creating datasets in JAX. Subclasses should implement
    the `__len__` method to return the size of the dataset and the `__getitem__`
    method to return a data item at a given index.

    Example usage:
    ```
        >>> class MyDataset(Dataset):
        ...     def __init__(self, data):
        ...         self.data = data
        ...     def __len__(self):
        ...         return len(self.data)
        ...     def __getitem__(self, index):
        ...         return self.data[index]
        >>> dataset = MyDataset(jnp.arange(10))
        >>> print(len(dataset))
        >>> print(dataset[5])
    ```
    """

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, index):
        raise NotImplementedError


class ArrayDataset(Dataset):
    """
    Dataset wrapping JAX numpy arrays.

    This class wraps multiple JAX numpy arrays into a dataset. Each array represents
    a different modality of the data (e.g., features and labels). All arrays must
    have the same first dimension (number of samples).

    Args:
        *arrays (jnp.array): Variable number of JAX numpy arrays to include in the dataset.

    Example usage:
    ```
        >>> dataset = ArrayDataset(jnp.array([1, 2, 3]), jnp.array([4, 5, 6]))
        >>> print(len(dataset))
        >>> print(dataset[1])
    ```
    """

    def __init__(self, *arrays: jnp.array):
        assert all(
            arrays[0].shape[0] == arr.shape[0] for arr in arrays
        ), "All arrays must have the same first dimension."
        self.arrays = arrays

    def __len__(self):
        return self.arrays[0].shape[0]

    def __getitem__(self, index):
        return tuple(arr[index] for arr in self.arrays)


class DataLoader:
    """
    DataLoader in Vanilla Jax.

    This class provides a way to iterate over batches of data from a given dataset.
    It supports batch processing, shuffling, and dropping the last batch if it's
    smaller than the specified batch size.

    Args:
        dataset (Dataset): The dataset from which to load the data.
        batch_size (int, optional): Number of samples per batch. Default is 1.
        shuffle (bool, optional): Whether to shuffle the data. Default is False.
        drop_last (bool, optional): Whether to drop the last incomplete batch.
                                    Default is False.

    Example usage:
    ```
        >>> dataset = ArrayDataset(jnp.ones((1001, 256, 256)), jnp.ones((1001, 256, 256)))
        >>> dataloader = DataLoader(dataset, batch_size=10, shuffle=True, drop_last=False)
        >>> for batch in dataloader:
        ...     print(batch.shape)
    ```
    """

    def __init__(
        self,
        dataset: Dataset,
        batch_size: int = 1,
        shuffle: bool = False,
        drop_last: bool = False,
        **kwargs
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last

        self.keys = _PRNGSequence(seed=Config.default().global_seed)
        self.data_len = len(dataset)  # Length of the dataset
        self.indices = jnp.arange(self.data_len)  # available indices in the dataset
        self.pose = 0  # record the current position in the dataset
        self._shuffle()

    def _shuffle(self):
        if self.shuffle:
            self.indices = jax.random.permutation(next(self.keys), self.indices)

    def _stop_iteration(self):
        self.pose = 0
        self._shuffle()
        raise StopIteration

    def __len__(self):
        if self.drop_last:
            batches = len(self.dataset) // self.batch_size  # get the floor of division
        else:
            batches = -(
                len(self.dataset) // -self.batch_size
            )  # get the ceil of division
        return batches

    def __next__(self):
        if self.pose + self.batch_size <= self.data_len:
            batch_indices = self.indices[self.pose : self.pose + self.batch_size]
            batch_data = self.dataset[batch_indices]
            self.pose += self.batch_size
            return batch_data
        elif self.pose < self.data_len and not self.drop_last:
            batch_indices = self.indices[self.pose :]
            batch_data = self.dataset[batch_indices]
            self.pose += self.batch_size
            return batch_data
        else:
            self._stop_iteration()

    def __iter__(self):
        return self


@dataclass
class Config:
    rng_reserve_size: int
    global_seed: int

    @classmethod
    def default(cls):
        return cls(rng_reserve_size=1, global_seed=42)


class _PRNGSequence(Iterator[jax.random.PRNGKey]):
    """
    An Iterator of Jax PRNGKey (minimal version of `haiku.PRNGSequence`).

    This class provides an iterator over PRNG keys generated from a seed. It is useful
    for generating random numbers in a reproducible way.

    Args:
        seed (int): Seed for generating the initial PRNG key.

    Example usage:
    ```
        >>> prng_seq = PRNGSequence(42)
        >>> key = next(prng_seq)
    ```
    """

    def __init__(self, seed: int):
        self._key = jax.random.PRNGKey(seed)
        self._subkeys = collections.deque()

    def reserve(self, num):
        if num > 0:
            new_keys = tuple(jax.random.split(self._key, num + 1))
            self._key = new_keys[0]
            self._subkeys.extend(new_keys[1:])

    def __next__(self):
        if not self._subkeys:
            self.reserve(Config.default().rng_reserve_size)
        return self._subkeys.popleft()
