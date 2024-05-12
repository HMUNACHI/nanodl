import jax
import jax.numpy as jnp
from flax import linen as nn
from jax.scipy.special import logsumexp
from jax import random


class KANLinear(nn.Module):
    """
    KANLinear is a class that represents a linear layer in a Kernelized Attention Network (KAN).
    It uses B-splines to model the attention mechanism, which allows for more flexibility than traditional attention mechanisms.

    Attributes:
        in_features (int): The number of input features.
        out_features (int): The number of output features.
        grid_size (int): The size of the grid used for the B-splines. Default is 5.
        spline_order (int): The order of the B-splines. Default is 3.
        scale_noise (float): The scale of the noise added to the B-splines. Default is 0.1.
        scale_base (float): The scale of the base weights. Default is 1.0.
        scale_spline (float): The scale of the spline weights. Default is 1.0.
        enable_standalone_scale_spline (bool): Whether to enable standalone scaling of the spline weights. Default is True.
        base_activation (callable): The activation function to use for the base weights. Default is nn.silu.
        grid_eps (float): The epsilon value used for the grid. Default is 0.02.
        grid_range (list): The range of the grid. Default is [-1, 1].
    """

    in_features: int
    out_features: int
    grid_size: int = 5
    spline_order: int = 3
    scale_noise: float = 0.1
    scale_base: float = 1.0
    scale_spline: float = 1.0
    enable_standalone_scale_spline: bool = True
    base_activation: callable = nn.silu
    grid_eps: float = 0.02
    grid_range: list = [-1, 1]

    def setup(self):
        h = (self.grid_range[1] - self.grid_range[0]) / self.grid_size
        grid = jnp.tile(
            jnp.arange(-self.spline_order, self.grid_size + self.spline_order + 1) * h
            + self.grid_range[0],
            (self.in_features, 1),
        )
        self.grid = self.param("grid", grid.shape, nn.initializers.zeros)

        self.base_weight = self.param(
            "base_weight",
            (self.out_features, self.in_features),
            nn.initializers.kaiming_uniform(),
        )
        self.spline_weight = self.param(
            "spline_weight",
            (self.out_features, self.in_features, self.grid_size + self.spline_order),
            nn.initializers.zeros,
        )
        if self.enable_standalone_scale_spline:
            self.spline_scaler = self.param(
                "spline_scaler",
                (self.out_features, self.in_features),
                nn.initializers.kaiming_uniform(),
            )

        self.reset_parameters()

    def reset_parameters(self):
        self.base_weight = (
            nn.initializers.kaiming_uniform()(
                self.base_weight.shape, self.base_weight.dtype
            )
            * self.scale_base
        )
        noise = (
            (
                random.uniform(
                    jax.random.PRNGKey(0),
                    (self.grid_size + 1, self.in_features, self.out_features),
                )
                - 1 / 2
            )
            * self.scale_noise
            / self.grid_size
        )
        self.spline_weight = self.curve2coeff(
            self.grid.T[self.spline_order : -self.spline_order], noise
        ) * (self.scale_spline if not self.enable_standalone_scale_spline else 1.0)
        if self.enable_standalone_scale_spline:
            self.spline_scaler = (
                nn.initializers.kaiming_uniform()(
                    self.spline_scaler.shape, self.spline_scaler.dtype
                )
                * self.scale_spline
            )

    def b_splines(self, x):
        grid = self.grid
        x = x[..., None]
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).astype(x.dtype)
        for k in range(1, self.spline_order + 1):
            bases = (x - grid[:, : -(k + 1)]) / (
                grid[:, k:-1] - grid[:, : -(k + 1)]
            ) * bases[..., :-1] + (grid[:, k + 1 :] - x) / (
                grid[:, k + 1 :] - grid[:, 1:(-k)]
            ) * bases[
                ..., 1:
            ]
        return bases

    def curve2coeff(self, x, y):
        A = self.b_splines(x).transpose((1, 0, 2))
        B = y.transpose((1, 0, 2))
        solution = jnp.linalg.lstsq(A, B)[0]
        result = solution.transpose((2, 0, 1))
        return result

    @property
    def scaled_spline_weight(self):
        return self.spline_weight * (
            self.spline_scaler[..., None]
            if self.enable_standalone_scale_spline
            else 1.0
        )

    def __call__(self, x):
        base_output = jnp.dot(self.base_activation(x), self.base_weight.T)
        spline_output = jnp.dot(
            self.b_splines(x).reshape(x.shape[0], -1),
            self.scaled_spline_weight.reshape(self.out_features, -1).T,
        )
        return base_output + spline_output

    def update_grid(self, x, margin=0.01):
        batch = x.shape[0]

        splines = self.b_splines(x).transpose((1, 0, 2))
        orig_coeff = self.scaled_spline_weight.transpose((1, 2, 0))
        unreduced_spline_output = jnp.matmul(splines, orig_coeff).transpose((1, 0, 2))

        x_sorted = jnp.sort(x, axis=0)
        grid_adaptive = x_sorted[
            jnp.linspace(0, batch - 1, self.grid_size + 1, dtype=jnp.int64)
        ]

        uniform_step = (x_sorted[-1] - x_sorted[0] + 2 * margin) / self.grid_size
        grid_uniform = (
            jnp.arange(self.grid_size + 1, dtype=jnp.float32)[..., None] * uniform_step
            + x_sorted[0]
            - margin
        )

        grid = self.grid_eps * grid_uniform + (1 - self.grid_eps) * grid_adaptive
        grid = jnp.concatenate(
            [
                grid[:1]
                - uniform_step * jnp.arange(self.spline_order, 0, -1)[..., None],
                grid,
                grid[-1:]
                + uniform_step * jnp.arange(1, self.spline_order + 1)[..., None],
            ],
            axis=0,
        )

        self.grid = grid.T
        self.spline_weight = self.curve2coeff(x, unreduced_spline_output)

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        l1_fake = jnp.mean(jnp.abs(self.spline_weight), axis=-1)
        regularization_loss_activation = jnp.sum(l1_fake)
        p = l1_fake / regularization_loss_activation
        regularization_loss_entropy = -jnp.sum(p * jnp.log(p))
        return (
            regularize_activation * regularization_loss_activation
            + regularize_entropy * regularization_loss_entropy
        )


class KAN(nn.Module):
    """
    KAN is a class that represents a Kernelized Attention Network (KAN).
    It is a type of neural network that uses a kernelized attention mechanism, which allows for more flexibility than traditional attention mechanisms.

    Attributes:
        layers_hidden (list): A list of integers representing the number of hidden units in each layer.
    """

    layers_hidden: list
    grid_size: int = 5
    spline_order: int = 3
    scale_noise: float = 0.1
    scale_base: float = 1.0
    scale_spline: float = 1.0
    base_activation: callable = nn.silu
    grid_eps: float = 0.02
    grid_range: list = [-1, 1]

    def setup(self):
        self.layers = [
            KANLinear(
                in_features,
                out_features,
                grid_size=self.grid_size,
                spline_order=self.spline_order,
                scale_noise=self.scale_noise,
                scale_base=self.scale_base,
                scale_spline=self.scale_spline,
                base_activation=self.base_activation,
                grid_eps=self.grid_eps,
                grid_range=self.grid_range,
            )
            for in_features, out_features in zip(
                self.layers_hidden, self.layers_hidden[1:]
            )
        ]

    def __call__(self, x, update_grid=False):
        for layer in self.layers:
            if update_grid:
                layer.update_grid(x)
            x = layer(x)
        return x

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        return sum(
            layer.regularization_loss(regularize_activation, regularize_entropy)
            for layer in self.layers
        )
