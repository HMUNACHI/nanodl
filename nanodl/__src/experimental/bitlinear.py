import jax
import jax.numpy as jnp
from flax import linen as nn


class BitLinear(nn.Module):
    """
    Implements a linear transformation layer with quantization for both activations and weights,
    optimized for low-bit inference. The layer is designed to operate in two modes: training and inference.
    During training, the activations and weights are quantized using separate quantization functions,
    aiming to simulate low-bit operations and reduce the quantization error. For inference, a more
    aggressive quantization scheme is applied to both activations and weights, potentially different
    from the training quantization, to maximize performance and efficiency on low-bit hardware.

    Attributes:
        output_features (int): The number of output features.
        kernel_init (callable): A function to initialize the weights. Default is LeCun normal initializer.
    """

    output_features: int
    kernel_init: callable = nn.initializers.lecun_normal()

    @nn.compact
    def __call__(self, x, training=False):
        w = self.param("kernel", self.kernel_init, (x.shape[-1], self.output_features))

        if not training:
            x_quant, x_scale = self.fused_activation_norm_quant(x)

            # HELP: How run externally on params at once for efficiency
            # Quantising weigts all over again each call is repeated work
            # This can be done on params dict using jax tree utils.
            # Albeit the weight scale for quantisation needs to be utilised at inference
            # Its easy to bypassed on its own by passing the weight scale during a call
            # This will be a module in various transformer models in my project (NanoDL)
            # Is there a way to achieve this without complication my existing codebase?
            w, w_scale = self.inference_weight_quant(w)

            return self.inference_lowbit_matmul(x_quant, w) / w_scale / x_scale

        x_norm = self.rmsnorm(x)
        x_quant = x_norm + jax.lax.stop_gradient(self.activation_quant(x_norm) - x_norm)
        w_quant = w + jax.lax.stop_gradient(self.weight_quant(w) - w)
        return jnp.dot(x_quant, w_quant)

    def rmsnorm(self, x):
        return x / jnp.sqrt(jnp.mean(jnp.square(x), axis=-1, keepdims=True) + 1e-5)

    def activation_quant(self, x):
        scale = 127.0 / jnp.max(jnp.abs(x), axis=-1, keepdims=True).clip(min=1e-5)
        y = jnp.round(x * scale).clip(-128, 127) / scale
        return y

    def weight_quant(self, w):
        scale = 1.0 / jnp.mean(jnp.abs(w)).clip(min=1e-5)
        u = jnp.round(w * scale).clip(-1, 1) / scale
        return u

    def fused_activation_norm_quant(self, x):
        x_norm = self.rmsnorm(x)
        scale = 127.0 / jnp.max(jnp.abs(x_norm), axis=-1, keepdims=True).clip(min=1e-5)
        x_quant = jnp.round(x_norm * scale).clip(-128, 127) / scale
        return x_quant, scale

    def inference_weight_quant(self, w):
        scale = jnp.abs(w).mean().clip(min=1e-5)
        u = jnp.sign(w - w.mean()) * scale
        return u, scale

    # Help: how to implement lowbit matmul kernel for efficiency that can be integrated into Flax model
    def inference_lowbit_matmul(self, x, w):
        return jnp.dot(x, w)
