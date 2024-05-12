import time

import jax
import jax.numpy as jnp


@jax.jit
def normalize_images(images: jnp.ndarray) -> jnp.ndarray:
    """
    Normalize images to have zero mean and unit variance.

    Args:
        images (jnp.ndarray): Input images of shape (N, H, W, C), where N is the number of images,
                              H is height, W is width, and C is the number of channels.

    Returns:
        jnp.ndarray: Normalized images of the same shape as the input.

    Example usage:
        ```
        >>> images = jnp.array([[[[0.0, 0.5], [1.0, 0.25]]]])  # One image of shape (1, 2, 2, 1)
        >>> normalized_images = normalize_images(images)
        >>> print(normalized_images)
        ```
    """
    mean = images.mean(axis=(1, 2, 3), keepdims=True)
    std = images.std(axis=(1, 2, 3), keepdims=True)
    return (images - mean) / (std + 1e-5)


def random_crop(images: jnp.ndarray, crop_size: int) -> jnp.ndarray:
    """
    Randomly crop a batch of images to a specified size using JAX.

    This function takes a batch of images and randomly crops each image to the specified size.
    It uses JAX for random number generation to determine the starting coordinates of the crop.

    Args:
        images (jax.numpy.ndarray): A 4D array of shape (batch_size, height, width, channels),
                                    representing a batch of images.
        crop_size (int): The size to which each image will be cropped. Both the height and width
                         of the crop will be equal to `crop_size`.

    Returns:
        jax.numpy.ndarray: The cropped images, with shape (batch_size, crop_size, crop_size, channels).

    Example usage:
        ```
        >>> images = jnp.ones((10, 100, 100, 3))  # Batch of 10 images of size 100x100 with 3 channels
        >>> crop_size = 64
        >>> cropped_images = random_crop(images, crop_size)
        >>> print(cropped_images.shape)
        ```
    """
    key = jax.random.PRNGKey(int(time.time()))
    _, height, width, _ = images.shape
    height_start = jax.random.randint(key, (), 0, height - crop_size + 1)
    width_start = jax.random.randint(key, (), 0, width - crop_size + 1)
    height_end = height_start + crop_size
    width_end = width_start + crop_size
    crops = images[:, height_start:height_end, width_start:width_end, :]
    return crops


def gaussian_blur(image: jnp.ndarray, kernel_size: int, sigma: float) -> jnp.ndarray:
    """
    Apply Gaussian blur to a multi-channel image.

    Args:
        image (jnp.ndarray): Input image of shape (H, W, C).
        kernel_size (int): Size of the Gaussian kernel (must be odd).
        sigma (float): Standard deviation of the Gaussian kernel.

    Returns:
        jnp.ndarray: Blurred image of the same shape as the input.

    Example usage:
        ```
        >>> image = jnp.ones((5, 5, 3))  # Example image with 3 channels
        >>> blurred_image = gaussian_blur(image, kernel_size=3, sigma=1.0)
        >>> print(blurred_image.shape)
        ```
    """
    assert kernel_size % 2 == 1, "Kernel size must be odd."
    ax = jnp.arange(-kernel_size // 2 + 1.0, kernel_size // 2 + 1.0)
    xx, yy = jnp.meshgrid(ax, ax)
    kernel = jnp.exp(-(xx**2 + yy**2) / (2.0 * sigma**2))
    kernel = kernel / jnp.sum(kernel)

    # Apply convolution to each channel
    blurred_image = jnp.stack(
        [
            jax.scipy.signal.convolve2d(image[:, :, i], kernel, mode="same")
            for i in range(image.shape[2])
        ],
        axis=-1,
    )
    return blurred_image


@jax.jit
def sobel_edge_detection(image: jnp.ndarray) -> jnp.ndarray:
    """
    Apply Sobel edge detection to a multi-channel image.

    Args:
        image (jnp.ndarray): Input image of shape (H, W, C).

    Returns:
        jnp.ndarray: Image representing the edges, of the same shape as the input.

    Example usage:
        ```
        >>> image = jnp.ones((5, 5, 3))  # Example image with 3 channels
        >>> edges = sobel_edge_detection(image)
        >>> print(edges.shape)
        ```
    """
    sobel_x = jnp.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=jnp.float32)
    sobel_y = jnp.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=jnp.float32)

    def apply_sobel(channel):
        gx = jax.scipy.signal.convolve2d(channel, sobel_x, mode="same")
        gy = jax.scipy.signal.convolve2d(channel, sobel_y, mode="same")
        return jnp.sqrt(gx**2 + gy**2)

    # Apply Sobel filter to each channel and sum the results
    edges = jnp.sum(
        jnp.stack(
            [apply_sobel(image[:, :, i]) for i in range(image.shape[2])], axis=-1
        ),
        axis=-1,
    )
    return edges


@jax.jit
def adjust_brightness(image: jnp.ndarray, factor: float) -> jnp.ndarray:
    """
    Adjust the brightness of an image.

    Args:
        image (jnp.ndarray): Input image of shape (H, W, C).
        factor (float): Factor to adjust brightness. Values > 1 increase brightness,
                        values < 1 decrease brightness.

    Returns:
        jnp.ndarray: Brightness-adjusted image of the same shape as the input.

    Example usage:
        ```
        >>> image = jnp.ones((5, 5, 3))  # Example image with 3 channels
        >>> adjusted_image = adjust_brightness(image, factor=1.5)
        >>> print(adjusted_image.shape)
        ```
    """
    return jnp.clip(image * factor, 0, 1)


@jax.jit
def adjust_contrast(image: jnp.ndarray, factor: float) -> jnp.ndarray:
    """
    Adjust the contrast of an image.

    Args:
        image (jnp.ndarray): Input image of shape (H, W, C).
        factor (float): Factor to adjust contrast. Values > 1 increase contrast,
                        values < 1 decrease contrast.

    Returns:
        jnp.ndarray: Contrast-adjusted image of the same shape as the input.

    Example usage:
        ```
        >>> image = jnp.ones((5, 5, 3))  # Example image with 3 channels
        >>> adjusted_image = adjust_contrast(image, factor=1.5)
        >>> print(adjusted_image.shape)
        ```
    """
    mean = jnp.mean(image, axis=(0, 1), keepdims=True)
    return jnp.clip((image - mean) * factor + mean, 0, 1)


@jax.jit
def flip_image(image: jnp.ndarray, horizontal: jnp.ndarray) -> jnp.ndarray:
    """
    Flip an image horizontally or vertically.

    Args:
        image (jnp.ndarray): Input image of shape (H, W, C).
        horizontal (jnp.ndarray): If True (jax.numpy.array with a single True value), flip horizontally;
                                  otherwise, flip vertically.

    Returns:
        jnp.ndarray: Flipped image of the same shape as the input.

    Example usage:
        ```
        >>> image = jnp.ones((5, 5, 3))  # Example image with 3 channels
        >>> flipped_image_horizontally = flip_image(image, jnp.array([True]))
        >>> flipped_image_vertically = flip_image(image, jnp.array([False]))
        >>> print(flipped_image_horizontally.shape, flipped_image_vertically.shape)
        ```
    """
    return jnp.where(horizontal, image[:, ::-1, :], image[::-1, :, :])


@jax.jit
def random_flip_image(
    image: jnp.ndarray, key: jax.random.PRNGKey, horizontal: jnp.ndarray
) -> jnp.ndarray:
    """
    Randomly flip an image horizontally or vertically using JAX.

    Args:
        image (jnp.ndarray): Input image of shape (H, W, C).
        key (jax.random.PRNGKey): A PRNG key used for random number generation.
        horizontal (jnp.ndarray): JAX array with a single boolean value indicating the flip direction.
                                  If True (jax.numpy.array with a single True value), flip horizontally;
                                  otherwise, flip vertically.

    Returns:
        jnp.ndarray: Randomly flipped image of the same shape as the input.

    Example usage:
        ```
        >>> key = jax.random.PRNGKey(0)
        >>> image = jnp.ones((5, 5, 3))  # Example image with 3 channels
        >>> flipped_image = random_flip_image(image, key, jnp.array([True]))
        >>> print(flipped_image.shape)
        ```
    """
    flip = jax.random.uniform(key) > 0.5
    flip_horizontal = jnp.where(horizontal, image[:, ::-1, :], image)
    flip_vertical = jnp.where(horizontal, image, image[::-1, :, :])
    return jnp.where(flip, flip_horizontal, flip_vertical)
