from typing import Tuple

from . import operators
from .autodiff import Context
from .fast_ops import FastOps
from .tensor import Tensor
from .tensor_functions import Function, rand


# List of functions in this file:
# - avgpool2d: Tiled average pooling 2D
# - argmax: Compute the argmax as a 1-hot tensor
# - Max: New Function for max operator
# - max: Apply max reduction
# - softmax: Compute the softmax as a tensor
# - logsoftmax: Compute the log of the softmax as a tensor - See https://en.wikipedia.org/wiki/LogSumExp#log-sum-exp_trick_for_log-domain_calculations
# - maxpool2d: Tiled max pooling 2D
# - dropout: Dropout positions based on random noise, include an argument to turn off


def tile(input: Tensor, kernel: Tuple[int, int]) -> Tuple[Tensor, int, int]:
    """Reshape an image tensor for 2D pooling

    Args:
    ----
        input: batch x channel x height x width
        kernel: height x width of pooling

    Returns:
    -------
        Tensor of size batch x channel x new_height x new_width x (kernel_height * kernel_width) as well as the new_height and new_width value.

    """
    batch, channel, height, width = input.shape
    kh, kw = kernel
    assert height % kh == 0
    assert width % kw == 0

    new_height = height // kh
    new_width = width // kw

    # Reshape the tensor to create tiles
    # Final shape: (batch, channel, new_height, new_width, kh * kw)
    output = input.contiguous()
    output = output.view(batch, channel, height, new_width, kw)
    output = output.permute(0, 1, 3, 2, 4)
    output = output.contiguous()
    output = output.view(batch, channel, new_height, new_width, kh * kw)

    return output, new_height, new_width


# TODO: Implement for Task 4.3.
def avgpool2d(input: Tensor, kernel: Tuple[int, int]) -> Tensor:
    """Compute 2D average pooling

    Args:
    ----
        input: batch x channel x height x width
        kernel: height x width of pooling

    Returns:
    -------
        Tensor of size batch x channel x new_height x new_width

    """
    # Use the tile function to reshape the input tensor
    output, new_height, new_width = tile(input, kernel)
    # output shape: (batch, channel, new_height, new_width, kh * kw)

    # First sum over the last dimension
    pooled = output.mean(dim=4).contiguous()

    return pooled.view(output.shape[0], output.shape[1], new_height, new_width)


fast_max = FastOps.reduce(operators.max, -float("inf"))


def argmax(input: Tensor, dim: int = -1) -> Tensor:
    """Compute the argmax as a 1-hot tensor

    Args:
    ----
        input: batch x channel x height x width
        dim: dimension to apply argmax

    Returns:
    -------
        Tensor of size batch x channel x height x width x 1

    """
    # Create a tensor of zeros with the same shape as the input
    # Set the dimension to apply argmax to the maximum value
    max_tensor = fast_max(input, dim)
    return max_tensor == input


class Max(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor, dim: Tensor) -> Tensor:
        """Forward pass for max operation.

        Args:
        ----
            ctx : Context
            a : input tensor
            dim : dimension to reduce

        Returns:
        -------
            Tensor: maximum values

        """
        # Save input for backward pass
        ctx.save_for_backward(a, dim)
        # Get the maximum value along dimension
        return fast_max(a, int(dim.item()))

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, float]:
        """Backward pass for max operation."""
        a, dim = ctx.saved_values
        # Create a mask of where the maximum values occurred
        max_mask = argmax(a, int(dim.item()))
        # Multiply gradient by the mask
        return grad_output * max_mask, 0.0


def max(input: Tensor, dim: int) -> Tensor:
    """Compute the max reduction.

    Args:
    ----
        input: input tensor
        dim: dimension to reduce

    Returns:
    -------
        Tensor of maximum values

    """
    return Max.apply(input, input._ensure_tensor(dim))


def softmax(input: Tensor, dim: int) -> Tensor:
    """Compute the softmax as a tensor."""
    exp_values = input.exp()
    return exp_values / exp_values.sum(dim)


def logsoftmax(input: Tensor, dim: int) -> Tensor:
    """Compute the log of the softmax as a tensor."""
    max_vals = max(input, dim)
    shifted = (input - max_vals).exp()
    sum_exp = shifted.sum(dim)
    return shifted - sum_exp.log()


def maxpool2d(input: Tensor, kernel: Tuple[int, int]) -> Tensor:
    """Tiled max pooling 2D."""
    tiled, new_height, new_width = tile(input, kernel)
    pooled = max(tiled, dim=4)
    return pooled.view(input.shape[0], input.shape[1], new_height, new_width)


def dropout(input: Tensor, rate: float, ignore: bool = False) -> Tensor:
    """Dropout positions based on random noise.

    Args:
    ----
        input: input tensor
        rate: dropout rate (probability of dropping)
        ignore: if True, disable dropout

    Returns:
    -------
        Tensor with dropout applied

    """
    if ignore or rate <= 0.0:
        return input

    # Create random mask
    rand_tensor = rand(input.shape, backend=input.backend)
    dropout_mask = rand_tensor > input._ensure_tensor(rate)

    # # Scale output during training so no rescaling is needed during inference
    # scale = 1.0 / (1.0 - rate)
    return input * dropout_mask
