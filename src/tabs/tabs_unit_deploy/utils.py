import chex
import jax
import jax.numpy as jnp
from jax._src.numpy.util import promote_dtypes_inexact


def convert_unit_layer(field: chex.Array, n_units: int) -> chex.Array:
    return jax.vmap(lambda x, i: jnp.where(x == i + 1, 1.0, 0.0), (None, 0))(
        field, jnp.arange(n_units)
    )


def conv_lower_right_padding(lhs: chex.Array, rhs: chex.Array):
    if lhs.ndim != rhs.ndim:
        raise ValueError("lhs and rhs must have the same number of dimensions.")
    if lhs.size == 0 or rhs.size == 0:
        raise ValueError(
            f"zero-size arrays not supported in convolutions, got shapes {lhs.shape} and {rhs.shape}."
        )
    lhs, rhs = promote_dtypes_inexact(lhs, rhs)

    no_swap = all(s1 >= s2 for s1, s2 in zip(lhs.shape, rhs.shape))
    swap = all(s1 <= s2 for s1, s2 in zip(lhs.shape, rhs.shape))
    if not (no_swap or swap):
        raise ValueError("One input must be smaller than the other in every dimension.")

    if swap:
        lhs, rhs = rhs, lhs
    shape = rhs.shape
    rhs = jnp.flip(rhs)
    h, w = lhs.shape

    padding = [(0, s - 1) for s in shape]

    strides = tuple(1 for s in shape)
    result = jax.lax.conv_general_dilated(lhs[None, None], rhs[None, None], strides, padding)

    return result[0, 0][:h, :w]
