from typing import Tuple
import chex
import jax
import jax.numpy as jnp
from jax._src.numpy.util import promote_dtypes_inexact

from src.maenv.tabs.units import get_all_unit_spec


def convert_unit_layer(field: chex.Array) -> chex.Array:
    n_units = len(get_all_unit_spec()[0])
    return jax.vmap(lambda x, i: jnp.where(x == i + 1, 1.0, 0.0), (None, 0))(
        field, jnp.arange(n_units)
    )


def get_valid_battle_field_mask(battle_field: chex.Array) -> Tuple[bool, chex.Array]:
    space_occupied_spec = get_all_unit_spec()[-1]
    field_mask = convert_unit_layer(battle_field)
    directions = jnp.array([[0, 0], [1, 0], [0, 1], [1, 1]])

    is_valid = True
    for i, j, k in zip(*list(jnp.nonzero(field_mask))):
        n = space_occupied_spec[i].astype(int)
        for idx, d in enumerate(directions[:n]):
            if idx == 0:
                continue
            if field_mask[i, j + d[0], k + d[1]]:
                is_valid = False  # Overlap among same unit type
                break
            field_mask = field_mask.at[i, j + d[0], k + d[1]].set(1)

    if is_valid:
        is_valid = not jnp.where(
            jnp.sum(field_mask, axis=0) > 1, True, False
        ).any()  # Overlap among different unit type

    return is_valid, field_mask


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
