import jax.numpy as jnp
from collections import namedtuple


class Texture(namedtuple("Texture", ["color", "alpha"])):
    color: jnp.array
    alpha: jnp.array

    def __new__(cls, color=jnp.array((0.0, 0.0, 0.0)), alpha=jnp.array([1.0])):
        return super().__new__(cls, color, alpha)

    def update(self, color, alpha):
        return self._replace(color=jnp.clip(color, 0.0, 1.0), alpha=jnp.clip(alpha, 0.0, 1.0))
