from typing import NamedTuple
import jax.numpy as jnp


class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    reward: jnp.ndarray
    obs: jnp.ndarray
    info: jnp.ndarray
    avail_action: jnp.ndarray


def notify(sprites, event, info):
    for key, sprite in sprites.items():
        if hasattr(sprite, "on_" + event):
            sprites[key] = getattr(sprite, "on_" + event)(sprites, info)
    return sprites
