from typing import NamedTuple

import chex


class Transition(NamedTuple):
    done: chex.Array
    action: chex.Array
    reward: chex.Array
    obs: chex.Array
    info: chex.Array
    unavail_action: chex.Array


def notify(sprites, event, info):
    for key, sprite in sprites.items():
        if hasattr(sprite, "on_" + event):
            sprites[key] = getattr(sprite, "on_" + event)(sprites, info)
    return sprites
