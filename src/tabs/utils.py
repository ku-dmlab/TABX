from typing import NamedTuple

import chex


class Transition(NamedTuple):
    global_done: chex.Array
    done: chex.Array
    action: chex.Array
    value: chex.Array
    reward: chex.Array
    log_prob: chex.Array
    obs: chex.Array
    world_state: chex.Array
    info: chex.Array
    avail_actions: chex.Array


def notify(sprites, event, info):
    for key, sprite in sprites.items():
        if hasattr(sprite, "on_" + event):
            sprites[key] = getattr(sprite, "on_" + event)(sprites, info)
    return sprites
