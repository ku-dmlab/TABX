import jax.numpy as jnp


class Action:
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3
    RIGHT_UP = 4
    RIGHT_DOWN = 5
    LEFT_UP = 6
    LEFT_DOWN = 7
    NONE = 8
    KICK = 9


action_table = (
    jnp.array(
        [
            [0, 1.0],
            [0, -1.0],
            [-1.0, 0],
            [1.0, 0],
            [1.0 / jnp.sqrt(2), 1.0 / jnp.sqrt(2)],
            [1.0 / jnp.sqrt(2), -1.0 / jnp.sqrt(2)],
            [-1.0 / jnp.sqrt(2), 1.0 / jnp.sqrt(2)],
            [-1.0 / jnp.sqrt(2), -1.0 / jnp.sqrt(2)],
            [0, 0],
            [0, 0],
        ]
    )
    * 0.2
)
