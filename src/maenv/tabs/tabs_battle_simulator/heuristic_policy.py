import jax
import jax.numpy as jnp

from src.maenv.tabs.tabs_battle_simulator.tabs_battle_simulator import UnitAction


def angle_wrap_to_pi(x):
    return (x + jnp.pi) % (2 * jnp.pi) - jnp.pi


def heuristic_policy(key, obs, num_agents, epsilon=0.1):
    own_status = obs[:14]
    observation = obs[14:].reshape(num_agents - 1, -1)

    attack_damage = own_status[6]
    is_healer = attack_damage < 0
    exist_attackable_target = jnp.sum(observation[:, -2]) > 0

    is_ally = (observation[:, -3]) > 0

    visible_target = ((observation[:, 0]) > 0) & ((is_ally & is_healer) | (~is_ally & ~is_healer))
    exist_visible_target = jnp.sum(visible_target) > 0
    rotation = own_status[4] * jnp.pi * 2

    relative_position = observation[:, 2:4]

    masekd_distance = jnp.sum(jnp.square(relative_position), axis=-1) + (~visible_target) * 1e6

    min_distance_index = jnp.argmin(masekd_distance)

    min_relative_position = relative_position[min_distance_index]
    max_relative_axis = jnp.argmax(jnp.abs(min_relative_position))
    max_relative_axis_value = min_relative_position[max_relative_axis]
    max_relative_axis_direction = jnp.sign(max_relative_axis_value)

    x_axis = max_relative_axis == 0
    positive_direction = max_relative_axis_direction > 0

    move_action = (
        UnitAction.RIGHT * (x_axis & positive_direction)
        + UnitAction.LEFT * (x_axis & ~positive_direction)
        + UnitAction.UP * (~x_axis & positive_direction)
        + UnitAction.DOWN * (~x_axis & ~positive_direction)
    ) * (exist_visible_target) + UnitAction.IDLE * (~exist_visible_target)

    discrete_action = (
        exist_attackable_target * UnitAction.ATTACK + ~exist_attackable_target * move_action
    )
    rotate_action = jnp.pi * 0.1 * (~exist_visible_target) + exist_visible_target * (
        angle_wrap_to_pi(jnp.arctan2(min_relative_position[1], min_relative_position[0]) - rotation)
    )

    discrete_key, rotate_key = jax.random.split(key, 2)
    random_discrete_action = jax.random.choice(
        discrete_key, jnp.array([UnitAction.UP, UnitAction.DOWN, UnitAction.LEFT, UnitAction.RIGHT])
    )
    random_rotate_action = jax.random.normal(rotate_key) / jnp.pi

    random_actions = jnp.stack([random_rotate_action, random_discrete_action])

    actions = jnp.stack([rotate_action, discrete_action])

    is_random = jax.random.bernoulli(key, epsilon)
    return actions * ~is_random + random_actions * is_random
