import jax
import jax.numpy as jnp
import chex

from src.maenv.tabs.tabs_battle_simulator.tabs_battle_simulator import UnitAction


def angle_wrap_to_pi(x):
    return (x + jnp.pi) % (2 * jnp.pi) - jnp.pi


def heuristic_policy(
    key: jax.random.PRNGKey,
    obs: chex.Array,
    num_agents: int,
    epsilon: float = 0.1,
    aggressive_threshold: float = 1.0,
    rotate_noise_scale: float = 0.5,
) -> chex.Array:
    """
    0 : health
    1 : max_health
    2 : relative_x
    3 : relative_y
    4 : rotation
    5 : attack_range
    6 : attack_damage
    7 : cooldown
    8 : cooldown / attack_cooldown
    9 : body_radius
    10 : body_weight
    11 : sight_angle
    12 : is_alive
    13 : is_ally
    14 : is_attackable
    15 : speed
    """

    own_features = obs[:14]
    other_features = obs[14:].reshape(num_agents - 1, -1)
    attack_damage = own_features[6]
    own_is_healer = attack_damage < 0  # If the attack damage is negative, the unit is healer
    own_is_on_cooldown = (
        own_features[8] < 1.0
    )  # If the cooldown is less than 1.0, the unit is on cooldown
    own_rotation = own_features[4] * jnp.pi * 2
    own_attack_range = own_features[5]
    other_max_health = other_features[:, 1]  # Normalized between 0 and 1
    other_is_attackable = other_features[:, -2].astype(jnp.bool_)
    other_hp = other_features[
        :, 0
    ]  # Since the rotation is normalized to 0-1, we need to multiply by 2pi to get the actual rotation
    other_relative_position = other_features[:, 2:4]
    other_is_ally = other_features[:, -3].astype(jnp.bool_)
    other_injured_ally = other_is_ally & (other_max_health < 1)
    exist_injured_ally = jnp.sum(other_injured_ally) > 0
    # If there is injured ally, healer target is the injured ally, otherwise healer target is the closest ally
    healer_target = (
        jnp.where(exist_injured_ally, other_injured_ally, other_is_ally)
    ) & own_is_healer
    # If the unit is not healer
    normal_target = jnp.logical_not(other_is_ally | own_is_healer)
    exist_attackable_target = jnp.sum(other_is_attackable & (healer_target | ~own_is_healer)) > 0
    # Visible target is the target that is alive and either healer target or normal target
    other_is_alive = other_hp > 0
    # If the target units in observation are alive, the unit hp is larger than 0
    visible_target = other_is_alive & (healer_target | normal_target)  # Visible unit + target
    exist_visible_target = jnp.sum(visible_target) > 0
    L2_distnace = jnp.sum(jnp.square(other_relative_position), axis=-1)
    masekd_distance = jnp.where(
        visible_target, L2_distnace, jnp.inf
    )  # Exclude invisible target by setting the distance to a large value
    min_distance_index = jnp.argmin(masekd_distance)
    min_relative_position = other_relative_position[min_distance_index]  # [x, y]
    max_relative_axis = jnp.argmax(
        jnp.abs(min_relative_position)
    )  # Find the axis with the largest absolute value
    max_relative_axis_value = min_relative_position[max_relative_axis]
    max_relative_axis_direction = jnp.sign(max_relative_axis_value)
    x_axis = max_relative_axis == 0  # If the axis is 0, the unit is moving in the x-axis
    positive_direction = (
        max_relative_axis_direction > 0
    )  # If the direction is positive, the unit is moving in the positive direction (right or up)
    # Kiting logic
    own_is_ranger = own_attack_range > 10.0
    other_is_agressive = (
        masekd_distance < (own_attack_range * aggressive_threshold) ** 2
    )  # If the distance is less than the attack range * aggressive threshold, the unit is aggressive
    exist_agressive = jnp.sum(other_is_agressive) > 0
    kiting = (
        own_is_ranger & exist_agressive
    )  # If the unit is ranger and there is aggressive target, the unit is kiting to the target
    positive_direction = jnp.where(
        kiting, ~positive_direction, positive_direction
    )  # If kiting, the unit is not moving in the positive direction to distance from the aggressive target
    move_action = (
        UnitAction.RIGHT * (x_axis & positive_direction)
        + UnitAction.LEFT * (x_axis & ~positive_direction)
        + UnitAction.UP * (~x_axis & positive_direction)
        + UnitAction.DOWN * (~x_axis & ~positive_direction)
    ) * (exist_visible_target) + UnitAction.IDLE * (~exist_visible_target)

    discrete_action = jnp.where(
        exist_attackable_target & jnp.logical_not(own_is_on_cooldown),
        UnitAction.ATTACK,
        move_action,
    )  # If there exists attackable target and not on cooldown, attack, otherwise move
    discrete_key, rotate_key, noise_key = jax.random.split(key, 3)
    rotate_action = jnp.where(
        exist_visible_target,
        angle_wrap_to_pi(
            jnp.arctan2(min_relative_position[1], min_relative_position[0]) - own_rotation
        )
        + jax.random.normal(key=noise_key) * rotate_noise_scale * own_is_ranger,
        jnp.pi * 0.1,
    )  # If there exists visible target, rotate to the target, otherwise rotate 0.1pi to find target
    random_discrete_action = jax.random.choice(
        discrete_key, jnp.array([UnitAction.UP, UnitAction.DOWN, UnitAction.LEFT, UnitAction.RIGHT])
    )
    random_rotate_action = jax.random.normal(rotate_key) / jnp.pi
    random_actions = jnp.stack([random_rotate_action, random_discrete_action])
    actions = jnp.stack([rotate_action, discrete_action])
    is_random = jax.random.bernoulli(key, epsilon)
    return actions * ~is_random + random_actions * is_random
