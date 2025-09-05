import jax
import jax.numpy as jnp

from src.maenv.tabs.tabs_battle_simulator.tabs_battle_simulator import UnitAction


def angle_wrap_to_pi(x):
    return (x + jnp.pi) % (2 * jnp.pi) - jnp.pi


def heuristic_policy(key, obs, num_agents, epsilon=0.1, aggressive_threshold=0.3):
    '''
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
    '''

    own_status = obs[:14]
    observation = obs[14:].reshape(num_agents - 1, -1)
    attack_damage = own_status[6]
    is_healer = attack_damage < 0 # If the attack damage is negative, the unit is healer
    is_cooldown = own_status[8] < 1.0 # If the cooldown is less than 1.0, the unit is on cooldown

    is_ally = (observation[:, -3]) > 0
    injured_ally = is_ally & (observation[:, 1] < 1)
    exist_injured_ally = jnp.sum(injured_ally) > 0
    
    # If there is injured ally, healer target is the injured ally, otherwise healer target is the closest ally
    healer_target = ((exist_injured_ally & injured_ally) | (~exist_injured_ally & is_ally)) & is_healer 
    # If the unit is not healer
    normal_target = ~is_ally & ~is_healer
    exist_attackable_target = jnp.sum(observation[:, -2].astype(jnp.bool_) & (healer_target | ~is_healer)) > 0
    
    # Visible target is the target that is alive and either healer target or normal target
    is_alive = (observation[:, 0]) > 0  # If the target units in observation are alive, the unit hp is larger than 0
    visible_target = is_alive & (healer_target | normal_target) # Visible unit + target
    exist_visible_target = jnp.sum(visible_target) > 0
    
    rotation = own_status[4] * jnp.pi * 2 # Since the rotation is normalized to 0-1, we need to multiply by 2pi to get the actual rotation
    relative_position = observation[:, 2:4]
    L2_distnace = jnp.sum(jnp.square(relative_position), axis=-1)
    masekd_distance = L2_distnace + (~visible_target) * 1e6 # Exclude invisible target by setting the distance to a large value
    min_distance_index = jnp.argmin(masekd_distance)
    min_relative_position = relative_position[min_distance_index]
    max_relative_axis = jnp.argmax(jnp.abs(min_relative_position)) # Find the axis with the largest absolute value
    max_relative_axis_value = min_relative_position[max_relative_axis]
    max_relative_axis_direction = jnp.sign(max_relative_axis_value)
    x_axis = max_relative_axis == 0 # If the axis is 0, the unit is moving in the x-axis
    positive_direction = max_relative_axis_direction > 0 # If the direction is positive, the unit is moving in the positive direction (right or up)
    
    # Kiting logic
    attack_range = own_status[5]
    is_ranger = attack_range > 10.0
    is_agressive = masekd_distance < (attack_range * aggressive_threshold) ** 2 # If the distance is less than the attack range * aggressive threshold, the unit is aggressive
    exist_agressive = jnp.sum(is_agressive) > 0
    kiting = is_ranger & exist_agressive # If the unit is ranger and there is aggressive target, the unit is kiting to the target
    positive_direction = jnp.where(kiting, ~positive_direction, positive_direction) # if kiting, the unit is not moving in the positive direction to distance from the aggressive target

    move_action = (
        UnitAction.RIGHT * (x_axis & positive_direction)
        + UnitAction.LEFT * (x_axis & ~positive_direction)
        + UnitAction.UP * (~x_axis & positive_direction)
        + UnitAction.DOWN * (~x_axis & ~positive_direction)
    ) * (exist_visible_target) + UnitAction.IDLE * (~exist_visible_target)

    discrete_action = jnp.where(exist_attackable_target & ~is_cooldown, UnitAction.ATTACK, move_action) # If there exists attackable target and not on cooldown, attack, otherwise move
    
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
