import jax
import jax.numpy as jnp
import chex

from src.tabs.tabs_battle_simulator.tabs_battle_simulator import UnitAction
from src.tabs.config import TABSHeuristicConfig


def angle_wrap_to_pi(x):
    return (x + jnp.pi) % (2 * jnp.pi) - jnp.pi


def heuristic_policy(
    key: jax.random.PRNGKey,
    obs: chex.Array,
    num_agents: int,
    heuristic_config: TABSHeuristicConfig,
) -> chex.Array:
    """
    Heuristic policy for different unit types in TABS battle simulator.

    Basic behavior for all units:
    - If no target is visible, rotate by 0.1π radians to search for enemies
    - If there exists attackable target and not on cooldown, execute attack and rotate toward the target

    Args:
        key: JAX random key for stochastic decisions
        obs: i'th unit's Observation array
        num_agents: Total number of agents in the environment (including self)
        epsilon: Probability of taking random action
        aggressive_threshold: Threshold for aggressive behavior (0.0-1.0)
        rotate_noise_scale: Scale factor for rotation noise (used to adjust balance for ranger range)
        assasin_speed: Minimum speed threshold to classify unit as assassin
        ranger_attack_range: Minimum attack range threshold to classify unit as archer

    Returns:
        i'th unit's action (rotate_action, discrete_action)

    Special Unit Logic:
    - Healer: Units with negative attack damage
      - If all visible allies are at full health: move to closest ally and heal if possible
      - If any visible ally has less than 100% health: move to and heal the closest injured ally
      - Does not consider obstructions from other units during movement

    - Archer: Units with attack range above a certain threshold
      - If enemies are within 100 * aggressive_threshold% of attack range and unit is on cooldown or has no target:
        move away from the closest enemy in the fastest escape direction
      - Does not consider map boundaries or unit obstructions

    - Assassin: Units with speed >= assasin_speed
      - Target the closest enemy among those with the lowest maximum health
      - Move to the target's back (opposite to target's facing direction)

    """
    """
    Observation Features Indices:
    own features:
    0 : health
    1 : max_health / max_health
    2 : absolute_x
    3 : absolute_y
    4 : rotation / 2pi
    5 : attack_range
    6 : attack_damage
    7 : cooldown
    8 : cooldown / attack_cooldown
    9 : body_radius
    10 : body_weight
    11 : sight_angle
    12 : is_alive
    13 : speed

    other features (per visible unit):
    0 : health
    1 : health / max_health
    2 : relative_x
    3 : relative_y
    4 : rotation / 2pi
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
    own_speed = own_features[13]
    own_is_assassin = own_speed >= heuristic_config.assasin_speed
    own_rotation = own_features[4] * jnp.pi * 2
    own_attack_range = own_features[5]
    own_radius = own_features[9]
    other_max_health = other_features[:, 1]  # Normalized between 0 and 1
    other_is_attackable = other_features[:, -2].astype(jnp.bool_)
    other_hp = other_features[
        :, 0
    ]  # Since the rotation is normalized to 0-1, we need to multiply by 2pi to get the actual rotation
    other_relative_position = other_features[:, 2:4]
    other_rotation = other_features[:, 4] * jnp.pi * 2
    other_radius = other_features[:, 9]
    other_is_ally = other_features[:, -3].astype(jnp.bool_)
    other_injured_ally = other_is_ally & (other_max_health < 1) & (other_max_health > 0)
    exist_injured_ally = jnp.sum(other_injured_ally) > 0
    other_enemy_max_hp = jnp.where(
        other_is_ally | (other_hp == 0),
        jnp.inf,
        other_hp
        / other_features[:, 1],  # hp / normalized_hp = max_hp since normalized_hp = hp / max_hp
    )
    min_other_enemy_max_hp = (
        other_enemy_max_hp.min()
    )  # For assassin unit, find the target with the smallest max hp

    # other_relative_position for assassin unit
    backward_rotate_vector = jnp.stack(
        [jnp.cos(other_rotation), jnp.sin(other_rotation)], axis=-1
    ) * (other_radius[:, None] + own_radius)  # [n_unit, 2]
    target_move_position = jnp.where(
        own_is_assassin,
        other_relative_position - backward_rotate_vector,
        jnp.where(
            own_is_healer, other_relative_position, other_relative_position + backward_rotate_vector
        ),
    )  # If assassin, target move position is the target's back
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
    L2_distnace = jnp.sum(jnp.square(target_move_position), axis=-1)
    masekd_distance = jnp.where(
        visible_target
        & (
            ~own_is_assassin | (jnp.abs(other_enemy_max_hp - min_other_enemy_max_hp) < 0.01)
        ),  # if assassin, find the target with the smallest max hp within 0.01 for the sake of floating point error
        L2_distnace,
        jnp.inf,
    )  # Exclude invisible target by setting the distance to a large value
    min_distance_index = jnp.argmin(masekd_distance)
    min_relative_position = other_relative_position[
        min_distance_index
    ]  # [x, y], used for rotation calculation
    min_target_move_position = target_move_position[
        min_distance_index
    ]  # [x, y], used for move calculation
    max_relative_axis = jnp.argmax(
        jnp.abs(min_target_move_position)
    )  # Find the axis with the largest absolute value
    max_relative_axis_value = min_target_move_position[max_relative_axis]
    max_relative_axis_direction = jnp.sign(max_relative_axis_value)
    x_axis = max_relative_axis == 0  # If the axis is 0, the unit is moving in the x-axis
    positive_direction = (
        max_relative_axis_direction > 0
    )  # If the direction is positive, the unit is moving in the positive direction (right or up)
    # Kiting logic
    own_is_ranger = own_attack_range >= heuristic_config.ranger_attack_range
    other_is_agressive = (
        masekd_distance
        < (
            own_attack_range
            * jnp.where(
                own_is_healer,
                heuristic_config.healer_aggressive_threshold,
                heuristic_config.aggressive_threshold,
            )
        )
        ** 2
    )  # If the distance is less than the attack range * aggressive threshold, the unit is aggressive
    exist_agressive = jnp.sum(other_is_agressive) > 0
    kiting = (
        own_is_ranger & exist_agressive
    )  # If the unit is ranger and there is aggressive target, the unit is kiting to the target
    positive_direction = jnp.where(
        kiting, ~positive_direction, positive_direction
    )  # If kiting, the unit is not moving in the positive direction to distance from the aggressive target

    discrete_key, rotate_key, noise_key = jax.random.split(key, 3)
    rotate_action = angle_wrap_to_pi(
        jnp.where(
            exist_visible_target,
            jnp.arctan2(min_relative_position[1], min_relative_position[0])
            - own_rotation
            + jax.random.normal(key=noise_key)
            * (
                heuristic_config.rotate_noise_scale * (own_is_ranger & ~own_is_healer)
                + heuristic_config.healer_rotate_noise_scale * (own_is_healer & own_is_ranger)
            ),
            jnp.pi * 0.1,
        )
    )
    rotate_action = jnp.where(rotate_action > 0, UnitAction.TURN_RIGHT, UnitAction.TURN_LEFT)
    # If there exists visible target, rotate to the target, otherwise rotate 0.1pi to find target
    move_action = jnp.where(
        exist_visible_target,
        UnitAction.RIGHT * (x_axis & positive_direction)
        + UnitAction.LEFT * (x_axis & ~positive_direction)
        + UnitAction.UP * (~x_axis & positive_direction)
        + UnitAction.DOWN * (~x_axis & ~positive_direction),
        rotate_action,
    )

    discrete_action = jnp.where(
        exist_attackable_target & jnp.logical_not(own_is_on_cooldown),
        UnitAction.ATTACK,
        move_action,
    )  # If there exists attackable target and not on cooldown, attack, otherwise move

    random_discrete_action = jax.random.choice(
        discrete_key,
        jnp.array(
            [
                UnitAction.UP,
                UnitAction.DOWN,
                UnitAction.LEFT,
                UnitAction.RIGHT,
                UnitAction.TURN_LEFT,
                UnitAction.TURN_RIGHT,
            ]
        ),
    )

    # Priority
    # 1. If attackable target exists, attack
    # 2. If there exists aggressive target, move away from the target
    # 3. If there exists visible target, rotate to the target
    # 4. If there exists no target, turn left to find target

    is_random = jax.random.bernoulli(key, heuristic_config.epsilon)
    return jnp.array([discrete_action * ~is_random + random_discrete_action * is_random]).astype(
        jnp.int32
    )
