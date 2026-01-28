from dataclasses import dataclass
from typing import Tuple

import chex
import jax
import jax.numpy as jnp

from src.tabx.constants import TURN_ANGLE, UnitAction
from src.tabx.heuristic_policy.components import LastVisibleTarget, ParsedObservation
from src.tabx.heuristic_policy.params import TABXHeuristicParam
from src.tabx.physics.params import PhysicsParams


def angle_wrap_to_pi(x):
    return (x + jnp.pi) % (2 * jnp.pi) - jnp.pi


def get_visible_target(parsed_obs: ParsedObservation) -> chex.Array:
    """
    Get the visible target for the own unit.

    If the own unit is healer, the visible target is the injured ally.
    If the own unit is not healer, the visible target is the normal target that is not ally.

    Args:
        parsed_obs: Parsed observation

    Returns:
        Visible target
    """

    def get_healer_target(parsed_obs: ParsedObservation) -> chex.Array:
        other_injured_ally = (
            parsed_obs.other.is_ally
            & (parsed_obs.other.health_ratio < 1)
            & (parsed_obs.other.health_ratio > 0)
        )
        exist_injured_ally = jnp.sum(other_injured_ally) > 0
        return (
            jnp.where(exist_injured_ally, other_injured_ally, parsed_obs.other.is_ally)
            & parsed_obs.own.is_healer
        )

    healer_target = get_healer_target(parsed_obs)
    normal_target = jnp.logical_not(parsed_obs.other.is_ally)

    other_is_alive = parsed_obs.other.is_alive

    return jnp.where(parsed_obs.own.is_healer, healer_target, normal_target) & other_is_alive


def get_target_move_position(
    heuristic_config: TABXHeuristicParam, parsed_obs: ParsedObservation
) -> chex.Array:
    """
    Get the target move position for the own unit.

    Args:
        parsed_obs: Parsed observation

    Returns:
        Target move position
    """

    backward_rotate_vector = jnp.stack(
        [jnp.cos(parsed_obs.other.rotation), jnp.sin(parsed_obs.other.rotation)], axis=-1
    ) * (parsed_obs.other.radius[:, None] + parsed_obs.own.radius)

    def get_assassin_target_move_position(parsed_obs: ParsedObservation) -> chex.Array:
        return parsed_obs.other.rel_pos - backward_rotate_vector

    def get_healer_target_move_position(parsed_obs: ParsedObservation) -> chex.Array:
        return parsed_obs.other.rel_pos

    def get_normal_target_move_position(parsed_obs: ParsedObservation) -> chex.Array:
        return parsed_obs.other.rel_pos + backward_rotate_vector

    assassin_target_move_position = get_assassin_target_move_position(parsed_obs)
    healer_target_move_position = get_healer_target_move_position(parsed_obs)
    normal_target_move_position = get_normal_target_move_position(parsed_obs)

    return jnp.where(
        parsed_obs.own.speed >= heuristic_config.assasin_speed,
        assassin_target_move_position,
        jnp.where(
            parsed_obs.own.is_healer, healer_target_move_position, normal_target_move_position
        ),
    )


VisibleTarget = chex.Array
Action = chex.Array


def get_discrete_action(
    heuristic_config: TABXHeuristicParam,
    parsed_obs: ParsedObservation,
    physics_params: PhysicsParams,
    last_visible_target: LastVisibleTarget,
) -> Tuple[Action, VisibleTarget, LastVisibleTarget]:
    """
    Get the move action for the own unit.

    Args:
        parsed_obs: Parsed observation

    Returns:
        Move action
    """

    visible_target = get_visible_target(parsed_obs)
    target_move_position = get_target_move_position(heuristic_config, parsed_obs)

    L2_distnace = jnp.sum(jnp.square(target_move_position), axis=-1)
    is_assassin = parsed_obs.own.speed >= heuristic_config.assasin_speed

    masekd_distance = jnp.where(
        visible_target
        & (
            ~is_assassin | (jnp.abs(parsed_obs.other.max_hp - parsed_obs.other.max_hp.min()) < 0.01)
        ),  # if assassin, find the target with the smallest max hp within 0.01 for the sake of floating point error
        L2_distnace,
        jnp.inf,
    )  # Exclude invisible target by setting the distance to a large value

    # --------------- Move action calculation ---------------
    min_distance_index = jnp.argmin(masekd_distance)
    min_relative_position = parsed_obs.other.rel_pos[
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
    own_is_ranger = parsed_obs.own.attack_range >= heuristic_config.ranger_attack_range
    other_is_agressive = (
        masekd_distance
        < (
            parsed_obs.own.attack_range
            * jnp.where(
                parsed_obs.own.is_healer,
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

    negative_direction = ~positive_direction
    kite_direction_move_action = (
        UnitAction.RIGHT * (x_axis & negative_direction)
        + UnitAction.LEFT * (x_axis & ~negative_direction)
        + UnitAction.UP * (~x_axis & negative_direction)
        + UnitAction.DOWN * (~x_axis & ~negative_direction)
    )
    direction_move_action = (
        UnitAction.RIGHT * (x_axis & positive_direction)
        + UnitAction.LEFT * (x_axis & ~positive_direction)
        + UnitAction.UP * (~x_axis & positive_direction)
        + UnitAction.DOWN * (~x_axis & ~positive_direction)
    )

    # --------------- Rotate action calculation ---------------
    exist_visible_target = jnp.sum(visible_target) > 0

    rotate_angle = angle_wrap_to_pi(
        jnp.where(
            exist_visible_target,
            jnp.arctan2(min_relative_position[1], min_relative_position[0])
            - parsed_obs.own.rotation,
            jnp.pi * 0.1,
        )
    )

    step_angle = TURN_ANGLE * physics_params.dt

    minimum_angle_if_rotate = (
        jnp.rint(rotate_angle / step_angle) * step_angle
    )  # due to rotate action always have step_angle resolution
    rotate_action = jnp.where(rotate_angle > 0, UnitAction.TURN_LEFT, UnitAction.TURN_RIGHT)
    angle_when_rotate = minimum_angle_if_rotate + parsed_obs.own.rotation

    cos_attack_range_half_angle = jnp.cos(parsed_obs.own.sight_angle / 2) * parsed_obs.own.radius
    sin_attack_range_half_angle = jnp.sin(parsed_obs.own.sight_angle / 2) * parsed_obs.own.radius

    width = parsed_obs.own.attack_range

    unit_cosine_vector = jnp.cos(angle_when_rotate)
    unit_sine_vector = jnp.sin(angle_when_rotate)

    relative_unit_x = min_relative_position[0]
    relative_unit_y = min_relative_position[1]
    local_unit_x = (
        relative_unit_x * unit_cosine_vector + relative_unit_y * unit_sine_vector
    )  # rotate -theta to get local coordinate
    local_unit_y = -relative_unit_x * unit_sine_vector + relative_unit_y * unit_cosine_vector

    rx = cos_attack_range_half_angle
    ry = sin_attack_range_half_angle

    closest_x = jnp.clip(local_unit_x, rx, rx + width)
    closest_y = jnp.clip(local_unit_y, -ry, ry)

    dx = local_unit_x - closest_x
    dy = local_unit_y - closest_y

    attackable_if_rotate = (
        dx** 2 + dy** 2 < parsed_obs.own.radius** 2 * jnp.where(own_is_ranger, 1, 0.1)
    )  # if own is ranger, the attackable range is 1, otherwise it is 0.1 because melee unit mostly attack by moving and this helps with chasing

    # --------------- action calculation for bush zone ---------------
    is_bush_zone = parsed_obs.zone.zone_type == 2
    exist_bush_zone = jnp.sum(is_bush_zone) > 0
    zone_distnace = jnp.sum(
        jnp.square(parsed_obs.zone.position), axis=-1
    ) + jnp.inf * jnp.logical_not(is_bush_zone)
    min_zone_distance_index = jnp.argmin(zone_distnace)
    min_zone_distance = parsed_obs.zone.position[min_zone_distance_index]
    max_relative_axis = jnp.argmax(
        jnp.abs(min_zone_distance)
    )  # Find the axis with the largest absolute value
    max_relative_axis_value = min_zone_distance[max_relative_axis]
    max_relative_axis_direction = jnp.sign(max_relative_axis_value)
    x_axis = max_relative_axis == 0  # If the axis is 0, the unit is moving in the x-axis
    zone_positive_direction = max_relative_axis_direction > 0
    zone_direction_move_action = (
        UnitAction.RIGHT * (x_axis & zone_positive_direction)
        + UnitAction.LEFT * (x_axis & ~zone_positive_direction)
        + UnitAction.UP * (~x_axis & zone_positive_direction)
        + UnitAction.DOWN * (~x_axis & ~zone_positive_direction)
    )

    is_in_bush_zone = jnp.all(
        min_zone_distance**2 < parsed_obs.zone.axes[min_zone_distance_index] ** 2
    )

    # --------------- Calculate action when there is no target---------------
    min_target_move_position_nt = last_visible_target.abs_position - parsed_obs.own.position
    max_relative_axis_nt = jnp.argmax(
        jnp.abs(min_target_move_position_nt)
    )  # Find the axis with the largest absolute value
    max_relative_axis_value_nt = min_target_move_position_nt[max_relative_axis_nt]
    max_relative_axis_direction_nt = jnp.sign(max_relative_axis_value_nt)
    x_axis_nt = max_relative_axis_nt == 0  # If the axis is 0, the unit is moving in the x-axis
    positive_direction_nt = (
        max_relative_axis_direction_nt > 0
    )  # If the direction is positive, the unit is moving in the positive direction (right or up)
    direction_move_action_nt = (
        UnitAction.RIGHT * (x_axis_nt & positive_direction_nt)
        + UnitAction.LEFT * (x_axis_nt & ~positive_direction_nt)
        + UnitAction.UP * (~x_axis_nt & positive_direction_nt)
        + UnitAction.DOWN * (~x_axis_nt & ~positive_direction_nt)
    )
    rotate_angle_nt = angle_wrap_to_pi(
        jnp.arctan2(min_target_move_position_nt[1], min_target_move_position_nt[0])
        - parsed_obs.own.rotation,
    )
    rotate_action_nt = jnp.where(rotate_angle_nt > 0, UnitAction.TURN_LEFT, UnitAction.TURN_RIGHT)

    turned_enough = jnp.abs(rotate_angle_nt) <= TURN_ANGLE * physics_params.dt

    action_when_no_target = jnp.where(
        turned_enough,
        direction_move_action_nt,
        rotate_action_nt,
    )

    # if the visible unit may die, set the ever_visible to False.
    rel_pos_to_last_visible_target = jnp.square(
        parsed_obs.other.rel_pos - min_target_move_position_nt
    ).sum(axis=-1)
    target_idx = jnp.argmin(rel_pos_to_last_visible_target)
    may_die = (parsed_obs.other.health[target_idx] <= 0.0) & (
        rel_pos_to_last_visible_target[target_idx] < parsed_obs.other.radius[target_idx] ** 2
    )
    last_visible_target = last_visible_target.replace(
        ever_visible=jnp.where(may_die, False, last_visible_target.ever_visible)
    )

    # --------------- Calculate final action ---------------
    final_action = jnp.where(
        attackable_if_rotate
        & jnp.logical_not(parsed_obs.other.is_attackable[min_distance_index])
        & exist_visible_target,
        rotate_action,
        jnp.where(
            kiting,
            kite_direction_move_action,
            jnp.where(
                exist_visible_target,
                direction_move_action,
                jnp.where(
                    last_visible_target.ever_visible,
                    action_when_no_target,
                    jnp.where(
                        exist_bush_zone & jnp.logical_not(is_in_bush_zone) & own_is_ranger,
                        zone_direction_move_action,
                        UnitAction.TURN_RIGHT,
                    ),
                ),
            ),
        ),
    )

    return (
        final_action,
        visible_target,
        LastVisibleTarget(
            abs_position=jnp.where(
                exist_visible_target,
                min_relative_position + parsed_obs.own.position,
                last_visible_target.abs_position,
            ),
            ever_visible=(exist_visible_target | last_visible_target.ever_visible)
            & jnp.logical_not(
                (
                    jnp.square(last_visible_target.abs_position - parsed_obs.own.position).sum()
                    < 2 * physics_params.dt**2
                )
            ),
        ),
    )


def heuristic_policy(
    key: jax.random.PRNGKey,
    obs: chex.Array,
    last_visible_target: LastVisibleTarget,
    num_agents: int,
    num_zones: int,
    heuristic_config: TABXHeuristicParam,
    physics_params: PhysicsParams,
) -> Tuple[Action, LastVisibleTarget]:
    """
    Role-appropriate heuristic policy for TABX units.

    This policy implements a priority-based decision structure that leverages unit-specific
    statistical advantages through three orthogonal role classes: ranger, assassin, and healer.
    These roles are determined by intrinsic unit attributes (attack range, movement speed, and
    attack damage polarity) and are non-mutually exclusive.

    The policy operates through three hierarchical stages:
    1. Target Selection: Role-specific heuristics prioritize entities within the agent's field of view.
       - Healers prioritize the proximal injured ally, defaulting to the nearest allied unit.
       - Assassins target visible enemies by prioritizing the lowest maximum health.
       - Other agents follow a nearest-neighbor heuristic.

    2. Spatial Positioning: Agents calculate target coordinates based on role-specific objectives.
       - Assassins prioritize posterior positioning relative to the target's orientation.
       - Healers minimize Euclidean distance to maximize healing coverage.
       - Other units seek frontal engagement to maintain direct line of sight.

    3. Threat Mitigation: Rangers implement distance-maintenance protocols.
       - Rangers retreat when opponents breach a critical threshold of their attack range.
       - In the absence of targets, rangers bias toward environmental 'bush zones'.

    The final action follows a priority hierarchy:
    1. Attack if a target is within effective engagement region and cooldown has lapsed.
    2. Rotate to align heading with a visible target if rotation would enable engagement.
    3. Execute kiting protocol for rangers under threat to maintain standoff advantage.
    4. Navigate toward visible role-dependent targets.
    5. Pursue last known coordinates when targets are occluded (memory-based pursuit).
    6. Default to search rotation (rangers first navigate to bush zones, then rotate).
    7. Apply ε-greedy override for limited stochasticity.

    Args:
        key: JAX random key for stochastic action selection
        obs: Observation array containing unit and environment state
        last_visible_target: Memory of the last observed target position
        num_agents: Number of agents in the environment
        num_zones: Number of zones in the environment
        heuristic_config: Configuration parameters for heuristic behavior
        physics_params: Physics parameters of the environment

    Returns:
        Tuple containing:
        - Action: Selected discrete action (with ε-greedy override applied)
        - LastVisibleTarget: Updated memory of last visible target
    """
    parsed_obs = ParsedObservation.from_obs(obs, num_agents, num_zones)

    discrete_action, visible_target, last_visible_target = get_discrete_action(
        heuristic_config, parsed_obs, physics_params, last_visible_target
    )
    # If there exists visible target, rotate to the target, otherwise rotate 0.1pi to find target

    attackable_target = visible_target & parsed_obs.other.is_attackable
    exist_attackable_target = jnp.sum(attackable_target) > 0

    discrete_action = jnp.where(
        exist_attackable_target & jnp.logical_not(parsed_obs.own.is_on_cooldown),
        UnitAction.ATTACK,
        discrete_action,
    )  # If there exists attackable target and not on cooldown, attack, otherwise move

    random_discrete_action = jax.random.choice(
        key,
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

    is_random = jax.random.bernoulli(key, heuristic_config.epsilon)
    return (
        (discrete_action * ~is_random + random_discrete_action * is_random).astype(jnp.int32),
        last_visible_target,
    )
