from dataclasses import dataclass

import chex
import jax
import jax.numpy as jnp

from src.tabs.config import TABSHeuristicConfig
from src.tabs.tabs import UnitAction


def angle_wrap_to_pi(x):
    return (x + jnp.pi) % (2 * jnp.pi) - jnp.pi


@dataclass
class OwnState:
    health: chex.Array
    max_health_ratio: chex.Array
    position: chex.Array
    rotation: chex.Array
    attack_range: chex.Array
    attack_damage: chex.Array
    cooldown_ratio: chex.Array
    radius: chex.Array
    speed: chex.Array

    @property
    def is_healer(self):
        return self.attack_damage < 0

    @property
    def is_on_cooldown(self):
        return self.cooldown_ratio < 1.0

    @classmethod
    def from_obs(cls, obs: chex.Array) -> "OwnState":
        own = obs[:14]

        return cls(
            health=own[0],
            max_health_ratio=own[1],
            position=own[2:4],
            rotation=own[4] * 2 * jnp.pi,
            attack_range=own[5],
            attack_damage=own[6],
            cooldown_ratio=own[8],
            radius=own[9],
            speed=own[13],
        )


@dataclass
class OtherState:
    health: chex.Array
    health_ratio: chex.Array
    rel_pos: chex.Array
    rotation: chex.Array
    attack_range: chex.Array
    attack_damage: chex.Array
    is_ally: chex.Array
    is_attackable: chex.Array
    max_hp: chex.Array
    radius: chex.Array
    speed: chex.Array

    @property
    def is_alive(self):
        return self.health > 0

    @classmethod
    def from_obs(cls, obs: chex.Array, num_agents: int) -> "OtherState":
        other = obs[14:].reshape(num_agents - 1, -1)

        return cls(
            health=other[:, 0],
            health_ratio=other[:, 1],
            rel_pos=other[:, 2:4],
            rotation=other[:, 4] * 2 * jnp.pi,
            attack_range=other[:, 5],
            attack_damage=other[:, 6],
            is_ally=other[:, 13].astype(jnp.bool_),
            is_attackable=other[:, 14].astype(jnp.bool_),
            radius=other[:, 9],
            speed=other[:, 15],
            max_hp=jnp.where(
                other[:, 13].astype(jnp.bool_) | (other[:, 0] == 0),
                jnp.inf,
                other[:, 0]
                / other[:, 1],  # hp / normalized_hp = max_hp since normalized_hp = hp / max_hp
            ),
        )


@chex.dataclass
class ParsedObservation:
    own: OwnState
    other: OtherState

    @classmethod
    def from_obs(cls, obs: chex.Array, num_agents: int) -> "ParsedObservation":
        return cls(
            own=OwnState.from_obs(obs),
            other=OtherState.from_obs(obs, num_agents),
        )


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
    heuristic_config: TABSHeuristicConfig, parsed_obs: ParsedObservation
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


def get_move_action(
    heuristic_config: TABSHeuristicConfig, parsed_obs: ParsedObservation
) -> chex.Array:
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
    positive_direction = jnp.where(kiting, ~positive_direction, positive_direction)

    move_action = (
        UnitAction.RIGHT * (x_axis & positive_direction)
        + UnitAction.LEFT * (x_axis & ~positive_direction)
        + UnitAction.UP * (~x_axis & positive_direction)
        + UnitAction.DOWN * (~x_axis & ~positive_direction)
    )
    return move_action, min_relative_position


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
    parsed_obs = ParsedObservation.from_obs(obs, num_agents)

    move_action, min_relative_position = get_move_action(heuristic_config, parsed_obs)
    visible_target = get_visible_target(parsed_obs)
    exist_visible_target = jnp.sum(visible_target) > 0

    discrete_key, rotate_key, noise_key = jax.random.split(key, 3)
    own_is_ranger = parsed_obs.own.attack_range >= heuristic_config.ranger_attack_range
    rotate_action = angle_wrap_to_pi(
        jnp.where(
            exist_visible_target,
            jnp.arctan2(min_relative_position[1], min_relative_position[0])
            - parsed_obs.own.rotation
            + jax.random.normal(key=noise_key)
            * (
                heuristic_config.rotate_noise_scale * (own_is_ranger & ~parsed_obs.own.is_healer)
                + heuristic_config.healer_rotate_noise_scale
                * (parsed_obs.own.is_healer & own_is_ranger)
            ),
            jnp.pi * 0.1,
        )
    )
    rotate_action = jnp.where(rotate_action > 0, UnitAction.TURN_RIGHT, UnitAction.TURN_LEFT)
    # If there exists visible target, rotate to the target, otherwise rotate 0.1pi to find target
    move_action = jnp.where(
        exist_visible_target,
        move_action,
        rotate_action,
    )

    attackable_target = visible_target & parsed_obs.other.is_attackable
    exist_attackable_target = jnp.sum(attackable_target) > 0

    discrete_action = jnp.where(
        exist_attackable_target & jnp.logical_not(parsed_obs.own.is_on_cooldown),
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
    return (discrete_action * ~is_random + random_discrete_action * is_random).astype(jnp.int32)
