from typing import Dict

import chex
import jax
import jax.numpy as jnp
from flax import struct

from src.tabs.constants import TURN_ANGLE
from src.tabs.environments.base_maenv import BaseMAEnv
from src.tabs.environments.physics import (
    CircleCollider,
    Ellipse,
    RigidBody,
    Transform,
    physics_step,
    physics_update,
)
from src.tabs.environments.spaces import Box, Discrete
from src.tabs.scenarios import TABSConfig, VectorizedScenario, get_vectorized_scenario
from src.tabs.utils import notify

action_table = jnp.array(
    [
        [0, 1.0, 0.0],
        [0, -1.0, 0.0],
        [-1.0, 0, 0.0],
        [1.0, 0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, -TURN_ANGLE],
        [0.0, 0.0, TURN_ANGLE],
        [0.0, 0.0, 0.0],
    ]
)  # [x_move, y_move, rotate_angle]


class AttackType:
    DEFAULT = 0
    HEALING = 1


class UnitAction:
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3
    ATTACK = 4
    TURN_LEFT = 5
    TURN_RIGHT = 6
    IDLE = 7


@struct.dataclass
class UnitStatus:
    id: chex.Array  # Unique identifier for the unit in the environment
    unit_id: chex.Array  # identifier for the unit in unit info array
    health: chex.Array  # Current health points of the unit
    max_health: chex.Array  # Normalized health points
    attack_damage: chex.Array  # Damage dealt by the unit's attacks
    attack_range: chex.Array  # Maximum distance the unit can attack
    attack_cooldown: chex.Array  # Required cooldown time between attacks
    cooldown: chex.Array  # Time elapsed since the most recent attack
    sight_angle: chex.Array  # Field of view angle in radians
    is_alive: chex.Array  # Boolean showing whether the unit is alive
    attack_type: chex.Array  # Type of attack the unit performs
    is_disabled: chex.Array  # Boolean showing whether the unit is disabled
    speed: chex.Array  # Speed of the unit


@struct.dataclass
class DefaultUnit:
    transform: (
        Transform  # The spatial transform (position, rotation) of the unit in the environment
    )
    rigidbody: RigidBody  # The physical properties of the unit for physics simulation (e.g., velocity, mass)
    collider: CircleCollider  # The collision shape and parameters for detecting overlaps with other objects
    team: chex.Array  # The team identifier to distinguish between different groups of units
    pos_min: chex.Array  # The positional boundaries or limits within which the unit can move
    pos_max: chex.Array  # The positional boundaries or limits within which the unit can move
    status: UnitStatus  # The current status of the unit, including health, attack stats, and other attributes
    damage_dealt: chex.Array  # The damage dealt by the unit
    is_attacking: chex.Array  # Boolean showing whether the unit is attacking

    def update(self, **kwargs):
        config = kwargs["config"]
        next_cooldown = self.status.cooldown + config.dt
        updated_object = physics_update(config, self)

        updated_transform = self.transform._replace(
            position=jnp.clip(updated_object.transform.position, self.pos_min, self.pos_max),
        )

        return updated_object.replace(
            transform=updated_transform,
            status=self.status.replace(cooldown=next_cooldown),
            damage_dealt=self.damage_dealt,
            is_attacking=self.is_attacking,
        )

    def act(self, objects, action, **kwargs):
        action = action.reshape()
        is_attack = UnitAction.ATTACK == action
        game_manager: GameManager = objects["game_manager"]
        target_id = game_manager.attack_target[self.status.id.reshape()]
        target_attackable = game_manager.attackable_matrix[self.status.id.reshape()]
        action_able = ~self.status.is_disabled & self.status.is_alive
        can_attack = (
            self.status.cooldown > self.status.attack_cooldown
        ) & action_able  # if unit is dead, do not attack

        notify(objects, "hit", (self, is_attack, target_id, target_attackable, can_attack))
        attack_success = can_attack & is_attack & target_attackable[target_id.reshape()]

        objects["game_manager"] = objects["game_manager"].replace(
            attack_matrix=objects["game_manager"].attack_matrix.at[
                self.status.id.reshape(), target_id.reshape()
            ].set(objects["game_manager"].attack_matrix[self.status.id.reshape(), target_id.reshape()] |attack_success.reshape())
        )
        objects["game_manager"] = objects["game_manager"].replace(
            attack_matrix=objects["game_manager"].attack_matrix.at[
                target_id.reshape(), self.status.id.reshape()
            ].set(objects["game_manager"].attack_matrix[target_id.reshape(), self.status.id.reshape()] | attack_success.reshape())
        )
        

        move_action = (
            action_table[action, :2] * action_able * self.status.speed
        )  # if unit is dead, do not move
        rotate_action = action_table[action, 2] * action_able * kwargs["physics_params"].dt
        cooldown = is_attack & can_attack

        return self.replace(
            rigidbody=self.rigidbody._replace(velocity=move_action),
            transform=self.transform._replace(
                rotation=(self.transform.rotation + rotate_action) % (2 * jnp.pi)
            ),
            status=self.status.replace(
                cooldown=jnp.clip(
                    0.0 * cooldown + self.status.cooldown * (1 - cooldown),
                    0.0,
                    self.status.attack_cooldown,
                )
            ),
            damage_dealt=attack_success * self.status.attack_damage,
            is_attacking=is_attack & can_attack,
        )

    def on_hit(self, objects, info):
        attacker: DefaultUnit
        attacker, is_attack, target_id, target_attackable, can_attack = info

        is_target = (
            (is_attack & (target_id == self.status.id))
            & can_attack
            & (target_attackable[self.status.id.reshape()])
        )

        damage = attacker.status.attack_damage
        damaged_status = self.on_damage(damage)
        new_status = jax.tree.map(
            lambda x, y: jnp.where(is_target, y, x), self.status, damaged_status
        )

        return self.replace(status=new_status)

    def on_damage(self, damage) -> UnitStatus:
        return self.status.replace(
            health=jnp.clip(self.status.health - damage, 0.0, self.status.max_health)
        )


@struct.dataclass
class Zone:
    zone_type: chex.Array  # The zone type, 1 for lava, 2 for bush
    ellipse: Ellipse  # The parameter of ellipse
    damage: chex.Array  # The damage dealt by the lava zone

    def act(self, objects, physics_params):
        return jax.lax.switch(
            self.zone_type.reshape(),
            [self.act_nothing, self.act_lava, self.act_bush],
            objects,
            physics_params,
        )

    def is_in(self, objects):
        # Return whether an unit is in the zone or not.
        unit_positions = jnp.array(
            [value.transform.position for (key, value) in objects.items() if "unit" in key]
        )
        return (
            jnp.sum((unit_positions - self.ellipse.position) ** 2 / self.ellipse.axes**2, axis=1)
            <= 1
        )

    def act_lava(self, objects, physics_params):
        # Get damaged if the unit is in the lava zone.
        unit_objects = {key: value for (key, value) in objects.items() if "unit" in key}
        is_in = self.is_in(objects)

        for key, unit in unit_objects.items():
            objects[key] = unit.replace(
                status=unit.on_damage(is_in[unit.status.id] * self.damage * physics_params.dt)
            )

        return objects

    def act_bush(self, objects, physics_params):
        """
        Visibility Rules:
            0. We assume that only units within the field of view (FoV) can be observed in all cases.
            1. A unit in a bush is not observable, but units on the same team can observe it.
            2. Units in the same bush can observe each other.
            3. When a unit in a bush succeeds in attacking or is attacked, that unit is observed by the attacking team members / victim team members.
        """

        is_in = self.is_in(objects)
        is_team = jnp.array(
            [value.team for (key, value) in objects.items() if "unit" in key]
        ).flatten()

        # Rule 1.
        # Note that we assume there is two team (0 and 1).
        visible_mask_ally = jnp.logical_or(jnp.logical_not(is_in), jnp.logical_not(is_team))
        visible_mask_enemy = jnp.logical_or(jnp.logical_not(is_in), is_team)
        visible_matrix = jnp.where(
            jnp.repeat(is_team.astype(jnp.bool), repeats=len(is_team)).reshape(
                len(is_team), len(is_team)
            ),
            jnp.logical_and(objects["game_manager"].visible_matrix, visible_mask_enemy[None, :]),
            jnp.logical_and(objects["game_manager"].visible_matrix, visible_mask_ally[None, :]),
        )

        # Rule 2.
        visible_matrix = jnp.logical_and(
            objects["game_manager"].visible_matrix,
            jnp.logical_or(visible_matrix, jnp.logical_and(is_in[:, None], is_in[None, :])),
        )
        
        parsed_state = objects["game_manager"].parsed_state

        # The attacker in bush can be observed.
        # TODO: Observations are shared with members of the same team. (Rule 3)
        attack_in_bush = jnp.logical_and(is_in[:, None], objects["game_manager"].attack_matrix)
        visible_matrix = jnp.logical_or(visible_matrix, attack_in_bush.T)
        visible_matrix = jax.vmap(lambda is_team, visible_matrix : is_team[:, None] & visible_matrix[None], in_axes=(0, 0))(parsed_state.is_teams[..., 0], visible_matrix).sum(axis=0, dtype=jnp.bool)
        visible_matrix = jnp.logical_and(visible_matrix, objects["game_manager"].visible_matrix)

        objects["game_manager"] = objects["game_manager"].replace(visible_matrix=visible_matrix)

        return objects

    def act_nothing(self, objects, physics_params):
        return objects


@struct.dataclass
class ParsedState:
    healths: chex.Array
    positions: chex.Array
    rotations: chex.Array
    cooldowns: chex.Array
    is_alives: chex.Array
    teams: chex.Array
    is_teams: chex.Array
    attack_cooldowns: chex.Array
    attack_damages: chex.Array
    attack_ranges: chex.Array
    speeds: chex.Array
    sight_angles: chex.Array
    body_radiuss: chex.Array
    body_weights: chex.Array
    max_healths: chex.Array
    is_disabled: chex.Array

    @classmethod
    def from_state(cls, state, keys) -> "ParsedState":
        healths = jnp.stack([state[unit].status.health for unit in keys])
        max_healths = healths / jnp.stack([state[unit].status.max_health for unit in keys])
        positions = jnp.stack([state[unit].transform.position for unit in keys])
        rotations = jnp.stack([state[unit].transform.rotation for unit in keys]) / (jnp.pi * 2)
        attack_ranges = jnp.stack([state[unit].status.attack_range for unit in keys])
        attack_damages = jnp.stack([state[unit].status.attack_damage for unit in keys])
        cooldowns = jnp.stack([state[unit].status.cooldown for unit in keys])
        attack_cooldowns = cooldowns / jnp.stack(
            [state[unit].status.attack_cooldown for unit in keys]
        )

        body_radiuss = jnp.stack([state[unit].collider.radius for unit in keys])
        body_weights = jnp.stack([state[unit].rigidbody.mass for unit in keys])
        sight_angles = jnp.stack([state[unit].status.sight_angle for unit in keys]) / (jnp.pi * 2)
        teams = jnp.stack([state[unit].team for unit in keys])
        is_alives = jnp.stack([state[unit].status.is_alive for unit in keys])
        speeds = jnp.stack([state[unit].status.speed for unit in keys])
        is_disabled = jnp.stack([state[unit].status.is_disabled for unit in keys])

        is_ally = teams[None] == teams[:, None]
        return cls(
            healths = healths,
            positions = positions,
            rotations = rotations,
            cooldowns = cooldowns,
            is_alives = is_alives,
            teams = teams,
            is_teams = is_ally,
            attack_cooldowns = attack_cooldowns,
            attack_damages = attack_damages,
            attack_ranges = attack_ranges,
            max_healths = max_healths,
            body_radiuss = body_radiuss,
            body_weights = body_weights,
            sight_angles = sight_angles,
            speeds = speeds,
            is_disabled = is_disabled
        )


@struct.dataclass
class GameManager:
    reward: chex.Array
    done: chex.Array
    timestep: chex.Array
    attack_target: chex.Array
    attackable_matrix: chex.Array
    attack_matrix: chex.Array
    visible_matrix: chex.Array
    distance_matrix: chex.Array
    team_hp_ratio: chex.Array
    last_team_hp_ratio: chex.Array
    parsed_state: ParsedState

    def get_units_in_attack_range(
        self,
        position_diff,
        unit_rotation_vector,
        unit_body_radius_vector,
        unit_attack_range_vector,
        unit_sight_angle_vector,
    ):
        """
        Compute a boolean matrix indicating which units are within each unit's rectangular attack range.

        This function calculates a rectangular attack zone in front of each unit based on their
        rotation direction and attack parameters. The attack zone is defined by the unit's attack range
        and a specified angle width.

        Args:
            position_diff (jnp.ndarray): [N, N, 2] Position differences between all pairs of units.
                                    position_diff[i][j] = position_j - position_i
            unit_rotation_vector (jnp.ndarray): [N, 1] Rotation angle of each unit in radians
            unit_body_radius_vector (jnp.ndarray): [N, 1] Body radius of each unit
            unit_attack_range_vector (jnp.ndarray): [N, 1] Attack range distance of each unit
            unit_sight_angle_vector (float): Half-width of the attack cone in radians (default: pi/4)

        Returns:
            jnp.ndarray: [N, N] Boolean matrix where result[i][j] is True if unit i can attack/detect unit j
                        within its rectangular attack range, False otherwise.

        Note:
            The attack zone is shaped like a rectangle extending forward from each unit's position,
            with width determined by the unit_sight_angle_vector and length by the unit's attack_range.
        """

        cos_attack_range_half_angle = jnp.cos(unit_sight_angle_vector / 2) * unit_body_radius_vector
        sin_attack_range_half_angle = jnp.sin(unit_sight_angle_vector / 2) * unit_body_radius_vector

        width = unit_attack_range_vector[:, None]

        unit_cosine_vector = jnp.cos(unit_rotation_vector)[:, None]
        unit_sine_vector = jnp.sin(unit_rotation_vector)[:, None]

        relative_unit_x = position_diff[:, :, 0:1]
        relative_unit_y = position_diff[:, :, 1:2]
        local_unit_x = (
            relative_unit_x * unit_cosine_vector + relative_unit_y * unit_sine_vector
        )  # rotate -theta to get local coordinate
        local_unit_y = -relative_unit_x * unit_sine_vector + relative_unit_y * unit_cosine_vector

        rx = cos_attack_range_half_angle
        ry = sin_attack_range_half_angle

        closest_x = jnp.clip(local_unit_x, rx[:, None], rx[:, None] + width)
        closest_y = jnp.clip(local_unit_y, -ry[:, None], ry[:, None])

        dx = local_unit_x - closest_x
        dy = local_unit_y - closest_y

        collision = dx**2 + dy**2 < unit_body_radius_vector**2

        available_target = (collision).squeeze(-1) & (
            ~jnp.identity(position_diff.shape[0], dtype=jnp.bool)
        )

        return available_target

    def update_parsed_state(self, state, unit_keys):
        return self.replace(parsed_state=ParsedState.from_state(state, unit_keys))

    def update_distance_matrix(self, objects, unit_keys):
        unit_position_vector = jnp.stack([objects[key].transform.position for key in unit_keys])
        unit_rotation_vector = jnp.stack([objects[key].transform.rotation for key in unit_keys])
        unit_body_radius_vector = jnp.stack([objects[key].collider.radius for key in unit_keys])
        unit_attack_range_vector = jnp.stack(
            [objects[key].status.attack_range for key in unit_keys]
        )
        unit_team_vector = jnp.stack([objects[key].team for key in unit_keys])
        unit_sight_angle_vector = jnp.stack([objects[key].status.sight_angle for key in unit_keys])
        unit_alive_vector = jnp.stack([objects[key].status.is_alive for key in unit_keys])
        unit_attack_type_vector = jnp.stack([objects[key].status.attack_type for key in unit_keys])
        unit_is_disabled_vector = jnp.stack([objects[key].status.is_disabled for key in unit_keys])
        # position_diff[i][j] = i'th unit's position - j'th unit's position
        position_diff = unit_position_vector[None] - unit_position_vector[:, None]
        is_team = unit_team_vector[None] == unit_team_vector[:, None]
        in_attack_range = self.get_units_in_attack_range(
            position_diff,
            unit_rotation_vector,
            unit_body_radius_vector,
            unit_attack_range_vector,
            unit_sight_angle_vector,
        )

        # unit sight processing
        u1_x = jnp.cos(unit_rotation_vector + unit_sight_angle_vector / 2)
        u2_x = jnp.cos(unit_rotation_vector - unit_sight_angle_vector / 2)
        u1_y = jnp.sin(unit_rotation_vector + unit_sight_angle_vector / 2)
        u2_y = jnp.sin(unit_rotation_vector - unit_sight_angle_vector / 2)

        n_u1_x = -u1_y
        n_u1_y = u1_x
        n_u2_x = -u2_y
        n_u2_y = u2_x

        rel_x = position_diff[:, :, 0:1]
        rel_y = position_diff[:, :, 1:2]

        cond_lower = (
            n_u1_x[None] * rel_x + n_u1_y[None] * rel_y + unit_body_radius_vector[:, None] >= 0
        )
        cond_upper = (
            n_u2_x[None] * rel_x + n_u2_y[None] * rel_y - unit_body_radius_vector[:, None] <= 0
        )
        sight_inside = cond_lower & cond_upper & jnp.logical_not(unit_is_disabled_vector[:, None])
        attackable_matrix = (
            in_attack_range
            & (
                (~is_team.squeeze() & (unit_attack_type_vector == AttackType.DEFAULT))
                | (is_team.squeeze() & (unit_attack_type_vector == AttackType.HEALING))
            )
            & unit_alive_vector.T
        ) & ~unit_is_disabled_vector.T
        maksed_relative_distnace = jnp.where(
            jnp.identity(position_diff.shape[0], dtype=jnp.bool_)
            | jnp.logical_not(attackable_matrix),
            jnp.inf,
            jnp.square(position_diff).sum(axis=-1),
        )

        return self.replace(
            attack_target=maksed_relative_distnace.argmin(axis=1),
            attackable_matrix=attackable_matrix,
            visible_matrix=(sight_inside).squeeze().T,
            distance_matrix=position_diff,
        )

    def update(self, **kwargs):
        return self.replace(
            reward=jnp.array([0.0]),
            done=jnp.array([False]),
            timestep=self.timestep + 1,
        )

    def update_team_hp_ratio(self, state, teams, is_disabled, unit_keys, max_team):
        hp = jnp.stack([state[unit].status.health for unit in unit_keys])
        max_hp = jnp.stack([state[unit].status.max_health for unit in unit_keys])

        def team_hp(team):
            return (((teams == team) & ~is_disabled) * hp).sum() / (
                ((teams == team) & ~is_disabled) * max_hp
            ).sum()

        # Clip to remove rewards for healing or damage that exceed max health but seems to be not happening
        team_hp_ratio = jnp.clip(jax.vmap(team_hp)(jnp.arange(max_team)), 0.0, 1.0)

        return self.replace(
            last_team_hp_ratio=self.team_hp_ratio,
            team_hp_ratio=team_hp_ratio,
        )


class TABS(BaseMAEnv):
    def __init__(
        self,
        cfg: TABSConfig,
        obs_type: str = "unit_spec",
        max_episode_steps: int = 512,
    ):
        max_n_ally = cfg.max_n_ally
        max_n_enemy = cfg.max_n_enemy
        max_n_zone = cfg.max_n_zone

        super().__init__(max_n_ally + max_n_enemy)
        self.obs_type = obs_type
        self.ally_keys = [f"unit_{i:02d}" for i in range(max_n_ally)]
        self.enemy_keys = [f"unit_{i:02d}" for i in range(max_n_ally, max_n_ally + max_n_enemy)]
        self.unit_keys = self.ally_keys + self.enemy_keys
        self.agents = self.unit_keys
        self.zone_keys = [f"zone_{i:02d}" for i in range(max_n_zone)]
        self.max_n_ally = max_n_ally
        self.max_n_enemy = max_n_enemy
        self.max_n_zone = max_n_zone
        self.max_episode_steps = max_episode_steps
        self.max_team = 2

        self.empty_state = {
            name: DefaultUnit(
                transform=Transform(position=jnp.array([0.0, 0.0]), rotation=jnp.array([0.0])),
                rigidbody=RigidBody(
                    mass=jnp.array([1.0]),
                    velocity=jnp.array([0.0, 0.0]),
                    acceleration=jnp.array([0.0, 0.0]),
                    is_kinematic=jnp.array([False]),
                ),
                collider=CircleCollider(radius=jnp.array([1.0])),
                team=jnp.array([0]),
                pos_min=jnp.array([0.0, 0.0]),
                pos_max=jnp.array([0.0, 0.0]),
                status=UnitStatus(
                    id=jnp.array([i]),
                    unit_id=jnp.array([0]),
                    health=jnp.array([1.0]),
                    attack_damage=jnp.array([1.0]),
                    attack_range=jnp.array([1.0]),
                    attack_cooldown=jnp.array([1.0]),
                    cooldown=jnp.array([1.0]),
                    sight_angle=jnp.array([1.0]),
                    is_alive=jnp.array([False]),
                    is_disabled=jnp.array([False]),
                    attack_type=jnp.array([AttackType.DEFAULT]),
                    max_health=jnp.array([1.0]),
                    speed=jnp.array([1.0]),
                ),
                damage_dealt=jnp.array([0.0]),
                is_attacking=jnp.array([False]),
            )
            for i, name in enumerate(self.unit_keys)
        }

        self.empty_state["zone"] = {
            name: Zone(
                zone_type=jnp.array([0]),
                ellipse=Ellipse(position=jnp.array([0.0, 0.0]), axes=jnp.array([1.0, 1.0])),
                damage=jnp.array([0.0]),
            )
            for name in self.zone_keys
        }

        self.empty_state["game_manager"] = GameManager(
            attack_target=jnp.zeros((len(self.unit_keys),), dtype=jnp.int32),
            attackable_matrix=jnp.zeros((len(self.unit_keys), len(self.unit_keys)), dtype=jnp.bool),
            attack_matrix=jnp.zeros((len(self.unit_keys), len(self.unit_keys)), dtype=jnp.bool),
            visible_matrix=jnp.zeros((len(self.unit_keys), len(self.unit_keys)), dtype=jnp.bool),
            distance_matrix=jnp.zeros(
                (len(self.unit_keys), len(self.unit_keys)), dtype=jnp.float32
            ),
            reward=jnp.zeros((len(self.unit_keys), 1), dtype=jnp.float32),
            done=jnp.zeros((len(self.unit_keys), 1), dtype=jnp.bool),
            timestep=jnp.array([0]),
            team_hp_ratio=jnp.ones((self.max_team,), dtype=jnp.float32),
            last_team_hp_ratio=jnp.ones((self.max_team,), dtype=jnp.float32),
            parsed_state=ParsedState.from_state(self.empty_state, self.unit_keys),
        )

        self.action_spaces = {
            agent: Discrete(num_categories=action_table.shape[0]) for agent in self.unit_keys
        }
        if self.obs_type == "unit_spec":
            self.observation_spaces = {
                agent: Box(
                    low=0,
                    high=1,
                    shape=(14 + 16 * (len(self.unit_keys) - 1) + len(self.zone_keys) * 6,),
                    dtype=jnp.float32,
                )
                for agent in self.unit_keys
            }
        elif self.obs_type == "unit_id":
            self.observation_spaces = {
                agent: Box(
                    low=0, high=1, shape=(6 + 9 * (len(self.unit_keys) - 1),), dtype=jnp.float32
                )
                for agent in self.unit_keys
            }
        else:
            raise ValueError(f"Invalid observation type: {self.obs_type}")

    def get_obs(self, state):
        if self.obs_type == "unit_spec":
            return self.get_spec_obs(state)
        elif self.obs_type == "unit_id":
            return self.get_id_obs(state)
        else:
            raise ValueError(f"Invalid observation type: {self.obs_type}")

    def get_id_obs(self, state):
        """
        own_feature : [unit_id, health, absolute_x, absolute_y, rotation / 2pi, cooldown, is_alive]
        other_feature : [unit_id, health, relative_x, relative_y, rotation / 2pi, is_alive, is_enemy, is_visible, is_attackable]
        """
        keys = self.unit_keys

        healths = jnp.stack([state[unit].status.health for unit in keys])
        positions = jnp.stack([state[unit].transform.position for unit in keys])
        rotations = jnp.stack([state[unit].transform.rotation for unit in keys]) / (jnp.pi * 2)
        cooldowns = jnp.stack([state[unit].status.cooldown for unit in keys])

        teams = jnp.stack([state[unit].team for unit in keys])
        is_alives = jnp.stack([state[unit].status.is_alive for unit in keys])

        is_teams = teams[None] == teams[:, None]

        n_unit = state["game_manager"].attackable_matrix.shape[0]

        roll_shifts = jnp.arange(0, -n_unit, step=-1)
        v_roll = jax.vmap(lambda array, shift: jnp.roll(array, shift))

        rolled_attackable_matrix = v_roll(state["game_manager"].attackable_matrix, roll_shifts)[
            :, 1:, None
        ]
        rolled_visible_matrix = v_roll(state["game_manager"].visible_matrix, roll_shifts)[
            :, 1:, None
        ]
        rolled_distnace_matrix = v_roll(state["game_manager"].distance_matrix, roll_shifts)[:, 1:]
        rolled_is_teams = v_roll(is_teams, roll_shifts)[:, 1:]

        repeated_health = jnp.repeat(healths[None], repeats=n_unit, axis=0)
        repeated_rotation = jnp.repeat(rotations[None], repeats=n_unit, axis=0)
        repeated_is_alive = jnp.repeat(is_alives[None], repeats=n_unit, axis=0)
        repeated_cooldown = jnp.repeat(cooldowns[None], repeats=n_unit, axis=0)

        rolled_health = v_roll(repeated_health, roll_shifts)[:, 1:]
        rolled_rotation = v_roll(repeated_rotation, roll_shifts)[:, 1:]
        rolled_is_alive = v_roll(repeated_is_alive, roll_shifts)[:, 1:]
        rolled_cooldown = v_roll(repeated_cooldown, roll_shifts)[:, 1:]

        own_feature = jnp.concatenate(
            (
                healths,
                positions,
                rotations,
                cooldowns,
                is_alives,
            ),
            axis=1,
        )

        other_feature = (
            jnp.concatenate(
                (
                    rolled_health,
                    rolled_distnace_matrix,
                    rolled_rotation,
                    rolled_cooldown,
                    rolled_is_alive,
                    rolled_is_teams,
                    rolled_visible_matrix,
                    rolled_attackable_matrix,
                ),
                axis=2,
            )
            * rolled_visible_matrix
        ).reshape(n_unit, -1)

        concated_obs = jnp.concatenate((own_feature, other_feature), axis=1)
        observations = {key: concated_obs[i] for i, key in enumerate(keys)}

        return observations
        # return {key: state[key].status.id for key in self.unit_keys}

    def get_spec_obs(self, state):
        """
        own_feature : [health, max_health, absolute_x, absolute_y, rotation / 2pi, attack_range, attack_damage, cooldown, cooldown / attack_cooldown, body_radius, body_weight, sight_angle, is_alive, speed]
        other_feature : [health, max_health, relative_x, relative_y, rotation / 2pi, attack_range, attack_damage, cooldown, cooldown / attack_cooldown, body_radius, body_weight, sight_angle, is_alive, is_ally, is_attackable, speed]
        """

        keys = self.unit_keys

        parsed_state = state["game_manager"].parsed_state
        n_unit = state["game_manager"].attackable_matrix.shape[0]

        roll_shifts = jnp.arange(0, -n_unit, step=-1)
        v_roll = jax.vmap(lambda array, shift: jnp.roll(array, shift, axis=0))

        rolled_attackable_matrix = v_roll(state["game_manager"].attackable_matrix, roll_shifts)[
            :, 1:, None
        ]
        rolled_visible_matrix = v_roll(state["game_manager"].visible_matrix, roll_shifts)[
            :, 1:, None
        ]
        rolled_distnace_matrix = v_roll(state["game_manager"].distance_matrix, roll_shifts)[:, 1:]
        rolled_is_ally = v_roll(parsed_state.is_teams, roll_shifts)[:, 1:]

        repeated_health = jnp.repeat(parsed_state.healths[None], repeats=n_unit, axis=0)
        repeated_max_health = jnp.repeat(parsed_state.max_healths[None], repeats=n_unit, axis=0)
        repeated_attack_damage = jnp.repeat(
            parsed_state.attack_damages[None], repeats=n_unit, axis=0
        )
        repeated_attack_range = jnp.repeat(parsed_state.attack_ranges[None], repeats=n_unit, axis=0)
        repeated_rotation = jnp.repeat(parsed_state.rotations[None], repeats=n_unit, axis=0)
        repeated_is_alive = jnp.repeat(parsed_state.is_alives[None], repeats=n_unit, axis=0)
        repeated_attack_cooldown = jnp.repeat(
            parsed_state.attack_cooldowns[None], repeats=n_unit, axis=0
        )
        repeated_radius = jnp.repeat(parsed_state.body_radiuss[None], repeats=n_unit, axis=0)
        repeated_mass = jnp.repeat(parsed_state.body_weights[None], repeats=n_unit, axis=0)
        repeated_sight_angle = jnp.repeat(parsed_state.sight_angles[None], repeats=n_unit, axis=0)
        repeated_cooldown = jnp.repeat(parsed_state.cooldowns[None], repeats=n_unit, axis=0)
        repeated_speed = jnp.repeat(parsed_state.speeds[None], repeats=n_unit, axis=0)
        rolled_health = v_roll(repeated_health, roll_shifts)[:, 1:]
        rolled_max_health = v_roll(repeated_max_health, roll_shifts)[:, 1:]
        rolled_attack_damage = v_roll(repeated_attack_damage, roll_shifts)[:, 1:]
        rolled_attack_range = v_roll(repeated_attack_range, roll_shifts)[:, 1:]
        rolled_rotation = v_roll(repeated_rotation, roll_shifts)[:, 1:]
        rolled_is_alive = v_roll(repeated_is_alive, roll_shifts)[:, 1:]
        rolled_attack_cooldown = v_roll(repeated_attack_cooldown, roll_shifts)[:, 1:]
        rolled_radius = v_roll(repeated_radius, roll_shifts)[:, 1:]
        rolled_mass = v_roll(repeated_mass, roll_shifts)[:, 1:]
        rolled_sight_angle = v_roll(repeated_sight_angle, roll_shifts)[:, 1:]
        rolled_cooldown = v_roll(repeated_cooldown, roll_shifts)[:, 1:]
        rolled_speed = v_roll(repeated_speed, roll_shifts)[:, 1:]

        own_feature = jnp.concatenate(
            (
                parsed_state.healths,
                parsed_state.max_healths,
                parsed_state.positions,
                parsed_state.rotations,
                parsed_state.attack_ranges,
                parsed_state.attack_damages,
                parsed_state.cooldowns,
                parsed_state.attack_cooldowns,
                parsed_state.body_radiuss,
                parsed_state.body_weights,
                parsed_state.sight_angles,
                parsed_state.is_alives,
                parsed_state.speeds,
            ),
            axis=1,
        )

        other_feature = (
            jnp.concatenate(
                (
                    rolled_health,
                    rolled_max_health,
                    rolled_distnace_matrix,
                    rolled_rotation,
                    rolled_attack_range,
                    rolled_attack_damage,
                    rolled_cooldown,
                    rolled_attack_cooldown,
                    rolled_radius,
                    rolled_mass,
                    rolled_sight_angle,
                    rolled_is_alive,
                    rolled_is_ally,
                    rolled_attackable_matrix,
                    rolled_speed,
                ),
                axis=2,
            )
            * rolled_visible_matrix
        ).reshape(n_unit, -1)
        zone_types = jnp.stack([state[zone].zone_type for zone in self.zone_keys])
        zone_positions = jnp.stack([state[zone].ellipse.position for zone in self.zone_keys])
        zone_axes = jnp.stack([state[zone].ellipse.axes for zone in self.zone_keys])
        zone_damages = jnp.stack([state[zone].damage for zone in self.zone_keys])

        # For world state
        zone_world_feature = jnp.concatenate(
            (
                zone_types,
                zone_positions,
                zone_axes,
                zone_damages,
            ),
            axis=1,
        ).flatten()

        zone_types = jnp.repeat(zone_types[None], repeats=n_unit, axis=0)
        zone_rel_positions = zone_positions[None] - parsed_state.positions[:, None]
        zone_axes = jnp.repeat(zone_axes[None], repeats=n_unit, axis=0)
        zone_damages = jnp.repeat(zone_damages[None], repeats=n_unit, axis=0)
        zone_feature = jnp.concatenate(
            (zone_types, zone_rel_positions, zone_axes, zone_damages), axis=-1
        )
        zone_feature = jnp.logical_not(zone_types == 0) * zone_feature
        zone_feature = zone_feature.reshape(n_unit, -1)

        concated_obs = jnp.concatenate((own_feature, other_feature, zone_feature), axis=1)
        observations = {key: concated_obs[i] for i, key in enumerate(keys)}
        unit_world_state = jnp.concatenate(
            (
                parsed_state.healths,
                parsed_state.max_healths,
                parsed_state.positions,
                parsed_state.rotations,
                parsed_state.attack_ranges,
                parsed_state.attack_damages,
                parsed_state.cooldowns,
                parsed_state.attack_cooldowns,
                parsed_state.body_radiuss,
                parsed_state.body_weights,
                parsed_state.sight_angles,
                parsed_state.is_alives,
                parsed_state.speeds,
            ),
            axis=-1,
        ).flatten()

        observations["world_state"] = jnp.concatenate([unit_world_state, zone_world_feature])
        return observations

    def reset(self, key, env_params):
        vscenario: VectorizedScenario = env_params["scenario"]

        state = {}
        for i, unit in enumerate(self.unit_keys):
            state[unit] = DefaultUnit(
                transform=Transform(
                    position=vscenario.positions[i],
                    rotation=vscenario.rotations[i],
                ),
                rigidbody=RigidBody(
                    mass=vscenario.body_weights[i],
                    velocity=jnp.array([0.0, 0.0]),
                    acceleration=jnp.array([0.0, 0.0]),
                    is_kinematic=jnp.array([False]),
                ),
                collider=CircleCollider(radius=vscenario.body_radiuss[i]),
                team=vscenario.teams[i],
                pos_min=vscenario.pos_min[i],
                pos_max=vscenario.pos_max[i],
                status=self.empty_state[unit].status.replace(
                    unit_id=vscenario.unit_ids[i],
                    health=vscenario.healths[i],
                    attack_damage=vscenario.attack_damages[i],
                    attack_range=vscenario.attack_ranges[i],
                    attack_cooldown=vscenario.attack_cooldowns[i],
                    cooldown=vscenario.attack_cooldowns[i] * 0.0,
                    sight_angle=vscenario.sight_angles[i],
                    is_alive=vscenario.is_alive[i],
                    is_disabled=vscenario.is_disabled[i],
                    attack_type=vscenario.attack_types[i],
                    max_health=vscenario.healths[i],
                    speed=vscenario.speeds[i],
                ),
                damage_dealt=jnp.array([0.0]),
                is_attacking=jnp.array([False]),
            )

        for i, zone in enumerate(self.zone_keys):
            state[zone] = Zone(
                zone_type=env_params["zone_scenario"].zone_type[i],
                ellipse=Ellipse(
                    position=env_params["zone_scenario"].position[i],
                    axes=env_params["zone_scenario"].axes[i],
                ),
                damage=env_params["zone_scenario"].damage[i],
            )

        state["game_manager"] = GameManager(
            attack_target=jnp.zeros((len(self.unit_keys),), dtype=jnp.int32),
            attackable_matrix=jnp.zeros((len(self.unit_keys), len(self.unit_keys)), dtype=jnp.bool),
            attack_matrix=jnp.zeros((len(self.unit_keys), len(self.unit_keys)), dtype=jnp.bool),
            visible_matrix=jnp.zeros((len(self.unit_keys), len(self.unit_keys)), dtype=jnp.bool),
            distance_matrix=jnp.zeros(
                (len(self.unit_keys), len(self.unit_keys)), dtype=jnp.float32
            ),
            reward=jnp.zeros((len(self.unit_keys), 1), dtype=jnp.float32),
            done=jnp.zeros((len(self.unit_keys), 1), dtype=jnp.bool),
            timestep=jnp.array([0]),
            team_hp_ratio=jnp.ones((self.max_team,), dtype=jnp.float32),
            last_team_hp_ratio=jnp.ones((self.max_team,), dtype=jnp.float32),
            parsed_state=ParsedState.from_state(state, self.unit_keys),
        )

        state["game_manager"] = state["game_manager"].update_distance_matrix(state, self.unit_keys)

        # handling zone effect
        for sprite in state.keys():
            if hasattr(state[sprite], "zone_type"):
                state = state[sprite].act(state, env_params["physics_params"])

        return self.get_obs(state), {"state": state, "physics_params": env_params["physics_params"]}

    def step(self, key, env_state, actions):
        state = env_state["state"]
        physics_params = env_state["physics_params"]
        parsed_state: ParsedState = state["game_manager"].parsed_state

        for sprite in state.keys():
            if hasattr(state[sprite], "update"):
                state[sprite] = state[sprite].update(config=physics_params)

        collider_filter = {
            unit: [key for key in state if hasattr(state[key], "collider") and key != unit]
            for unit in self.unit_keys
        }

        state = physics_step(physics_params, state, list(state.keys()), collider_filter)

        # action processing
        for sprite in self.unit_keys:
            state[sprite] = state[sprite].act(state, actions[sprite], physics_params=physics_params)

        state["game_manager"] = state["game_manager"].update_distance_matrix(state, self.unit_keys)

        # handling zone effect
        for sprite in state.keys():
            if hasattr(state[sprite], "zone_type"):
                state = state[sprite].act(state, physics_params)

        # alive processing after action step, for independent unit sequence
        dones = {}
        for sprite in self.unit_keys:
            state[sprite] = state[sprite].replace(
                status=state[sprite].status.replace(is_alive=(state[sprite].status.health > 0))
            )
            dones[sprite] = (
                jnp.logical_not(state[sprite].status.is_alive) | state[sprite].status.is_disabled
            )

        is_alives = parsed_state.is_alives
        teams = parsed_state.teams
        is_disabled = parsed_state.is_disabled

        def is_team_done(team):
            return ((teams == team) & (~is_alives | is_disabled)).sum() == (teams == team).sum()

        def the_number_of_dead_units(team):
            return ((teams == team) & jnp.logical_not(is_alives)).sum()

        team_dones = jax.vmap(is_team_done)(jnp.arange(self.max_team)) > 0
        team_dead_units = jax.vmap(the_number_of_dead_units)(jnp.arange(self.max_team))
        # If timestep is greater than max_episode_steps, the episode is truncated
        truncation = state["game_manager"].timestep >= self.max_episode_steps
        # If all teams except one are eliminated or truncated, the episode is done
        # Note that truncation does not mean the episode is done, but set done to True (please refer to https://github.com/FLAIROx/JaxMARL/blob/main/jaxmarl/environments/smax/smax_env.py)
        dones["__all__"] = (team_dones.sum() >= self.max_team - 1) | truncation
        state["game_manager"] = (
            state["game_manager"]
            .update_team_hp_ratio(state, teams, is_disabled, self.unit_keys, self.max_team)
            .update_parsed_state(state, self.unit_keys)
        )
        state["game_manager"] = state["game_manager"].replace(attack_matrix = jnp.zeros_like(state["game_manager"].attack_matrix, jnp.bool))
        team_hp_ratio = state["game_manager"].team_hp_ratio
        delta_hp = team_hp_ratio - state["game_manager"].last_team_hp_ratio
        reward_matrix = (jnp.identity(self.max_team) - 0.5) * 2.0

        # The team with the highest hp ratio gets reward 1.0 when the episode is done or truncated
        decision_win_reward = (
            jnp.zeros_like(team_hp_ratio)
            .at[self.max_team - 1 - jnp.argmax(team_hp_ratio[::-1])]
            .set(1.0)
        )
        win_reward = jnp.where(dones["__all__"], decision_win_reward, 0.0)[..., None]
        # dense reward
        rewards = (delta_hp[None] * reward_matrix).sum(axis=-1, keepdims=True)
        rewards += win_reward.reshape(rewards.shape)

        # For metric calculation
        damage_dealt = jnp.stack([state[unit].damage_dealt for unit in self.unit_keys])
        is_attacking = jnp.stack([state[unit].is_attacking for unit in self.unit_keys])

        info = {
            "timestep": state["game_manager"].timestep,
            "disabled_units": is_disabled,
            "done_reward": win_reward,
            "truncation": truncation,
            "is_attacking": is_attacking,
            "damage_dealt": damage_dealt,
            "team_dead_units": team_dead_units,
        }

        rewards = rewards.reshape(-1)
        dones = jax.tree.map(lambda x: x.reshape(), dones)
        info = jax.tree.map(lambda x: x.reshape(-1), info)

        return (
            self.get_obs(state),
            {"state": state, "physics_params": physics_params},
            rewards,
            dones,
            info,
        )

    def world_state_size(self):
        return (
            14 * self.num_agents + len(self.zone_keys) * 6
        )  # n_features * n_agents + n_zones * n_features

    def get_avail_actions(self, state):
        return {agent: jnp.ones((self.action_spaces[agent].n,)) for agent in self.unit_keys}

    def init_render(self, ax, state: Dict, scenario_name: str):
        from src.tabs.visualize.rendering import get_battle_simulator_render

        self.scenario_name = scenario_name

        frame = get_battle_simulator_render(
            scenario_name=self.scenario_name, state=state["state"], unit_keys=self.unit_keys
        )

        # Render
        ax.clear()
        # NOTE: We fix 480p
        ax.set_xlim([0.0, 640])
        ax.set_ylim([0.0, 480])
        ax.axis("off")

        return ax.imshow(frame)

    def update_render(self, im, state: Dict):
        ax = im.axes
        return self.init_render(ax, state, self.scenario_name)
