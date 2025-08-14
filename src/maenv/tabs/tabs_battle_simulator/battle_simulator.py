from collections import namedtuple
import chex
import jax
import jax.numpy as jnp
from src.maenv.physics import Transform, RigidBody, CircleCollider, physics_update
from src.maenv.utils import notify
from src.maenv.environments.base_maenv import BaseMAEnv
from src.maenv.physics import Transform, RigidBody, CircleCollider, physics_step
from easydict import EasyDict
from typing import Dict
from src.maenv.tabs.scenarios import Scenario, get_vectorized_scenario, VectorizedScenario


class UnitStatus(
    namedtuple(
        "UnitStatus",
        [
            "id",
            "unit_id",
            "health",
            "max_health",
            "attack_damage",
            "attack_range",
            "attack_cooldown",
            "cooldown",
            "sight_angle",
            "is_alive",
            "attack_type",
            "is_disabled",
        ],
    )
):
    id: chex.Array  # Unique identifier for the unit in the environment
    unit_id: chex.Array  # identifier for the unit in unit info array
    health: chex.Array  # Current health points of the unit
    attack_damage: chex.Array  # Damage dealt by the unit's attacks
    attack_range: chex.Array  # Maximum distance the unit can attack
    attack_cooldown: chex.Array  # Required cooldown time between attacks
    cooldown: chex.Array  # Time elapsed since the most recent attack
    sight_angle: chex.Array  # Field of view angle in radians
    is_alive: chex.Array  # Boolean showing whether the unit is alive
    attack_type: chex.Array  # Type of attack the unit performs
    is_disabled: chex.Array  # Boolean showing whether the unit is disabled

    def to_array(self):
        return jnp.concatenate(
            (
                self.health,
                self.max_health,
                self.attack_damage,
                self.attack_range,
                self.attack_cooldown,
                self.cooldown,
                self.sight_angle,
                self.is_alive,
                self.attack_type,
                self.is_disabled,
            )
        )


move_table = jnp.array(
    [
        [0, 1.0],
        [0, -1.0],
        [-1.0, 0],
        [1.0, 0],
        [0.0, 0.0],
        [0.0, 0.0],
    ]
)


class AttackType:
    DEFAULT = 0
    HEALING = 1


class UnitAction:
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3
    ATTACK = 4
    IDLE = 5


class DefaultUnit(
    namedtuple(
        "DefaultUnit",
        ["transform", "rigidbody", "collider", "team", "pos_min", "pos_max", "status", "attacking"],
    )
):
    transform: (
        Transform  # The spatial transform (position, rotation) of the unit in the environment
    )
    rigidbody: RigidBody  # The physical properties of the unit for physics simulation (e.g., velocity, mass)
    collider: CircleCollider  # The collision shape and parameters for detecting overlaps with other objects
    team: chex.Array  # The team identifier to distinguish between different groups of units
    pos_min: chex.Array  # The positional boundaries or limits within which the unit can move
    pos_max: chex.Array  # The positional boundaries or limits within which the unit can move
    status: UnitStatus  # The current status of the unit, including health, attack stats, and other attributes
    attacking: (
        chex.Array
    )  # Boolean showing whether the unit is currently performing an attack TODO : is it needed?

    def to_array(self):
        return jnp.concatenate(
            (
                self.transform.to_array(),
                self.rigidbody.to_array(),
                self.collider.to_array(),
                self.team,
                self.status.to_array(),
                self.attacking,
            )
        )

    # def __new__(cls, transform, rigidbody, collider, team, pos_limit, status, attacking):
    #     return super().__new__(
    #         cls, transform, rigidbody, collider, team, pos_limit, status, attacking
    #     )

    def update(self, **kwargs):
        config = kwargs["config"]
        next_cooldown = jnp.where(self.attacking, 0.0, self.status.cooldown + config["dt"])
        updated_object = physics_update(config, self)

        updated_transform = self.transform._replace(
            position=jnp.clip(updated_object.transform.position, self.pos_min, self.pos_max)
        )

        return updated_object._replace(
            transform=updated_transform, status=self.status._replace(cooldown=next_cooldown)
        )

    def act(self, objects, action, **kwargs):
        # action : [rotate_angle, discrete action]

        discrete_action = action[1].astype(int).reshape()
        is_attack = UnitAction.ATTACK == discrete_action
        game_manager: GameManager = objects["game_manager"]
        target_id = game_manager.attack_target[self.status.id.reshape()]
        target_attackable = game_manager.attackable_matrix[self.status.id.reshape()]
        action_able = ~self.status.is_disabled & self.status.is_alive
        can_attack = (
            self.status.cooldown > self.status.attack_cooldown
        ) & action_able  # if unit is dead, do not attack

        notify(objects, "hit", (self, is_attack, target_id, target_attackable, can_attack))

        move_action = move_table[discrete_action] * action_able  # if unit is dead, do not move

        cooldown = is_attack & can_attack

        return self._replace(
            rigidbody=self.rigidbody._replace(velocity=move_action),
            transform=self.transform._replace(
                rotation=(self.transform.rotation + action[0].reshape() * action_able)
                % (2 * jnp.pi)
            ),
            status=self.status._replace(
                cooldown=jnp.clip(
                    0.0 * cooldown
                    + self.status.cooldown * (1 - cooldown),  # TODO: normalize between 0 and 1?
                    0.0,
                    self.status.attack_cooldown,
                )
            ),
        )

    def on_hit(self, objects, info):
        attacker: DefaultUnit
        attacker, is_attack, target_id, target_attackable, can_attack = info

        is_target = (
            (is_attack & (target_id == self.status.id))
            & can_attack
            & (target_attackable[self.status.id.reshape()])
        )

        # need to calculate cooldown of attacker
        is_attack_by_self = attacker.status.id == self.status.id

        damage = attacker.status.attack_damage * (
            attacker.status.attack_type == AttackType.DEFAULT
        ) - attacker.status.attack_damage * (attacker.status.attack_type == AttackType.HEALING)

        damaged_status = self.on_damage(damage)
        new_status = jax.tree.map(
            lambda x, y: jnp.where(is_target, y, x), self.status, damaged_status
        )

        return self._replace(status=new_status, attacking=is_attack_by_self)

    def on_damage(self, damage) -> UnitStatus:
        return self.status._replace(
            health=jnp.clip(self.status.health - damage, 0.0, self.status.max_health)
        )


class GameManager(
    namedtuple(
        "GameManager",
        [
            "reward",
            "done",
            "timestep",
            "attack_target",
            "attackable_matrix",
            "visible_matrix",
            "distance_matrix",
        ],
    )
):
    reward: chex.Array
    done: chex.Array
    timestep: chex.Array
    attack_target: chex.Array
    attackable_matrix: chex.Array
    visible_matrix: chex.Array
    distance_matrix: chex.Array

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
            attack_range_angle (float): Half-width of the attack cone in radians (default: pi/4)

        Returns:
            jnp.ndarray: [N, N] Boolean matrix where result[i][j] is True if unit i can attack/detect unit j
                        within its rectangular attack range, False otherwise.

        Note:
            The attack zone is shaped like a rectangle extending forward from each unit's position,
            with width determined by the attack_range_angle and length by the unit's attack_range.
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
        position_diff = (
            unit_position_vector[None] - unit_position_vector[:, None]
        )  # position_diff[i][j] = i'th unit's position - j'th unit's position
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
            n_u1_x[None] * rel_x + n_u1_y[None] * rel_y + unit_body_radius_vector[:, None] > 0
        )
        cond_upper = (
            -(n_u2_x[None] * rel_x + n_u2_y[None] * rel_y) + unit_body_radius_vector[:, None] > 0
        )
        fwd_x = jnp.cos(unit_rotation_vector)
        fwd_y = jnp.sin(unit_rotation_vector)
        cond_front = (fwd_x[None] * rel_x + fwd_y[None] * rel_y) + unit_body_radius_vector[None] < 0

        sight_inside = cond_lower & cond_upper & cond_front & ~unit_is_disabled_vector[:, None]
        attackable_matrix = (
            in_attack_range
            & (
                (~is_team.squeeze() & (unit_attack_type_vector == AttackType.DEFAULT))
                | (is_team.squeeze() & (unit_attack_type_vector == AttackType.HEALING))
            )
            & unit_alive_vector.T
        ) & ~unit_is_disabled_vector.T
        maksed_relative_distnace = (
            jnp.square(position_diff).sum(axis=-1)
            + 1e6 * jnp.identity(position_diff.shape[0])
            + 1e6 * (~attackable_matrix)
        )

        return self._replace(
            attack_target=maksed_relative_distnace.argmin(axis=1),
            attackable_matrix=attackable_matrix,
            visible_matrix=(sight_inside).squeeze().T,
            distance_matrix=position_diff,
        )

    def update(self, **kwargs):
        return self._replace(
            reward=jnp.array([0.0]),
            done=jnp.array([False]),
            timestep=self.timestep + 1,
        )


class TABS(BaseMAEnv):
    def __init__(
        self,
        num_agents: int = 4,
        physics_config: Dict[str, float] = EasyDict(
            {"dt": 0.2, "percent": 0.5, "slop": 0.01, "restitution": 0.8}
        ),
        obs_type: str = "unit_spec",
    ):
        super().__init__(num_agents, physics_config)
        self.obs_type = obs_type
        self.unit_keys = [f"unit_{i}" for i in range(num_agents)]

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
                ),
                attacking=jnp.array([False]),
            )
            for i, name in enumerate(self.unit_keys)
        }

        self.empty_state["game_manager"] = GameManager(
            attack_target=jnp.array([0]),
            attackable_matrix=jnp.array([[False]]),
            visible_matrix=jnp.array([[False]]),
            distance_matrix=jnp.array([[0]]),
            reward=jnp.array([0.0]),
            done=jnp.array([False]),
            timestep=jnp.array([0]),
        )

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

        return {key: state[key].status.id for key in self.unit_keys}

    def get_spec_obs(self, state):
        """
        own_feature : [health, max_health, absolute_x, absolute_y, rotation / 2pi, attack_range, attack_damage, cooldown, cooldown / attack_cooldown, body_radius, body_weight, sight_angle, is_alive]
        other_feature : [health, max_health, relative_x, relative_y, rotation / 2pi, attack_range, attack_damage, cooldown, cooldown / attack_cooldown, body_radius, body_weight, sight_angle, is_alive, is_ally, is_attackable]
        """

        keys = self.unit_keys

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

        is_ally = teams[None] == teams[:, None]

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
        rolled_is_ally = v_roll(is_ally, roll_shifts)[:, 1:]

        repeated_health = jnp.repeat(healths[None], repeats=n_unit, axis=0)
        repeated_max_health = jnp.repeat(max_healths[None], repeats=n_unit, axis=0)
        repeated_attack_damage = jnp.repeat(attack_damages[None], repeats=n_unit, axis=0)
        repeated_attack_range = jnp.repeat(attack_ranges[None], repeats=n_unit, axis=0)
        repeated_rotation = jnp.repeat(rotations[None], repeats=n_unit, axis=0)
        repeated_is_alive = jnp.repeat(is_alives[None], repeats=n_unit, axis=0)
        repeated_attack_cooldown = jnp.repeat(attack_cooldowns[None], repeats=n_unit, axis=0)
        repeated_radius = jnp.repeat(body_radiuss[None], repeats=n_unit, axis=0)
        repeated_mass = jnp.repeat(body_weights[None], repeats=n_unit, axis=0)
        repeated_sight_angle = jnp.repeat(sight_angles[None], repeats=n_unit, axis=0)
        repeated_cooldown = jnp.repeat(cooldowns[None], repeats=n_unit, axis=0)

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

        own_feature = jnp.concatenate(
            (
                healths,
                max_healths,
                positions,
                rotations,
                attack_ranges,
                attack_damages,
                cooldowns,
                attack_cooldowns,
                body_radiuss,
                body_weights,
                sight_angles,
                is_alives,
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
                ),
                axis=2,
            )
            * rolled_visible_matrix
        ).reshape(n_unit, -1)

        concated_obs = jnp.concatenate((own_feature, other_feature), axis=1)
        observations = {key: concated_obs[i] for i, key in enumerate(keys)}
        return observations

    def reset(self, key, senario: Scenario):
        vectorized_scenario: VectorizedScenario = get_vectorized_scenario(
            senario, self.num_agents // 2
        )

        state = {}
        for i, unit in enumerate(self.unit_keys):
            state[unit] = DefaultUnit(
                transform=Transform(
                    position=vectorized_scenario.positions[i],
                    rotation=vectorized_scenario.rotations[i],
                ),
                rigidbody=RigidBody(
                    mass=vectorized_scenario.body_weights[i],
                    velocity=jnp.array([0.0, 0.0]),
                    acceleration=jnp.array([0.0, 0.0]),
                    is_kinematic=jnp.array([False]),
                ),
                collider=CircleCollider(radius=vectorized_scenario.body_radiuss[i]),
                team=vectorized_scenario.teams[i],
                pos_min=vectorized_scenario.pos_min[i],
                pos_max=vectorized_scenario.pos_max[i],
                status=self.empty_state[unit].status._replace(
                    unit_id=vectorized_scenario.unit_ids[i],
                    health=vectorized_scenario.healths[i],
                    attack_damage=vectorized_scenario.attack_damages[i],
                    attack_range=vectorized_scenario.attack_ranges[i],
                    attack_cooldown=vectorized_scenario.attack_cooldowns[i],
                    cooldown=vectorized_scenario.attack_cooldowns[i] * 0.0,
                    sight_angle=vectorized_scenario.sight_angles[i],
                    is_alive=vectorized_scenario.is_alive[i],
                    is_disabled=vectorized_scenario.is_disabled[i],
                    attack_type=vectorized_scenario.attack_types[i],
                    max_health=vectorized_scenario.healths[i],
                ),
                attacking=jnp.array([False]),
            )
        state["game_manager"] = GameManager(
            attack_target=jnp.array([0]),
            attackable_matrix=jnp.array([[False]]),
            visible_matrix=jnp.array([[False]]),
            distance_matrix=jnp.array([[0]]),
            reward=jnp.array([0.0]),
            done=jnp.array([False]),
            timestep=jnp.array([0]),
        )

        state["game_manager"] = state["game_manager"].update_distance_matrix(state, self.unit_keys)

        return self.get_obs(state), state

    def step(self, key, state, action):
        state["game_manager"] = state["game_manager"].update_distance_matrix(state, self.unit_keys)

        for sprite in state.keys():
            if hasattr(state[sprite], "update"):
                state[sprite] = state[sprite].update(config=self.physics_config)

        collider_filter = {
            unit: [key for key in state if "unit" in key and key != unit] for unit in self.unit_keys
        }

        state = physics_step(self.physics_config, state, list(state.keys()), collider_filter)
        # action processing
        units = [key for key in state if "unit" in key]

        for sprite in units:
            state[sprite] = state[sprite].act(state, action[sprite])

        # alive processing after action step, for independent unit sequence
        for sprite in units:
            state[sprite] = state[sprite]._replace(
                status=state[sprite].status._replace(is_alive=(state[sprite].status.health > 0))
            )

        return self.get_obs(state), state, 0.0, False, {"timestep": 0}

    def render(self, state):
        return None
