from collections import namedtuple
import chex
from src.maenv.util import notify
import jax
import jax.numpy as jnp
from src.maenv.physics import Transform, RigidBody, CircleCollider, physics_update


class UnitStatus(
    namedtuple(
        "UnitStatus",
        [
            "id",
            "health",
            "attack_damage",
            "attack_range",
            "attack_cooldown",
            "cooldown",
            "sight_angle",
        ],
    )
):
    id: chex.Array  # Unique identifier for the unit
    health: chex.Array  # Current health points of the unit
    attack_damage: chex.Array  # Damage dealt by the unit's attacks
    attack_range: chex.Array  # Maximum distance the unit can attack
    attack_cooldown: chex.Array  # Required cooldown time between attacks
    cooldown: chex.Array  # Time elapsed since the most recent attack
    sight_angle: chex.Array  # Field of view angle in radians


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
        ["transform", "rigidbody", "collider", "team", "pos_limit", "status", "attacking"],
    )
):
    transform: (
        Transform  # The spatial transform (position, rotation) of the unit in the environment
    )
    rigidbody: RigidBody  # The physical properties of the unit for physics simulation (e.g., velocity, mass)
    collider: CircleCollider  # The collision shape and parameters for detecting overlaps with other objects
    team: chex.Array  # The team identifier to distinguish between different groups of units
    pos_limit: chex.Array  # The positional boundaries or limits within which the unit can move
    status: UnitStatus  # The current status of the unit, including health, attack stats, and other attributes
    attacking: chex.Array  # Boolean showing whether the unit is currently performing an attack

    def __new__(cls, transform, rigidbody, collider, team, pos_limit, status, attacking):
        return super().__new__(
            cls, transform, rigidbody, collider, team, pos_limit, status, attacking
        )

    def update(self, **kwargs):
        config = kwargs["config"]
        next_cooldown = jnp.where(self.attacking, 0.0, self.status.cooldown + config["dt"])
        updated_object = physics_update(config, self)
        return updated_object._replace(status=self.status._replace(cooldown=next_cooldown))

    def act(self, objects, action, **kwargs):
        # action : [rotate_angle, discrete action]

        discrete_action = action[1].astype(int).reshape()
        is_attack = UnitAction.ATTACK == discrete_action
        game_manager: GameManager = objects["game_manager"]
        target = game_manager.attack_target[self.status.id.reshape()]
        can_attack = self.status.cooldown > self.status.attack_cooldown

        notify(objects, "hit", (self, is_attack, target, can_attack))

        move_action = move_table[discrete_action]

        cooldown = is_attack & can_attack

        return self._replace(
            rigidbody=self.rigidbody._replace(velocity=move_action),
            transform=self.transform._replace(
                rotation=(self.transform.rotation + action[0].reshape()) % (2 * jnp.pi)
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
        attacker, is_attack, target, can_attack = info

        print(is_attack, target, can_attack)

        # print(target.shape)

        is_target = (
            (is_attack & (target[self.status.id.reshape()].reshape()))
            & can_attack
            & (attacker.status.health > 0)
        )

        # need to calculate cooldown of attacker
        is_attack_by_self = attacker.status.id == self.status.id

        damaged_status = self.on_damage(attacker.status.attack_damage)
        new_status = jax.tree.map(
            lambda x, y: jnp.where(is_target, y, x), self.status, damaged_status
        )

        return self._replace(status=new_status, attacking=is_attack_by_self)

    def on_damage(self, damage) -> UnitStatus:
        return self.status._replace(health=self.status.health - damage)


class GameManager(
    namedtuple(
        "GameManager",
        ["reward", "done", "timestep", "attack_target", "visible_matrix", "distance_matrix"],
    )
):
    reward: chex.Array
    done: chex.Array
    timestep: chex.Array
    attack_target: chex.Array
    visible_matrix: chex.Array
    distance_matrix: chex.Array

    def get_units_in_attack_range(
        self,
        position_diff,
        unit_rotation_vector,
        unit_body_radius_vector,
        unit_attack_range_vector,
        attack_range_angle=jnp.pi / 4,
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

        cos_attack_range_half_angle = jnp.cos(attack_range_angle / 2)
        sin_attack_range_half_angle = jnp.sin(attack_range_angle / 2)

        local_unit_p2 = (
            jnp.array([[cos_attack_range_half_angle, sin_attack_range_half_angle]])
            * unit_body_radius_vector
        )
        local_unit_p1 = (
            jnp.array([[cos_attack_range_half_angle, -sin_attack_range_half_angle]])
            * unit_body_radius_vector
        )

        local_height = local_unit_p1 - local_unit_p2

        height = jnp.linalg.norm(local_height, axis=-1, keepdims=True)[None]
        width = unit_attack_range_vector[:, None]

        unit_cosine_vector = jnp.cos(unit_rotation_vector)[:, None]
        unit_sine_vector = jnp.sin(unit_rotation_vector)[:, None]

        relative_unit_x = position_diff[:, :, 0:1]
        relative_unit_y = position_diff[:, :, 1:2]
        local_unit_x = (
            relative_unit_x * unit_cosine_vector + relative_unit_y * unit_sine_vector
        )  # rotate -theta to get local coordinate
        local_unit_y = -relative_unit_x * unit_sine_vector + relative_unit_y * unit_cosine_vector

        rx = local_unit_p2[:, 0:1]
        ry = local_unit_p2[:, 1:2]

        closest_x = jnp.clip(local_unit_x, rx[:, None], rx[:, None] + width)
        closest_y = jnp.clip(local_unit_y, -ry[:, None], ry[:, None])

        dx = local_unit_x - closest_x
        dy = local_unit_y - closest_y

        collision = dx**2 + dy**2 < unit_body_radius_vector**2

        available_target = (collision).squeeze(-1) & (
            ~jnp.identity(position_diff.shape[0], dtype=jnp.bool)
        )  # exclude self

        return available_target

    def update_distance_matrix(self, objects):
        unit_position_vector = jnp.stack(
            [unit.transform.position for (key, unit) in objects.items() if "unit" in key]
        )
        unit_rotation_vector = jnp.stack(
            [unit.transform.rotation for (key, unit) in objects.items() if "unit" in key]
        )
        unit_body_radius_vector = jnp.stack(
            [unit.collider.radius for (key, unit) in objects.items() if "unit" in key]
        )
        unit_attack_range_vector = jnp.stack(
            [unit.status.attack_range for (key, unit) in objects.items() if "unit" in key]
        )

        unit_team_vector = jnp.stack(
            [unit.team for (key, unit) in objects.items() if "unit" in key]
        )
        unit_sight_angle_vector = jnp.stack(
            [unit.status.sight_angle for (key, unit) in objects.items() if "unit" in key]
        )

        position_diff = (
            unit_position_vector[None] - unit_position_vector[:, None]
        )  # position_diff[i][j] = i'th unit's position - j'th unit's position
        is_team = unit_team_vector[None] == unit_team_vector[:, None]
        in_attack_range = self.get_units_in_attack_range(
            position_diff, unit_rotation_vector, unit_body_radius_vector, unit_attack_range_vector
        )

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
            -(n_u2_x[None] * rel_x + n_u2_y[None] * rel_y) + unit_body_radius_vector[:, None] >= 0
        )

        sight_inside = cond_lower & cond_upper

        return self._replace(
            attack_target=(in_attack_range & ~is_team.squeeze()),
            visible_matrix=(sight_inside).squeeze().T,
            distance_matrix=position_diff,
        )

    def update(self, **kwargs):
        return self._replace(
            reward=jnp.array([0.0]),
            done=jnp.array([False]),
            timestep=self.timestep + 1,
        )
