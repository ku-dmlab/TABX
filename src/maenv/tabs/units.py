from typing import Dict, List

import chex
from flax import struct
import jax.numpy as jnp

sight_angle = 60.0
sight_radius = 10.0
non_target_balance = 0.5


@struct.dataclass
class Unit:
    unit_type_id: chex.Array
    price: chex.Array
    health: chex.Array
    body_radius: chex.Array
    body_weight: chex.Array
    velocity: chex.Array
    attack_damage: chex.Array
    attack_range: chex.Array  # WM
    attack_cooldown: chex.Array  # sec
    sight_angle: chex.Array
    sight_radius: chex.Array
    alive: chex.Array
    team: chex.Array  # 1: alley, 0: enemy


def get_all_unit_names() -> List:
    return ["farmer", "archer", "theking", "bombthrower", "mammoth", "deadeye", "healer"]


def get_unit_spec(unit: Unit) -> Dict[str, chex.Array]:
    return {
        "price": unit.price,
        "health": unit.health,
        "body_radius": unit.body_radius,
        "body_weight": unit.body_weight,
        "velocity": unit.velocity,
        "attack_damage": unit.attack_damage,
        "attack_range": unit.attack_range,
        "attack_cooldown": unit.attack_cooldown,
        "sight_angle": unit.sight_angle,
        "sight_radius": unit.sight_radius,
    }


def get_all_unit_spec_dict() -> Dict[str, Dict[str, chex.Array]]:
    all_units = [Farmer, Archer, TheKing, BombThrower, Mammoth, Deadeye, Healer]
    all_unit_names = get_all_unit_names()
    spec = {}
    for idx, unit in enumerate(all_units):
        spec[all_unit_names[idx]] = get_unit_spec(unit)

    return spec


def get_all_unit_spec() -> chex.Array:
    all_units = [Farmer, Archer, TheKing, BombThrower, Mammoth, Deadeye, Healer]

    prices = jnp.array([unit.price for unit in all_units]).flatten()
    healths = jnp.array([unit.health for unit in all_units]).flatten()
    body_radiuses = jnp.array([unit.body_radius for unit in all_units]).flatten()
    body_weights = jnp.array([unit.body_weight for unit in all_units]).flatten()
    velocities = jnp.array([unit.velocity for unit in all_units]).flatten()
    attack_damages = jnp.array([unit.attack_damage for unit in all_units]).flatten()
    attack_cooldown = jnp.array([unit.attack_cooldown for unit in all_units]).flatten()
    sight_angles = jnp.array([unit.sight_angle for unit in all_units]).flatten()
    sight_radiuses = jnp.array([unit.sight_radius for unit in all_units]).flatten()

    return jnp.vstack(
        (
            prices,
            healths,
            body_radiuses,
            body_weights,
            velocities,
            attack_damages,
            attack_cooldown,
            sight_angles,
            sight_radiuses,
        )
    )


class UnitID:
    Farmer = 0
    Archer = 1
    TheKing = 2
    BombThrower = 3
    Mammoth = 4
    Deadeye = 5
    Healer = 6


@struct.dataclass
class Farmer(Unit):
    unit_type_id = jnp.array([UnitID.Farmer])
    price = jnp.array([80])
    health = jnp.array([60])
    body_radius = jnp.array([1.0])
    body_weight = jnp.array([1.0])
    velocity = jnp.array([1.0])
    attack_damage = jnp.array([60])
    attack_range = jnp.array([2.5])
    attack_cooldown = jnp.array([2.5])
    sight_angle = jnp.array([sight_angle])
    sight_radius = jnp.array([sight_radius])
    alive = jnp.array([1])
    team = jnp.array([1])


@struct.dataclass
class Archer(Unit):
    unit_type_id = jnp.array([UnitID.Archer])
    price = jnp.array([140])
    health = jnp.array([40])
    body_radius = jnp.array([1.0])
    body_weight = jnp.array([1.0])
    velocity = jnp.array([1.0])
    attack_damage = jnp.array([190])
    attack_range = jnp.array([30.0])
    attack_cooldown = jnp.array([8])
    sight_angle = jnp.array([sight_angle])
    sight_radius = jnp.array([sight_radius])
    alive = jnp.array([1])
    team = jnp.array([1])


@struct.dataclass
class TheKing(Unit):
    unit_type_id = jnp.array([UnitID.TheKing])
    price = jnp.array([1500])
    health = jnp.array([2377])
    body_radius = jnp.array([1.47])
    body_weight = jnp.array([10.0])
    velocity = jnp.array([1.0])
    attack_damage = jnp.array([330])
    attack_range = jnp.array([3.2])
    attack_cooldown = jnp.array([2.5])
    sight_angle = jnp.array([sight_angle])
    sight_radius = jnp.array([sight_radius])
    alive = jnp.array([1])
    team = jnp.array([1])


@struct.dataclass
class BombThrower(Unit):
    unit_type_id = jnp.array([UnitID.BombThrower])
    price = jnp.array([250])
    health = jnp.array([150])
    body_radius = jnp.array([1.0])
    body_weight = jnp.array([1.0])
    velocity = jnp.array([1.0])
    attack_damage = jnp.array([160])
    attack_range = jnp.array([15])
    attack_cooldown = jnp.array([10.0])
    sight_angle = jnp.array([sight_angle])
    sight_radius = jnp.array([sight_radius])
    alive = jnp.array([1])
    team = jnp.array([1])


@struct.dataclass
class Mammoth(Unit):
    unit_type_id = jnp.array([UnitID.Mammoth])
    price = jnp.array([2200])
    health = jnp.array([2526])
    body_radius = jnp.array([4.25])
    body_weight = jnp.array([50.0])
    velocity = jnp.array([1.2])
    attack_damage = jnp.array([100])  # NOTE: arbitrary value
    attack_range = jnp.array([3])
    attack_cooldown = jnp.array([4.0])
    sight_angle = jnp.array([sight_angle])
    sight_radius = jnp.array([sight_radius])
    alive = jnp.array([1])
    team = jnp.array([1])


@struct.dataclass
class Deadeye(Unit):
    unit_type_id = jnp.array([UnitID.Deadeye])
    price = jnp.array([900])
    health = jnp.array([75])
    body_radius = jnp.array([1.0])
    body_weight = jnp.array([1.0])
    velocity = jnp.array([1.0])
    attack_damage = jnp.array([650])
    attack_range = jnp.array([40])
    attack_cooldown = jnp.array([5.0])
    sight_angle = jnp.array([sight_angle])
    sight_radius = jnp.array([sight_radius])
    alive = jnp.array([1])
    team = jnp.array([1])


@struct.dataclass
class Healer(Unit):
    unit_type_id = jnp.array([UnitID.Healer])
    price = jnp.array([180])
    health = jnp.array([25])
    body_radius = jnp.array([1.0])
    body_weight = jnp.array([1.0])
    velocity = jnp.array([1.0])
    attack_damage = jnp.array([35])
    attack_range = jnp.array([10.0])
    attack_cooldown = jnp.array([1])
    sight_angle = jnp.array([sight_angle])
    sight_radius = jnp.array([sight_radius + 1])
    alive = jnp.array([1])
    team = jnp.array([1])
