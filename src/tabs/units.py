from typing import Dict

import chex
import jax.numpy as jnp
from flax import struct

from src.tabs.constants import ALL_UNIT_NAMES, UnitID, SIGHT_ANGLE


@struct.dataclass
class Unit:
    unit_type_id: chex.Array
    price: chex.Array
    health: chex.Array
    body_radius: chex.Array
    body_weight: chex.Array
    speed: chex.Array
    attack_damage: chex.Array
    attack_range: chex.Array  # WM
    attack_cooldown: chex.Array  # sec
    sight_angle: chex.Array
    space_occupied: chex.Array  # area of rectangle shape


def get_unit_spec(unit: Unit) -> Dict[str, chex.Array]:
    return {
        "price": unit.price,
        "health": unit.health,
        "body_radius": unit.body_radius,
        "body_weight": unit.body_weight,
        "speed": unit.speed,
        "attack_damage": unit.attack_damage,
        "attack_range": unit.attack_range,
        "attack_cooldown": unit.attack_cooldown,
        "sight_angle": unit.sight_angle,
        "space_occupied": unit.space_occupied,
    }


def get_all_unit_spec_dict() -> Dict[str, Dict[str, chex.Array]]:
    all_units = [Farmer, Assassin, TheKing, Mammoth, Archer, Cannon, Deadeye, Healer, Paladin]
    all_unit_names = ALL_UNIT_NAMES
    spec = {}
    for idx, unit in enumerate(all_units):
        spec[all_unit_names[idx]] = get_unit_spec(unit)

    return spec


def get_all_unit_spec() -> chex.Array:
    """
    Return all predefined unit specifications.
    """
    all_units = [Farmer, Assassin, TheKing, Mammoth, Archer, Cannon, Deadeye, Healer, Paladin]

    prices = jnp.array([unit.price for unit in all_units]).flatten()
    healths = jnp.array([unit.health for unit in all_units]).flatten()
    body_radiuses = jnp.array([unit.body_radius for unit in all_units]).flatten()
    body_weights = jnp.array([unit.body_weight for unit in all_units]).flatten()
    speeds = jnp.array([unit.speed for unit in all_units]).flatten()
    attack_damages = jnp.array([unit.attack_damage for unit in all_units]).flatten()
    attack_ranges = jnp.array([unit.attack_range for unit in all_units]).flatten()
    attack_cooldown = jnp.array([unit.attack_cooldown for unit in all_units]).flatten()
    sight_angles = jnp.array([unit.sight_angle for unit in all_units]).flatten()
    space_occupied = jnp.array([unit.space_occupied for unit in all_units]).flatten()

    return {
        "prices": prices,
        "healths": healths,
        "body_radiuses": body_radiuses,
        "body_weights": body_weights,
        "speeds": speeds,
        "attack_damages": attack_damages,
        "attack_ranges": attack_ranges,
        "attack_cooldown": attack_cooldown,
        "sight_angles": sight_angles,
        "space_occupied": space_occupied,
    }


@struct.dataclass
class Farmer(Unit):
    unit_type_id = jnp.array([UnitID.Farmer])
    price = jnp.array([120])
    health = jnp.array([60])
    body_radius = jnp.array([1.0])
    body_weight = jnp.array([1.0])
    speed = jnp.array([1.1])
    attack_damage = jnp.array([14])
    attack_range = jnp.array([2.5])
    attack_cooldown = jnp.array([2.5])
    sight_angle = jnp.array([SIGHT_ANGLE])
    space_occupied = jnp.array([1])


@struct.dataclass
class Assassin(Unit):
    unit_type_id = jnp.array([UnitID.Assassin])
    price = jnp.array([430])
    health = jnp.array([70])
    body_radius = jnp.array([1.0])
    body_weight = jnp.array([1.0])
    speed = jnp.array([1.4])
    attack_damage = jnp.array([22])
    attack_range = jnp.array([2.5])
    attack_cooldown = jnp.array([1.5])
    sight_angle = jnp.array([SIGHT_ANGLE])
    space_occupied = jnp.array([1])


@struct.dataclass
class TheKing(Unit):
    unit_type_id = jnp.array([UnitID.TheKing])
    price = jnp.array([1320])
    health = jnp.array([346])
    body_radius = jnp.array([1.47])
    body_weight = jnp.array([10.0])
    speed = jnp.array([1.2])
    attack_damage = jnp.array([46])
    attack_range = jnp.array([3.2])
    attack_cooldown = jnp.array([2.5])
    sight_angle = jnp.array([SIGHT_ANGLE])
    space_occupied = jnp.array([1])


@struct.dataclass
class Mammoth(Unit):
    unit_type_id = jnp.array([UnitID.Mammoth])
    price = jnp.array([980])
    health = jnp.array([685])
    body_radius = jnp.array([4.25])
    body_weight = jnp.array([50.0])
    speed = jnp.array([1.2])
    attack_damage = jnp.array([20])
    attack_range = jnp.array([3])
    attack_cooldown = jnp.array([6.5])
    sight_angle = jnp.array([SIGHT_ANGLE])
    space_occupied = jnp.array([4])


@struct.dataclass
class Archer(Unit):
    unit_type_id = jnp.array([UnitID.Archer])
    price = jnp.array([450])
    health = jnp.array([40])
    body_radius = jnp.array([1.0])
    body_weight = jnp.array([1.0])
    speed = jnp.array([1.0])
    attack_damage = jnp.array([28])
    attack_range = jnp.array([27.0])
    attack_cooldown = jnp.array([8.0])
    sight_angle = jnp.array([SIGHT_ANGLE])
    space_occupied = jnp.array([1])


@struct.dataclass
class Cannon(Unit):
    unit_type_id = jnp.array([UnitID.Cannon])
    price = jnp.array([1080])
    health = jnp.array([100])
    body_radius = jnp.array([1.0])
    body_weight = jnp.array([5.2])
    speed = jnp.array([0.5])
    attack_damage = jnp.array([80])
    attack_range = jnp.array([40.0])
    attack_cooldown = jnp.array([10.0])
    sight_angle = jnp.array([SIGHT_ANGLE])
    space_occupied = jnp.array([1])


@struct.dataclass
class Deadeye(Unit):
    unit_type_id = jnp.array([UnitID.Deadeye])
    price = jnp.array([470])
    health = jnp.array([40])
    body_radius = jnp.array([1.0])
    body_weight = jnp.array([1.0])
    speed = jnp.array([1.1])
    attack_damage = jnp.array([25])
    attack_range = jnp.array([20.0])
    attack_cooldown = jnp.array([8.0])
    sight_angle = jnp.array([SIGHT_ANGLE])
    space_occupied = jnp.array([1])


@struct.dataclass
class Healer(Unit):
    unit_type_id = jnp.array([UnitID.Healer])
    price = jnp.array([190])
    health = jnp.array([25])
    body_radius = jnp.array([1.0])
    body_weight = jnp.array([1.0])
    speed = jnp.array([1.0])
    attack_damage = jnp.array([-7])
    attack_range = jnp.array([10.0])
    attack_cooldown = jnp.array([2.0])
    sight_angle = jnp.array([SIGHT_ANGLE])
    space_occupied = jnp.array([1])


@struct.dataclass
class Paladin(Unit):
    unit_type_id = jnp.array([UnitID.Paladin])
    price = jnp.array([230])
    health = jnp.array([220])
    body_radius = jnp.array([1.32])
    body_weight = jnp.array([8.5])
    speed = jnp.array([1.2])
    attack_damage = jnp.array([-6])
    attack_range = jnp.array([7.5])
    attack_cooldown = jnp.array([2.0])
    sight_angle = jnp.array([SIGHT_ANGLE])
    space_occupied = jnp.array([1])
