import chex
from flax import struct


@struct.dataclass
class VectorizedScenario:
    positions: chex.Array
    rotations: chex.Array
    body_weights: chex.Array
    body_radiuss: chex.Array
    teams: chex.Array
    pos_min: chex.Array
    pos_max: chex.Array
    unit_ids: chex.Array
    healths: chex.Array
    attack_damages: chex.Array
    attack_ranges: chex.Array
    attack_cooldowns: chex.Array
    sight_angles: chex.Array
    is_alive: chex.Array
    attack_types: chex.Array
    is_disabled: chex.Array
    speeds: chex.Array


@struct.dataclass
class ZoneScenario:
    n_zone: chex.Array
    zone_type: chex.Array
    position: chex.Array
    axes: chex.Array
    effect_value: chex.Array


@struct.dataclass
class UnitScenario:
    ally_unit_comp: chex.Array
    enemy_unit_comp: chex.Array
    unit_comp_mask: chex.Array
    battle_field: chex.Array
    battle_field_mask: chex.Array
    enemy_battle_field: chex.Array
    enemy_battle_field_mask: chex.Array
    # unit spec
    health: chex.Array
    body_radius: chex.Array
    body_weight: chex.Array
    speed: chex.Array
    attack_damage: chex.Array
    attack_range: chex.Array  # WM
    attack_cooldown: chex.Array  # sec
    sight_angle: chex.Array
    space_occupied: chex.Array  # area of rectangle shape
