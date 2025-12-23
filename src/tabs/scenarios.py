import functools
import itertools
from typing import Tuple

import chex
import jax
import jax.numpy as jnp
from flax import struct

from src.tabs.config import TABSConfig
from src.tabs.constants import PREDEFINED_SCENARIOS, SCENARIOS, UNITID2CHAR, ZONESCENARIO, UnitID
from src.tabs.units import get_all_unit_spec


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
    damage: chex.Array


@struct.dataclass
class Scenario:
    ally_unit_comp: chex.Array
    enemy_unit_comp: chex.Array
    battle_field: chex.Array
    enemy_battle_field: chex.Array
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


@struct.dataclass
class UnitScenario:
    budget: int
    ally_unit_comp: chex.Array
    enemy_unit_comp: chex.Array
    unit_comp_mask: chex.Array
    battle_field: chex.Array
    battle_field_mask: chex.Array
    enemy_battle_field: chex.Array
    enemy_battle_field_mask: chex.Array
    # unit spec
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


def get_scenario_list():
    name_list = []
    for name, zname in itertools.product(SCENARIOS, ZONESCENARIO):
        if zname == "void":
            name_list.append(name)
        else:
            name_list.append(name + "_" + zname)

    return name_list


def generate_scenario_config(
    scenario_name: str,
    max_n_ally: int = None,
    max_n_enemy: int = None,
    max_n_zone: int = None,
) -> Tuple[VectorizedScenario, ZoneScenario, TABSConfig]:
    tabs_config = TABSConfig(scenario_name=scenario_name)
    h, w = tabs_config.max_field_height, tabs_config.max_field_width
    n_zone = tabs_config.max_n_zone if max_n_zone is None else max_n_zone

    battle_field = jnp.zeros((h, w), dtype=jnp.float32)
    enemy_battle_field = jnp.zeros_like(battle_field)

    all_spec = get_all_unit_spec()
    assert len(all_spec["healths"]) <= tabs_config.max_num_units
    m = tabs_config.max_num_units - len(all_spec["healths"])
    if m > 0:
        unit_comp_mask = unit_comp_mask.at[-m:].set(0)

    health = jnp.concatenate((all_spec["healths"], jnp.zeros(m)))
    body_radius = jnp.concatenate((all_spec["body_radiuses"], jnp.zeros(m)))
    body_weight = jnp.concatenate((all_spec["body_weights"], jnp.zeros(m)))
    speed = jnp.concatenate((all_spec["speeds"], jnp.zeros(m)))
    attack_damage = jnp.concatenate((all_spec["attack_damages"], jnp.zeros(m)))
    attack_range = jnp.concatenate((all_spec["attack_ranges"], jnp.zeros(m)))
    attack_cooldown = jnp.concatenate((all_spec["attack_cooldown"], jnp.zeros(m)))
    sight_angle = jnp.concatenate((all_spec["sight_angles"], jnp.zeros(m)))
    space_occupied = jnp.concatenate((all_spec["space_occupied"], jnp.zeros(m)))

    zone_type = jnp.zeros((n_zone, 1))
    position = jnp.zeros((n_zone, 2))
    axes = jnp.zeros((n_zone, 2))
    damage = jnp.zeros((n_zone, 1))

    scenario_name = tabs_config.scenario_name.split("_")[0]
    if "_" in tabs_config.scenario_name:
        zone_scenario_name = tabs_config.scenario_name.split("_")[1]
    else:
        if scenario_name in PREDEFINED_SCENARIOS:
            zone_scenario_name = scenario_name
        else:
            zone_scenario_name = "void"

    # Unit Scenario
    if scenario_name == SCENARIOS[0]:
        _battle_field = jnp.array(
            [
                [0, 0, UnitID.Mammoth, UnitID.Farmer, 0],
                [UnitID.Healer, UnitID.Archer, UnitID.Archer, UnitID.Archer, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
            ],
            dtype=jnp.float32,
        )
        _enemy_battle_field = jnp.array(
            [
                [0, UnitID.Farmer, UnitID.TheKing, UnitID.Farmer, 0],
                [0, 0, UnitID.Healer, 0, 0],
                [UnitID.Archer, 0, 0, 0, UnitID.Archer],
                [0, 0, 0, 0, 0],
            ],
            dtype=jnp.float32,
        )
    elif scenario_name == SCENARIOS[1]:
        _battle_field = jnp.array(
            [
                [UnitID.Farmer, UnitID.Farmer, UnitID.Mammoth, 0, 0],
                [0, UnitID.Archer, 0, UnitID.Archer, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
            ],
            dtype=jnp.float32,
        )
        _enemy_battle_field = jnp.array(
            [
                [0, 0, UnitID.TheKing, 0, 0],
                [UnitID.Assassin, 0, 0, 0, UnitID.Assassin],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
            ],
            dtype=jnp.float32,
        )
    elif scenario_name == SCENARIOS[2]:
        _battle_field = jnp.array(  # 3360
            [
                [UnitID.Farmer, UnitID.Farmer, UnitID.TheKing, UnitID.Farmer, UnitID.Farmer],
                [0, UnitID.Archer, UnitID.Paladin, UnitID.Archer, UnitID.Assassin],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
            ],
            dtype=jnp.float32,
        )
        _enemy_battle_field = jnp.array(
            [
                [0, 0, UnitID.Mammoth, 0, 0],
                [0, 0, 0, 0, 0],
                [0, UnitID.Cannon, 0, UnitID.Cannon, 0],
                [0, 0, UnitID.Paladin, 0, 0],
            ],
            dtype=jnp.float32,
        )
    elif scenario_name == SCENARIOS[3]:
        _battle_field = jnp.array(  # 1950
            [
                [UnitID.Farmer, UnitID.Farmer, UnitID.Farmer, UnitID.Farmer, UnitID.Farmer],
                [UnitID.Assassin, 0, UnitID.Archer, 0, UnitID.Deadeye],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
            ],
            dtype=jnp.float32,
        )
        _enemy_battle_field = jnp.array(
            [
                [UnitID.Farmer, UnitID.Farmer, UnitID.Farmer, UnitID.Farmer, UnitID.Farmer],
                [0, UnitID.Farmer, 0, UnitID.Farmer, 0],
                [UnitID.Deadeye, 0, 0, 0, UnitID.Deadeye],
                [UnitID.Healer, 0, 0, 0, 0],
            ],
            dtype=jnp.float32,
        )
    elif scenario_name == PREDEFINED_SCENARIOS[0]:
        _battle_field = jnp.array(
            [
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, UnitID.Deadeye, 0, UnitID.Deadeye, 0],
                [0, 0, UnitID.Deadeye, 0, 0],
            ],
            dtype=jnp.float32,
        )
        _enemy_battle_field = jnp.array(
            [
                [0, UnitID.Farmer, UnitID.Farmer, UnitID.Farmer, 0],
                [0, UnitID.Farmer, UnitID.Farmer, UnitID.Farmer, 0],
                [0, UnitID.Farmer, UnitID.Farmer, UnitID.Farmer, 0],
                [0, 0, 0, 0, 0],
            ],
            dtype=jnp.float32,
        )
    elif scenario_name == PREDEFINED_SCENARIOS[1]:
        _battle_field = jnp.array(
            [
                [0, 0, 0, 0, 0],
                [0, 0, UnitID.Mammoth, 0, 0],
                [0, UnitID.Archer, 0, UnitID.Archer, 0],
                [0, 0, 0, 0, 0],
            ],
            dtype=jnp.float32,
        )
        _enemy_battle_field = jnp.array(
            [
                [0, UnitID.TheKing, UnitID.TheKing, UnitID.TheKing, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
            ],
            dtype=jnp.float32,
        )
    elif scenario_name == PREDEFINED_SCENARIOS[2]:
        _battle_field = jnp.array(
            [
                [0, UnitID.Cannon, 0, UnitID.Cannon, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
            ],
            dtype=jnp.float32,
        )
        _enemy_battle_field = jnp.array(
            [
                [0, 0, UnitID.TheKing, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
            ],
            dtype=jnp.float32,
        )
    else:
        raise NotImplementedError

    battle_field = battle_field.at[:h, :w].set(_battle_field)
    enemy_battle_field = enemy_battle_field.at[:h, :w].set(_enemy_battle_field)

    ally_unit_comp = jax.vmap(lambda x: (battle_field == (x + 1)).sum())(
        jnp.arange(tabs_config.max_num_units)
    )
    enemy_unit_comp = jax.vmap(lambda x: (enemy_battle_field == (x + 1)).sum())(
        jnp.arange(tabs_config.max_num_units)
    )

    scenario: Scenario = Scenario(
        ally_unit_comp=ally_unit_comp,
        enemy_unit_comp=enemy_unit_comp,
        battle_field=battle_field,
        enemy_battle_field=enemy_battle_field,
        health=health,
        body_radius=body_radius,
        body_weight=body_weight,
        speed=speed,
        attack_damage=attack_damage,
        attack_range=attack_range,
        attack_cooldown=attack_cooldown,
        sight_angle=sight_angle,
        space_occupied=space_occupied,
    )
    vscenario: VectorizedScenario = get_vectorized_scenario(
        scenario=scenario,
        n_ally=ally_unit_comp.sum() if max_n_ally is None else max_n_ally,
        n_enemy=enemy_unit_comp.sum() if max_n_enemy is None else max_n_enemy,
    )

    # Zone Scenario
    if zone_scenario_name == ZONESCENARIO[0]:  # void
        zone_type = jnp.zeros((n_zone, 1), dtype=jnp.int32)
        position = jnp.zeros((n_zone, 2))
        axes = jnp.zeros((n_zone, 2))
        damage = jnp.zeros((n_zone, 1))
    elif zone_scenario_name == "elbow":
        zone_type = jnp.array([1, 1]).reshape(-1, 1)
        position = jnp.array([[-24.0, -3.0], [-16.0, -12.5]]).reshape(-1, 2)
        axes = jnp.array([[7.5, 3.0], [3.0, 10.0]]).reshape(-1, 2)
        damage = jnp.array([10.0, 10.0]).reshape(-1, 1)
    elif zone_scenario_name == "crossfire":
        zone_type = jnp.array([1, 1, 1, 1]).reshape(-1, 1)
        position = jnp.array([[-3.25, -6.0], [-3.25, 6.0], [4.0, 15.0], [4.0, -15.0]]).reshape(
            -1, 2
        )
        axes = jnp.array([[7.5, 3.0], [7.5, 3.0], [3.0, 9.0], [3.0, 9.0]]).reshape(-1, 2)
        damage = jnp.array([10.0, 10.0, 10.0, 10.0]).reshape(-1, 1)
    elif zone_scenario_name == "ambush":
        zone_type = jnp.array([2, 2]).reshape(-1, 1)
        position = jnp.array([[-24.25, -10.0], [-24, 10.0]]).reshape(-1, 2)
        axes = jnp.array([[3.0, 3.0], [3.0, 3.0]]).reshape(-1, 2)
        damage = jnp.array([0.0, 0.0]).reshape(-1, 1)
    elif zone_scenario_name == ZONESCENARIO[1]:
        zone_type = jnp.array([1, 1]).reshape(-1, 1)
        position = jnp.array([[-20.0, -5.0], [20.0, 5.0]]).reshape(-1, 2)
        axes = jnp.array([[10.0, 5.0], [10.0, 5.0]]).reshape(-1, 2)
        damage = jnp.array([10.0, 10.0]).reshape(-1, 1)
    elif zone_scenario_name == ZONESCENARIO[2]:
        zone_type = jnp.array([2, 2, 2]).reshape(-1, 1)
        position = jnp.array([[-22.5, -7.0], [22.5, 7.0], [0, 0.0]]).reshape(-1, 2)
        axes = jnp.array([[5.0, 7.0], [5.0, 7.0], [3.0, 3.0]]).reshape(-1, 2)
        damage = jnp.array([0.0, 0.0, 0.0]).reshape(-1, 1)
    elif zone_scenario_name == ZONESCENARIO[3]:
        zone_type = jnp.array([1, 1, 2, 2]).reshape(-1, 1)
        position = jnp.array([[-20.0, -5.0], [20.0, 5.0], [-24.0, 7.0], [24.0, -7.0]]).reshape(
            -1, 2
        )
        axes = jnp.array([[10.0, 5.0], [10.0, 5.0], [5.0, 5.0], [5.0, 5.0]]).reshape(-1, 2)
        damage = jnp.array([10.0, 10.0, 0.0, 0.0]).reshape(-1, 1)
    else:
        raise NotImplementedError

    actual_n_zone = len(zone_type)
    n_zone = max_n_zone

    zone_scenario = ZoneScenario(
        n_zone=jnp.array(n_zone),
        zone_type=jnp.zeros((n_zone, 1))
        .at[: min(zone_type.shape[0], n_zone)]
        .set(zone_type[: min(zone_type.shape[0], n_zone)])
        .astype(jnp.int32)
        if max_n_zone is not None
        else zone_type,
        position=jnp.zeros((n_zone, 2))
        .at[: min(position.shape[0], n_zone)]
        .set(position[: min(position.shape[0], n_zone)])
        .astype(jnp.float32)
        if max_n_zone is not None
        else position,
        axes=jnp.zeros((n_zone, 2))
        .at[: min(axes.shape[0], n_zone)]
        .set(axes[: min(axes.shape[0], n_zone)])
        .astype(jnp.float32)
        if max_n_zone is not None
        else axes,
        damage=jnp.zeros((n_zone, 1))
        .at[: min(damage.shape[0], n_zone)]
        .set(damage[: min(damage.shape[0], n_zone)])
        .astype(jnp.float32)
        if max_n_zone is not None
        else damage,
    )

    # TABS Configuration
    tabs_config = TABSConfig(
        scenario_name=scenario_name,
        max_n_ally=int(ally_unit_comp.sum().item()) if max_n_ally is None else max_n_ally,
        max_n_enemy=int(enemy_unit_comp.sum().item()) if max_n_enemy is None else max_n_enemy,
        max_n_zone=int(zone_scenario.n_zone.item()),
    )

    return vscenario, zone_scenario, tabs_config


def get_vectorized_scenario(
    scenario,
    n_ally,
    n_enemy,
    unit_spacing=8.5,
    side_gap=16.0,
    field_margin_width=10.0,
    field_margin_height=10.0,
):
    pos_max = jnp.repeat(
        jnp.array(
            [
                scenario.battle_field.shape[0] * unit_spacing + side_gap / 2 + field_margin_width,
                scenario.battle_field.shape[1] * unit_spacing * 0.5 + field_margin_height,
            ]
        ).reshape(1, 2),
        n_ally + n_enemy,
        axis=0,
    )

    vectorized_scenario = VectorizedScenario(
        positions=jnp.zeros((n_ally + n_enemy, 2)),
        rotations=jnp.zeros((n_ally + n_enemy, 1)),
        body_weights=jnp.full((n_ally + n_enemy, 1), 1.0),
        body_radiuss=jnp.full((n_ally + n_enemy, 1), 1.0),
        teams=jnp.zeros((n_ally + n_enemy, 1)).astype(jnp.int32),
        pos_min=-pos_max,
        pos_max=pos_max,
        unit_ids=jnp.zeros((n_ally + n_enemy, 1)).astype(jnp.int32),
        healths=jnp.zeros((n_ally + n_enemy, 1)) + 1.0,
        attack_damages=jnp.zeros((n_ally + n_enemy, 1)),
        attack_ranges=jnp.full((n_ally + n_enemy, 1), 1.0),
        attack_cooldowns=jnp.full((n_ally + n_enemy, 1), 1.0),
        sight_angles=jnp.zeros((n_ally + n_enemy, 1)) + jnp.pi / 2,
        is_alive=jnp.full((n_ally + n_enemy, 1), True).astype(jnp.bool_),
        attack_types=jnp.zeros((n_ally + n_enemy, 1)).astype(jnp.int32),
        is_disabled=jnp.full((n_ally + n_enemy, 1), True).astype(jnp.bool_),
        speeds=jnp.full((n_ally + n_enemy, 1), 1.0),
    )

    def vectorize_body(carry, i, is_ally):
        vectorized_scenario, scenario = carry

        battle_field = scenario.battle_field if is_ally else scenario.enemy_battle_field
        unit_comp = scenario.ally_unit_comp if is_ally else scenario.enemy_unit_comp

        unit_idx = (battle_field > 0).argmax()
        unit_remain = unit_comp.sum() > 0

        x, y = jnp.unravel_index(unit_idx, battle_field.shape)

        deployed_unit_id = battle_field[x, y].astype(int) - 1
        _space_occupied = jnp.sqrt(scenario.space_occupied[deployed_unit_id])
        space_occupied_offset = unit_spacing * (_space_occupied - 1) / 2
        positions = vectorized_scenario.positions.at[i].set(
            jnp.stack(
                (
                    # x
                    (
                        (1 - is_ally) * (x * unit_spacing + side_gap / 2 + space_occupied_offset)
                        + is_ally * -(x * unit_spacing + side_gap / 2 + space_occupied_offset)
                    ),
                    # y
                    (
                        unit_spacing * (y - scenario.battle_field.shape[0] / 2)
                        + space_occupied_offset
                    )
                    * (0.5 - is_ally)
                    * 2,
                )
            )
            * unit_remain
            + vectorized_scenario.positions[i] * (1 - unit_remain)
        )
        rotations = vectorized_scenario.rotations.at[i].set(
            (1 - is_ally) * jnp.pi * unit_remain
            + vectorized_scenario.rotations[i] * (1 - unit_remain)
        )
        body_weights = vectorized_scenario.body_weights.at[i].set(
            scenario.body_weight[deployed_unit_id] * unit_remain
            + vectorized_scenario.body_weights[i] * (1 - unit_remain)
        )
        body_radiuss = vectorized_scenario.body_radiuss.at[i].set(
            scenario.body_radius[deployed_unit_id] * unit_remain
            + vectorized_scenario.body_radiuss[i] * (1 - unit_remain)
        )
        teams = vectorized_scenario.teams.at[i].set(1 - is_ally)
        unit_ids = vectorized_scenario.unit_ids.at[i].set(
            deployed_unit_id * unit_remain + vectorized_scenario.unit_ids[i] * (1 - unit_remain)
        )
        healths = vectorized_scenario.healths.at[i].set(
            scenario.health[deployed_unit_id] * unit_remain
            + vectorized_scenario.healths[i] * (1 - unit_remain)
        )
        attack_damages = vectorized_scenario.attack_damages.at[i].set(
            scenario.attack_damage[deployed_unit_id] * unit_remain
            + vectorized_scenario.attack_damages[i] * (1 - unit_remain)
        )
        attack_ranges = vectorized_scenario.attack_ranges.at[i].set(
            scenario.attack_range[deployed_unit_id] * unit_remain
            + vectorized_scenario.attack_ranges[i] * (1 - unit_remain)
        )
        attack_cooldowns = vectorized_scenario.attack_cooldowns.at[i].set(
            scenario.attack_cooldown[deployed_unit_id] * unit_remain
            + vectorized_scenario.attack_cooldowns[i] * (1 - unit_remain)
        )
        sight_angles = vectorized_scenario.sight_angles.at[i].set(
            scenario.sight_angle[deployed_unit_id] * unit_remain
            + vectorized_scenario.sight_angles[i] * (1 - unit_remain)
        )
        is_alive = vectorized_scenario.is_alive.at[i].set(
            1 * unit_remain + vectorized_scenario.is_alive[i] * (1 - unit_remain)
        )
        is_disabled = vectorized_scenario.is_disabled.at[i].set(
            False * unit_remain + vectorized_scenario.is_disabled[i] * (1 - unit_remain)
        )
        attack_types = vectorized_scenario.attack_types.at[i].set(
            scenario.attack_damage[deployed_unit_id]
            < 0 * unit_remain + vectorized_scenario.attack_types[i] * (1 - unit_remain)
        )
        speeds = vectorized_scenario.speeds.at[i].set(
            scenario.speed[deployed_unit_id] * unit_remain
            + vectorized_scenario.speeds[i] * (1 - unit_remain)
        )
        next_battle_field = battle_field.at[x, y].set(
            unit_remain * 0 + (1 - unit_remain) * (deployed_unit_id + 1)
        )
        next_unit_comp = unit_comp.at[deployed_unit_id].set(
            unit_comp[deployed_unit_id] - 1 * unit_remain
        )

        if is_ally:
            next_scenario = scenario.replace(
                battle_field=next_battle_field, ally_unit_comp=next_unit_comp
            )
        else:
            next_scenario = scenario.replace(
                enemy_battle_field=next_battle_field, enemy_unit_comp=next_unit_comp
            )

        pos_min = vectorized_scenario.pos_min.at[i].set(
            vectorized_scenario.pos_min[i] + scenario.body_radius[deployed_unit_id]
        )
        pos_max = vectorized_scenario.pos_max.at[i].set(
            vectorized_scenario.pos_max[i] - scenario.body_radius[deployed_unit_id]
        )

        next_vectorized_scenario = vectorized_scenario.replace(
            positions=positions,
            rotations=rotations,
            body_weights=body_weights,
            body_radiuss=body_radiuss,
            teams=teams,
            unit_ids=unit_ids,
            healths=healths,
            attack_damages=attack_damages,
            attack_ranges=attack_ranges,
            attack_cooldowns=attack_cooldowns,
            sight_angles=sight_angles,
            is_alive=is_alive,
            attack_types=attack_types,
            is_disabled=is_disabled,
            pos_min=pos_min,
            pos_max=pos_max,
            speeds=speeds,
        )

        return (next_vectorized_scenario, next_scenario), (x, y)

    ally_vectorize_body = functools.partial(vectorize_body, is_ally=True)
    enemy_vectorize_body = functools.partial(vectorize_body, is_ally=False)

    carry = (vectorized_scenario, scenario)

    carry, ally_positions = jax.lax.scan(ally_vectorize_body, carry, jnp.arange(n_ally))
    carry, enemy_positions = jax.lax.scan(
        enemy_vectorize_body, carry, jnp.arange(n_ally, n_ally + n_enemy)
    )

    return carry[0]


def calculate_unit_comp_price(scenario: UnitScenario):
    return (scenario.ally_unit_comp * scenario.price).sum(), (
        scenario.enemy_unit_comp * scenario.price
    ).sum()


def _generate_scenario(cfg: TABSConfig):
    max_shape = (cfg.max_field_height, cfg.max_field_width)
    # init
    budget = 0
    ally_unit_comp = jnp.zeros(cfg.max_num_units, dtype=jnp.float32)
    enemy_unit_comp = jnp.zeros_like(ally_unit_comp)
    unit_comp_mask = jnp.ones_like(ally_unit_comp)
    battle_field = jnp.zeros(max_shape, dtype=jnp.float32)
    battle_field_mask = jnp.zeros_like(battle_field)
    enemy_battle_field = jnp.zeros_like(battle_field)
    enemy_battle_field_mask = jnp.zeros_like(enemy_battle_field)

    all_spec = get_all_unit_spec()
    assert len(all_spec["prices"]) <= cfg.max_num_units
    m = cfg.max_num_units - len(all_spec["prices"])
    if m > 0:
        unit_comp_mask = unit_comp_mask.at[-m:].set(0)

    price = jnp.concatenate((all_spec["prices"], jnp.zeros(m)))
    health = jnp.concatenate((all_spec["healths"], jnp.zeros(m)))
    body_radius = jnp.concatenate((all_spec["body_radiuses"], jnp.zeros(m)))
    body_weight = jnp.concatenate((all_spec["body_weights"], jnp.zeros(m)))
    speed = jnp.concatenate((all_spec["speeds"], jnp.zeros(m)))
    attack_damage = jnp.concatenate((all_spec["attack_damages"], jnp.zeros(m)))
    attack_range = jnp.concatenate((all_spec["attack_ranges"], jnp.zeros(m)))
    attack_cooldown = jnp.concatenate((all_spec["attack_cooldown"], jnp.zeros(m)))
    sight_angle = jnp.concatenate((all_spec["sight_angles"], jnp.zeros(m)))
    space_occupied = jnp.concatenate((all_spec["space_occupied"], jnp.zeros(m)))

    _scenario_name = cfg.scenario_name.split("_")[0]
    _scenario_level = cfg.scenario_name.split("_")[1]

    if _scenario_name == SCENARIOS[0]:
        h, w = 4, 5
        assert max_shape[0] >= h and max_shape[1] >= w
        if _scenario_level.lower() == "abundant":
            budget = 2930
        elif _scenario_level.lower() == "medium":
            budget = 2650
        elif _scenario_level.lower() == "tight":
            budget = 2320
        _battle_field = jnp.array(  # 2640
            [
                [0, 0, UnitID.Mammoth, UnitID.Farmer, 0],
                [UnitID.Healer, UnitID.Archer, UnitID.Archer, UnitID.Archer, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
            ],
            dtype=jnp.float32,
        )
        _enemy_battle_field = jnp.array(
            [
                [0, UnitID.Farmer, UnitID.TheKing, UnitID.Farmer, 0],
                [0, 0, UnitID.Healer, 0, 0],
                [UnitID.Archer, 0, 0, 0, UnitID.Archer],
                [0, 0, 0, 0, 0],
            ],
            dtype=jnp.float32,
        )
        battle_field = battle_field.at[:h, :w].set(_battle_field)
        battle_field_mask = battle_field_mask.at[:h, :w].set(jnp.ones((h, w), dtype=jnp.float32))
        enemy_battle_field = enemy_battle_field.at[:h, :w].set(_enemy_battle_field)
        enemy_battle_field_mask = enemy_battle_field_mask.at[:h, :w].set(
            jnp.ones((h, w), dtype=jnp.float32)
        )
    elif _scenario_name == SCENARIOS[1]:
        h, w = 4, 5
        assert max_shape[0] >= h and max_shape[1] >= w
        if _scenario_level.lower() == "abundant":
            budget = 2420
        elif _scenario_level.lower() == "medium":
            budget = 2180
        elif _scenario_level.lower() == "tight":
            budget = 1940
        _battle_field = jnp.array(  # 2120
            [
                [UnitID.Farmer, UnitID.Farmer, UnitID.Mammoth, 0, 0],
                [0, UnitID.Archer, 0, UnitID.Archer, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
            ],
            dtype=jnp.float32,
        )
        _enemy_battle_field = jnp.array(
            [
                [0, 0, UnitID.TheKing, 0, 0],
                [UnitID.Assassin, 0, 0, 0, UnitID.Assassin],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
            ],
            dtype=jnp.float32,
        )
        battle_field = battle_field.at[:h, :w].set(_battle_field)
        battle_field_mask = battle_field_mask.at[:h, :w].set(jnp.ones((h, w), dtype=jnp.float32))
        enemy_battle_field = enemy_battle_field.at[:h, :w].set(_enemy_battle_field)
        enemy_battle_field_mask = enemy_battle_field_mask.at[:h, :w].set(
            jnp.ones((h, w), dtype=jnp.float32)
        )
    elif _scenario_name == SCENARIOS[2]:
        h, w = 4, 5
        assert max_shape[0] >= h and max_shape[1] >= w
        if _scenario_level.lower() == "abundant":
            budget = 3520
        elif _scenario_level.lower() == "medium":
            budget = 3370
        elif _scenario_level.lower() == "tight":
            budget = 2570
        _battle_field = jnp.array(  # 3360
            [
                [UnitID.Farmer, UnitID.Farmer, UnitID.TheKing, UnitID.Farmer, UnitID.Farmer],
                [0, UnitID.Archer, UnitID.Paladin, UnitID.Archer, UnitID.Assassin],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
            ],
            dtype=jnp.float32,
        )
        _enemy_battle_field = jnp.array(
            [
                [0, 0, UnitID.Mammoth, 0, 0],
                [0, 0, 0, 0, 0],
                [0, UnitID.Cannon, 0, UnitID.Cannon, 0],
                [0, 0, UnitID.Paladin, 0, 0],
            ],
            dtype=jnp.float32,
        )
        battle_field = battle_field.at[:h, :w].set(_battle_field)
        battle_field_mask = battle_field_mask.at[:h, :w].set(jnp.ones((h, w), dtype=jnp.float32))
        enemy_battle_field = enemy_battle_field.at[:h, :w].set(_enemy_battle_field)
        enemy_battle_field_mask = enemy_battle_field_mask.at[:h, :w].set(
            jnp.ones((h, w), dtype=jnp.float32)
        )
    elif _scenario_name == SCENARIOS[3]:
        h, w = 4, 5
        assert max_shape[0] >= h and max_shape[1] >= w
        if _scenario_level.lower() == "abundant":
            budget = 2450
        elif _scenario_level.lower() == "medium":
            budget = 1970
        elif _scenario_level.lower() == "tight":
            budget = 1720
        _battle_field = jnp.array(  # 1950
            [
                [UnitID.Farmer, UnitID.Farmer, UnitID.Farmer, UnitID.Farmer, UnitID.Farmer],
                [UnitID.Assassin, 0, UnitID.Archer, 0, UnitID.Deadeye],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
            ],
            dtype=jnp.float32,
        )
        _enemy_battle_field = jnp.array(
            [
                [UnitID.Farmer, UnitID.Farmer, UnitID.Farmer, UnitID.Farmer, UnitID.Farmer],
                [0, UnitID.Farmer, 0, UnitID.Farmer, 0],
                [UnitID.Deadeye, 0, 0, 0, UnitID.Deadeye],
                [UnitID.Healer, 0, 0, 0, 0],
            ],
            dtype=jnp.float32,
        )
        battle_field = battle_field.at[:h, :w].set(_battle_field)
        battle_field_mask = battle_field_mask.at[:h, :w].set(jnp.ones((h, w), dtype=jnp.float32))
        enemy_battle_field = enemy_battle_field.at[:h, :w].set(_enemy_battle_field)
        enemy_battle_field_mask = enemy_battle_field_mask.at[:h, :w].set(
            jnp.ones((h, w), dtype=jnp.float32)
        )
    else:
        raise NotImplementedError
    ally_unit_comp = jax.vmap(lambda x: (battle_field == (x + 1)).sum())(
        jnp.arange(cfg.max_num_units)
    )
    enemy_unit_comp = jax.vmap(lambda x: (enemy_battle_field == (x + 1)).sum())(
        jnp.arange(cfg.max_num_units)
    )
    return UnitScenario(
        budget=jnp.array([budget]),
        ally_unit_comp=ally_unit_comp,
        enemy_unit_comp=enemy_unit_comp,
        unit_comp_mask=unit_comp_mask,
        battle_field=battle_field,
        battle_field_mask=battle_field_mask,
        enemy_battle_field=enemy_battle_field,
        enemy_battle_field_mask=enemy_battle_field_mask,
        price=price,
        health=health,
        body_radius=body_radius,
        body_weight=body_weight,
        speed=speed,
        attack_damage=attack_damage,
        attack_range=attack_range,
        attack_cooldown=attack_cooldown,
        sight_angle=sight_angle,
        space_occupied=space_occupied,
    )


def pprint_grid_with_units(grid_array):
    """
    Return string representation of the grid showing units with alphabet characters

    Args:
        grid_array: numpy array (height x width)

    Returns:
        string representation of the grid
    """
    # Unit ID to alphabet mapping

    if hasattr(grid_array, "shape"):
        grid = jnp.array(grid_array)
    else:
        grid = grid_array
    # Convert to integer type
    grid = grid.astype(int)

    # Convert grid to string
    result = []
    for row in grid:
        row_str = ""
        for cell in row:
            row_str += UNITID2CHAR.get(cell.item(), str(cell)) + " "
        result.append(row_str.strip())

    return "\n".join(result)
