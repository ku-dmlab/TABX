import functools

import chex
import jax
import jax.numpy as jnp
from flax import struct

from src.tabs.units import UnitID, get_all_unit_names, get_all_unit_spec


@struct.dataclass
class VectorizedScenario:
    positions: jnp.ndarray
    rotations: jnp.ndarray
    body_weights: jnp.ndarray
    body_radiuss: jnp.ndarray
    teams: jnp.ndarray
    pos_min: jnp.ndarray
    pos_max: jnp.ndarray
    unit_ids: jnp.ndarray
    healths: jnp.ndarray
    attack_damages: jnp.ndarray
    attack_ranges: jnp.ndarray
    attack_cooldowns: jnp.ndarray
    sight_angles: jnp.ndarray
    is_alive: jnp.ndarray
    attack_types: jnp.ndarray
    is_disabled: jnp.ndarray
    speeds: jnp.ndarray


@struct.dataclass
class Scenario:
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


@struct.dataclass
class TABSConf:
    scenario_name: str = "10F"  # The predefined scenario name
    max_num_units: int = len(get_all_unit_names())  # The maximum number of unit types
    max_field_height: int = 4  # The maximum height size of battle field
    max_field_width: int = 5  # The maximum width size of battle field
    max_n_ally: int = 10  # The maximum number of ally agents
    max_n_enemy: int = 10  # The maximum number of enemy agents


# unit names: F, A, K, B, M, D, H
def get_scenario_name_list():
    return ["2F1K2A1H"]


def calculate_unit_comp_price(scenario: Scenario):
    return (scenario.ally_unit_comp * scenario.price).sum(), (
        scenario.enemy_unit_comp * scenario.price
    ).sum()


def generate_scenario(cfg: TABSConf):
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

    scenario_names = get_scenario_name_list()

    if cfg.scenario_name == scenario_names[0]:
        h, w = 4, 5
        assert max_shape[0] >= h and max_shape[1] >= w
        budget = 2500
        _battle_field = jnp.array(
            [
                [0, UnitID.Farmer, UnitID.TheKing, UnitID.Farmer, 0],
                [0, 0, UnitID.Healer, 0, 0],
                [UnitID.Archer, 0, 0, 0, UnitID.Archer],
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
    else:
        raise NotImplementedError
    ally_unit_comp = jax.vmap(lambda x: (battle_field == (x + 1)).sum())(
        jnp.arange(cfg.max_num_units)
    )
    enemy_unit_comp = jax.vmap(lambda x: (enemy_battle_field == (x + 1)).sum())(
        jnp.arange(cfg.max_num_units)
    )
    return Scenario(
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


def get_vectorized_scenario(scenario, n_ally, n_enemy, unit_spacing=6, side_gap=0, field_margin=5):
    pos_max = (
        jnp.repeat(
            jnp.array(
                [
                    (scenario.battle_field.shape[0] * 3 / 2 + side_gap / 2) * unit_spacing,
                    (scenario.battle_field.shape[1] * 3 / 2) * unit_spacing,
                ]
            ).reshape(1, 2),
            n_ally + n_enemy,
            axis=0,
        )
        + field_margin
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
        positions = vectorized_scenario.positions.at[i].set(
            jnp.stack(
                (
                    unit_spacing
                    * (
                        (1 - is_ally) * (x + scenario.battle_field.shape[0] / 2 + side_gap / 2)
                        + is_ally * -(x + scenario.battle_field.shape[0] / 2 + side_gap / 2)
                    ),
                    unit_spacing * (y - scenario.battle_field.shape[1] / 2),
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
