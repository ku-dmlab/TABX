import jax
import jax.numpy as jnp
import chex
from flax import struct

from src.maenv.tabs.units import get_all_unit_names, get_all_unit_spec


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
    velocity: chex.Array
    attack_damage: chex.Array
    attack_range: chex.Array  # WM
    attack_cooldown: chex.Array  # sec
    sight_angle: chex.Array
    sight_radius: chex.Array
    space_occupied: chex.Array  # area of rectangle shape


@struct.dataclass
class VectorizedScenario:
    positions: jnp.ndarray
    rotations: jnp.ndarray
    body_weights: jnp.ndarray
    body_radiuss: jnp.ndarray
    teams: jnp.ndarray
    pos_limits: jnp.ndarray
    unit_ids: jnp.ndarray
    healths: jnp.ndarray
    attack_damages: jnp.ndarray
    attack_ranges: jnp.ndarray
    attack_cooldowns: jnp.ndarray
    sight_angles: jnp.ndarray
    is_alive: jnp.ndarray
    attack_types: jnp.ndarray


@struct.dataclass
class TABSConf:
    scenario_name: str  # The predefined scenario name
    max_agents: int  # The maximum number of ally agents
    max_num_units: int  # The maximum number of unit types
    max_field_height: int  # The maximum height size of battle field
    max_field_width: int  # The maximum width size of battle field


default_tabs_conf = TABSConf(
    scenario_name="10farmers",
    max_agents=10,
    max_num_units=len(get_all_unit_names()),
    max_field_height=4,
    max_field_width=5,
)


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
    assert len(all_spec[0]) <= cfg.max_num_units
    m = cfg.max_num_units - len(all_spec[0])
    if m > 0:
        unit_comp_mask = unit_comp_mask.at[-m:].set(0)

    price = jnp.concatenate((all_spec[0], jnp.zeros(m)))
    health = jnp.concatenate((all_spec[1], jnp.zeros(m)))
    body_radius = jnp.concatenate((all_spec[2], jnp.zeros(m)))
    body_weight = jnp.concatenate((all_spec[3], jnp.zeros(m)))
    velocity = jnp.concatenate((all_spec[4], jnp.zeros(m)))
    attack_damage = jnp.concatenate((all_spec[5], jnp.zeros(m)))
    attack_range = jnp.concatenate((all_spec[6], jnp.zeros(m)))
    attack_cooldown = jnp.concatenate((all_spec[7], jnp.zeros(m)))
    sight_angle = jnp.concatenate((all_spec[8], jnp.zeros(m)))
    sight_radius = jnp.concatenate((all_spec[9], jnp.zeros(m)))
    space_occupied = jnp.concatenate((all_spec[10], jnp.zeros(m)))

    if cfg.scenario_name == "10farmers":
        h, w = 4, 5
        assert max_shape[0] >= h and max_shape[1] >= w
        budget = 1600
        ally_unit_comp = ally_unit_comp.at[0].set(10)
        enemy_unit_comp = enemy_unit_comp.at[0].set(10)
        # Mirror matchup
        battle_field = battle_field.at[:h, :w].set(jnp.ones((h, w), dtype=jnp.float32))
        battle_field_mask = battle_field_mask.at[:h, :w].set(jnp.ones((h, w), dtype=jnp.float32))
        enemy_battle_field = enemy_battle_field.at[:h, :w].set(jnp.ones((h, w), dtype=jnp.float32))
        enemy_battle_field_mask = enemy_battle_field_mask.at[:h, :w].set(
            jnp.ones((h, w), dtype=jnp.float32)
        )
    elif cfg.scenario_name == "1theking":
        h, w = 4, 5
        assert max_shape[0] >= h and max_shape[1] >= w
        budget = 1600
        ally_unit_comp = ally_unit_comp.at[2].set(1)
        enemy_unit_comp = enemy_unit_comp.at[2].set(1)
        # Mirror matchup
        _battle_field = jnp.concatenate(
            (jnp.array([[0, 0, 3, 0, 0]]), jnp.zeros((h - 1, w), dtype=jnp.float32)), axis=0
        )
        battle_field = battle_field.at[:h, :w].set(_battle_field)
        battle_field_mask = battle_field_mask.at[:h, :w].set(jnp.ones((h, w), dtype=jnp.float32))
        enemy_battle_field = enemy_battle_field.at[:h, :w].set(_battle_field)
        enemy_battle_field_mask = enemy_battle_field_mask.at[:h, :w].set(
            jnp.ones((h, w), dtype=jnp.float32)
        )
    elif cfg.scenario_name == "4archer_1mammoth":
        h, w = 4, 5
        assert max_shape[0] >= h and max_shape[1] >= w
        budget = 2000
        ally_unit_comp = ally_unit_comp.at[1].set(4)
        ally_unit_comp = ally_unit_comp.at[4].set(1)
        enemy_unit_comp = enemy_unit_comp.at[2].set(4)
        enemy_unit_comp = enemy_unit_comp.at[4].set(1)
        # Mirror matchup
        _battle_field = jnp.array(
            [[0, 0, 5, 0, 0], [0, 2, 0, 0, 2], [2, 0, 0, 0, 2], [0, 0, 0, 0, 0]], dtype=jnp.float32
        )
        battle_field = battle_field.at[:h, :w].set(_battle_field)
        battle_field_mask = battle_field_mask.at[:h, :w].set(jnp.ones((h, w), dtype=jnp.float32))
        enemy_battle_field = enemy_battle_field.at[:h, :w].set(_battle_field)
        enemy_battle_field_mask = enemy_battle_field_mask.at[:h, :w].set(
            jnp.ones((h, w), dtype=jnp.float32)
        )
    else:
        raise NotImplementedError

    return Scenario(
        budget=budget,
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
        velocity=velocity,
        attack_damage=attack_damage,
        attack_range=attack_range,
        attack_cooldown=attack_cooldown,
        sight_angle=sight_angle,
        sight_radius=sight_radius,
        space_occupied=space_occupied,
    )


def get_vectorized_scenario(scenario, n_unit):
    vectorized_scenario = VectorizedScenario(
        positions=jnp.zeros((n_unit * 2 * 2, 2)),
        rotations=jnp.zeros((n_unit * 2 * 2, 1)),
        body_weights=jnp.full((n_unit * 2, 1), 1.0),
        body_radiuss=jnp.full((n_unit * 2, 1), 1.0),
        teams=jnp.zeros((n_unit * 2, 1)).astype(jnp.int32),
        pos_limits=jnp.zeros((n_unit * 2, 2)),
        unit_ids=jnp.zeros((n_unit * 2, 1)).astype(jnp.int32),
        healths=jnp.zeros((n_unit * 2, 1)) + 1.0,
        attack_damages=jnp.zeros((n_unit * 2, 1)),
        attack_ranges=jnp.full((n_unit * 2, 1), 1.0),
        attack_cooldowns=jnp.full((n_unit * 2, 1), 1.0),
        sight_angles=jnp.zeros((n_unit * 2, 1)) + jnp.pi / 2,
        is_alive=jnp.zeros((n_unit * 2, 1)).astype(jnp.bool_),
        attack_types=jnp.zeros((n_unit * 2, 1)).astype(jnp.int32),
    )

    def ally_vectorize_body(carry, i):
        vectorized_scenario, scenario = carry

        unit_idx = (scenario.battle_field > 0).argmax()
        unit_remain = scenario.ally_unit_comp.sum() > 0

        x, y = jnp.unravel_index(unit_idx, scenario.battle_field.shape)

        deployed_unit_id = scenario.battle_field[x, y].astype(int) - 1
        positions = vectorized_scenario.positions.at[i].set(
            jnp.stack((2 * x, 2 * y)) * unit_remain
            + vectorized_scenario.positions[i] * (1 - unit_remain)
        )
        body_weights = vectorized_scenario.body_weights.at[i].set(
            scenario.body_weight[deployed_unit_id] * unit_remain
            + vectorized_scenario.body_weights[i] * (1 - unit_remain)
        )
        body_radiuss = vectorized_scenario.body_radiuss.at[i].set(
            scenario.body_radius[deployed_unit_id] * unit_remain
            + vectorized_scenario.body_radiuss[i] * (1 - unit_remain)
        )
        teams = vectorized_scenario.teams.at[i].set(0)
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
        attack_types = vectorized_scenario.attack_types.at[i].set(
            scenario.attack_damage[deployed_unit_id]
            < 0 * unit_remain + vectorized_scenario.attack_types[i] * (1 - unit_remain)
        )

        next_battle_field = scenario.battle_field.at[x, y].set(
            unit_remain * 0 + (1 - unit_remain) * (deployed_unit_id + 1)
        )
        next_ally_unit_comp = scenario.ally_unit_comp.at[deployed_unit_id].set(
            scenario.ally_unit_comp[deployed_unit_id] - 1 * unit_remain
        )

        next_scenario = scenario.replace(
            battle_field=next_battle_field, ally_unit_comp=next_ally_unit_comp
        )

        next_vectorized_scenario = vectorized_scenario.replace(
            positions=positions,
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
        )

        return (next_vectorized_scenario, next_scenario), None

    def enemy_vectorize_body(carry, i):
        vectorized_scenario, scenario = carry

        unit_idx = (scenario.enemy_battle_field > 0).argmax()
        unit_remain = scenario.enemy_unit_comp.sum() > 0

        x, y = jnp.unravel_index(unit_idx, scenario.enemy_battle_field.shape)

        deployed_unit_id = scenario.enemy_battle_field[x, y].astype(int) - 1
        positions = vectorized_scenario.positions.at[i].set(
            jnp.stack((3 * (x + scenario.battle_field.shape[0]), 3 * y)) * unit_remain
            + vectorized_scenario.positions[i] * (1 - unit_remain)
        )
        body_weights = vectorized_scenario.body_weights.at[i].set(
            scenario.body_weight[deployed_unit_id] * unit_remain
            + vectorized_scenario.body_weights[i] * (1 - unit_remain)
        )
        body_radiuss = vectorized_scenario.body_radiuss.at[i].set(
            scenario.body_radius[deployed_unit_id] * unit_remain
            + vectorized_scenario.body_radiuss[i] * (1 - unit_remain)
        )
        teams = vectorized_scenario.teams.at[i].set(1)
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
        attack_types = vectorized_scenario.attack_types.at[i].set(
            scenario.attack_damage[deployed_unit_id]
            < 0 * unit_remain + vectorized_scenario.attack_types[i] * (1 - unit_remain)
        )

        next_enemy_battle_field = scenario.enemy_battle_field.at[x, y].set(
            unit_remain * 0 + (1 - unit_remain) * (deployed_unit_id + 1)
        )
        next_enemy_unit_comp = scenario.enemy_unit_comp.at[deployed_unit_id].set(
            scenario.enemy_unit_comp[deployed_unit_id] - 1 * unit_remain
        )

        next_scenario = scenario.replace(
            enemy_battle_field=next_enemy_battle_field, enemy_unit_comp=next_enemy_unit_comp
        )

        next_vectorized_scenario = vectorized_scenario.replace(
            positions=positions,
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
        )

        return (next_vectorized_scenario, next_scenario), None

    carry = (vectorized_scenario, scenario)

    carry, _ = jax.lax.scan(ally_vectorize_body, carry, jnp.arange(n_unit))
    carry, _ = jax.lax.scan(enemy_vectorize_body, carry, jnp.arange(n_unit, n_unit * 2))

    return carry[0]
