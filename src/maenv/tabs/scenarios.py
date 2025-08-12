from typing import Optional
import chex
from flax import struct
import jax.numpy as jnp
from src.maenv.tabs.units import get_all_unit_names, get_all_unit_spec


@struct.dataclass
class Scenario:
    budget: int
    ally_unit_comp: chex.Array
    enemy_unit_comp: chex.Array
    unit_comp_mask: chex.Array
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
class TABSConf:
    scenario_name: str
    max_agents: int
    max_num_units: int
    max_field_height: int
    max_field_width: int
    scenario: Optional[Scenario]


default_tabs_conf = TABSConf(
    scenario_name="20farmers",
    max_agents=20,
    max_num_units=len(get_all_unit_names()),
    max_field_height=4,
    max_field_width=5,
    scenario=None,
)


def generate_scenario(cfg: TABSConf):
    if cfg.scenario:  # Custom scenario
        return cfg.scenario

    max_shape = (cfg.max_field_height, cfg.max_field_width)
    # init
    budget = 0
    ally_unit_comp = jnp.zeros(cfg.max_num_units, dtype=jnp.float32)
    enemy_unit_comp = jnp.zeros_like(ally_unit_comp)
    unit_comp_mask = jnp.ones_like(ally_unit_comp)
    enemy_battle_field = jnp.zeros(max_shape, dtype=jnp.float32)
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

    if cfg.scenario_name == "20farmers":
        h, w = 4, 5
        assert max_shape[0] >= h and max_shape[1] >= w
        budget = 1600
        ally_unit_comp = ally_unit_comp.at[0].set(20)
        enemy_unit_comp = enemy_unit_comp.at[0].set(20)
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
        _enemy_battle_field = jnp.concatenate(
            (jnp.array([[0, 0, 3, 0, 0]]), jnp.zeros((h - 1, w), dtype=jnp.float32)), axis=0
        )
        enemy_battle_field = enemy_battle_field.at[:h, :w].set(_enemy_battle_field)
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
        _enemy_battle_field = jnp.array(
            [[0, 0, 5, 0, 0], [0, 2, 0, 0, 2], [2, 0, 0, 0, 2], [0, 0, 0, 0, 0]], dtype=jnp.float32
        )
        enemy_battle_field = enemy_battle_field.at[:h, :w].set(_enemy_battle_field)
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
