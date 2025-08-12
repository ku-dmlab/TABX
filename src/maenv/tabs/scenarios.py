from typing import Optional
import chex
from flax import struct
import jax.numpy as jnp
from src.maenv.tabs.units import get_all_unit_names


@struct.dataclass
class Scenario:
    budget: int
    ally_unit_comp: chex.Array
    enemy_unit_comp: chex.Array
    enemy_battle_field: chex.Array
    enemy_battle_field_mask: chex.Array


@struct.dataclass
class TABSConf:
    scenario_name: str
    max_agents: int
    max_field_height: int
    max_field_width: int
    scenario: Optional[Scenario]


default_tabs_conf = TABSConf(
    scenario_name="20farmers", max_agents=20, max_field_height=4, max_field_width=5, scenario=None
)


def generate_scenario(cfg: TABSConf):
    if cfg.scenario:  # Custom scenario
        return cfg.scenario

    num_units = len(get_all_unit_names())
    max_shape = (cfg.max_field_height, cfg.max_field_width)
    # init
    budget = 0
    ally_unit_comp = jnp.zeros(num_units, dtype=jnp.float32)
    enemy_unit_comp = jnp.zeros(num_units, dtype=jnp.float32)
    enemy_battle_field = jnp.zeros(max_shape, dtype=jnp.float32)
    enemy_battle_field_mask = jnp.zeros_like(enemy_battle_field)

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
        enemy_battle_field=enemy_battle_field,
        enemy_battle_field_mask=enemy_battle_field_mask,
    )
