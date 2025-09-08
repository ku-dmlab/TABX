import jax
import jax.numpy as jnp

from src.tabs.units import UnitID, get_all_unit_spec
from src.tabs.scenarios import Scenario, TABSConf, calculate_unit_comp_price


# unit names: F, A, K, B, M, D, H
def get_scenario_name_list():
    return ["debug"]


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

    if cfg.scenario_name == "debug":
        h, w = 4, 5
        assert max_shape[0] >= h and max_shape[1] >= w
        budget = 2000
        _battle_field = jnp.array(
            [
                [0, 0, 0, 0, 0],
                [0, UnitID.Farmer, UnitID.Farmer, UnitID.Farmer, 0],
                [UnitID.Assassin, UnitID.Paladin, 0, UnitID.Paladin, UnitID.Assassin],
                [0, 0, 0, 0, 0],
            ],
            dtype=jnp.float32,
        )
        _enemy_battle_field = jnp.array(
            [
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, UnitID.Archer, 0, UnitID.Archer, 0],
                [0, UnitID.Archer, UnitID.Archer, UnitID.Archer, 0],
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
