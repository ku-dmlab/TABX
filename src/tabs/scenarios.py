import functools
import itertools
import json
import os
from typing import Tuple

import chex
import jax
import jax.numpy as jnp
import numpy as np
from flax import struct

from src.tabs.config import TABSConfig

UNIT_SCENARIOS = [f.replace(".json", "") for f in os.listdir("src/scenarios/units")]
ZONE_SCENARIOS = [f.replace(".json", "") for f in os.listdir("src/scenarios/zones")]
CHALLENGES = [f.replace(".json", "") for f in os.listdir("src/scenarios/challenges")]


def load_challenge(challenge_name: str):
    return load_scenario_from_json(f"src/scenarios/challenges/{challenge_name}.json")


def load_scenario_from_json(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        loaded = json.load(f)

    def to_jnp_array(obj):
        if isinstance(obj, dict):
            return {k: to_jnp_array(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return jnp.array(obj)
        else:
            return jnp.array([obj]).reshape(-1, 1)

    env_params = to_jnp_array(loaded)
    if "scenario" in env_params:
        env_params["scenario"] = VectorizedScenario(**env_params["scenario"])
    if "zone_scenario" in env_params:
        env_params["zone_scenario"] = ZoneScenario(**env_params["zone_scenario"])

    return env_params


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


def generate_scenario_config(
    scenario_name: str,
    max_n_ally: int = None,
    max_n_enemy: int = None,
    max_n_zone: int = None,
) -> Tuple[VectorizedScenario, ZoneScenario, TABSConfig]:
    splited_scenario_name = scenario_name.split("_")
    if len(splited_scenario_name) > 1:
        scenario_name = splited_scenario_name[0]
        zone_name = splited_scenario_name[1]
        if scenario_name not in UNIT_SCENARIOS:
            raise ValueError(f"Scenario name {scenario_name} not found in {UNIT_SCENARIOS}")
        if zone_name not in ZONE_SCENARIOS:
            raise ValueError(f"Zone name {zone_name} not found in {ZONE_SCENARIOS}")
        vscenario = load_scenario_from_json(f"src/scenarios/units/{scenario_name}.json")["scenario"]
        zone_scenario = load_scenario_from_json(f"src/scenarios/zones/{zone_name}.json")[
            "zone_scenario"
        ]
    elif scenario_name in CHALLENGES:
        env_params = load_scenario_from_json(f"src/scenarios/challenges/{scenario_name}.json")
        vscenario = env_params["scenario"]
        zone_scenario = env_params["zone_scenario"]
    else:
        if scenario_name not in UNIT_SCENARIOS:
            raise ValueError(f"Scenario name {scenario_name} not found in {UNIT_SCENARIOS}")
        zone_name = "void"
        vscenario = load_scenario_from_json(f"src/scenarios/units/{scenario_name}.json")["scenario"]
        zone_scenario = load_scenario_from_json(f"src/scenarios/zones/{zone_name}.json")[
            "zone_scenario"
        ]

    # padding the scenario to the max_n_ally + max_n_enemy
    if max_n_ally is not None and max_n_enemy is not None:
        padded_vsscenario = VectorizedScenario(
            positions=jnp.zeros((max_n_ally + max_n_enemy, 2)),
            rotations=jnp.zeros((max_n_ally + max_n_enemy, 1)),
            body_weights=jnp.zeros((max_n_ally + max_n_enemy, 1)),
            body_radiuss=jnp.zeros((max_n_ally + max_n_enemy, 1)),
            teams=jnp.concat(
                [
                    jnp.zeros((max_n_ally, 1), dtype=jnp.int32),
                    jnp.ones((max_n_enemy, 1), dtype=jnp.int32),
                ]
            ),
            pos_min=jnp.zeros((max_n_ally + max_n_enemy, 2)),
            pos_max=jnp.zeros((max_n_ally + max_n_enemy, 2)),
            unit_ids=jnp.zeros((max_n_ally + max_n_enemy, 1), dtype=jnp.int32),
            healths=jnp.zeros((max_n_ally + max_n_enemy, 1)),
            attack_damages=jnp.zeros((max_n_ally + max_n_enemy, 1)),
            attack_ranges=jnp.zeros((max_n_ally + max_n_enemy, 1)),
            attack_cooldowns=jnp.zeros((max_n_ally + max_n_enemy, 1)),
            sight_angles=jnp.zeros((max_n_ally + max_n_enemy, 1)),
            is_alive=jnp.zeros((max_n_ally + max_n_enemy, 1), dtype=jnp.bool_),
            attack_types=jnp.zeros((max_n_ally + max_n_enemy, 1), dtype=jnp.int32),
            is_disabled=jnp.ones((max_n_ally + max_n_enemy, 1), dtype=jnp.bool_),
            speeds=jnp.zeros((max_n_ally + max_n_enemy, 1)),
        )

        n_ally = int((vscenario.teams == 0).sum().item())
        n_enemy = int((vscenario.teams == 1).sum().item())

        padded_vsscenario = jax.tree.map(
            lambda x, y: x.at[:n_ally].set(y[:n_ally]), padded_vsscenario, vscenario
        )
        padded_vsscenario = jax.tree.map(
            lambda x, y: x.at[max_n_ally : max_n_ally + n_enemy].set(y[n_ally : n_ally + n_enemy]),
            padded_vsscenario,
            vscenario,
        )

        vscenario = padded_vsscenario

    if max_n_zone is not None:
        # padding the zone scenario to the max_n_zone
        padded_zone_scenario = ZoneScenario(
            n_zone=jnp.array([[max_n_zone]], dtype=jnp.int32),
            zone_type=jnp.zeros((max_n_zone, 1), dtype=jnp.int32),
            position=jnp.zeros((max_n_zone, 2)),
            axes=jnp.zeros((max_n_zone, 2)),
            effect_value=jnp.zeros((max_n_zone, 1)),
        )
        n_zone = int(zone_scenario.n_zone.item())
        padded_zone_scenario = jax.tree.map(
            lambda x, y: x.at[:n_zone].set(y[:n_zone]), padded_zone_scenario, zone_scenario
        )
        padded_zone_scenario = jax.tree.map(
            lambda x, y: x.at[max_n_zone:].set(y[n_zone:]), padded_zone_scenario, zone_scenario
        )
        zone_scenario = padded_zone_scenario

    # TABS Configuration
    tabs_config = TABSConfig(
        scenario_name=scenario_name,
        max_n_ally=int((vscenario.teams == 0).sum().item()) if max_n_ally is None else max_n_ally,
        max_n_enemy=int((vscenario.teams == 1).sum().item())
        if max_n_enemy is None
        else max_n_enemy,
        max_n_zone=int(zone_scenario.n_zone.item()),
    )

    return vscenario, zone_scenario, tabs_config
