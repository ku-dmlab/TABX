import json
from typing import List, Optional, Tuple

import jax
import jax.numpy as jnp

from src.scenarios.constants import CHALLENGES, UNIT_SCENARIOS, ZONE_SCENARIOS
from src.scenarios.scenario import VectorizedScenario, ZoneScenario
from src.tabs.config import TABSConfig


def load_json_to_jnp(file_path):
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


def load_scenario_from_json(scenario_name: str):
    splited_scenario_name = scenario_name.split("_")
    if len(splited_scenario_name) > 1:
        scenario_name = splited_scenario_name[0]
        zone_name = splited_scenario_name[1]
        if scenario_name not in UNIT_SCENARIOS:
            raise ValueError(f"Scenario name {scenario_name} not found in {UNIT_SCENARIOS}")
        if zone_name not in ZONE_SCENARIOS:
            raise ValueError(f"Zone name {zone_name} not found in {ZONE_SCENARIOS}")
        vscenario = load_json_to_jnp(f"src/scenarios/units/{scenario_name}.json")["scenario"]
        zone_scenario = load_json_to_jnp(f"src/scenarios/zones/{zone_name}.json")["zone_scenario"]
    elif scenario_name in CHALLENGES:
        env_params = load_json_to_jnp(f"src/scenarios/challenges/{scenario_name}.json")
        vscenario = env_params["scenario"]
        zone_scenario = env_params["zone_scenario"]
    else:
        if scenario_name not in UNIT_SCENARIOS:
            raise ValueError(f"Scenario name {scenario_name} not found in {UNIT_SCENARIOS}")
        zone_name = "void"
        vscenario = load_json_to_jnp(f"src/scenarios/units/{scenario_name}.json")["scenario"]
        zone_scenario = load_json_to_jnp(f"src/scenarios/zones/{zone_name}.json")["zone_scenario"]

    return vscenario, zone_scenario


def generate_padded_unit_scenario(
    vscenario: VectorizedScenario, max_n_ally: int, max_n_enemy: int, max_n_zone: int
):
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
        lambda x, y: x.at[: min(n_ally, max_n_ally)].set(y[: min(n_ally, max_n_ally)]),
        padded_vsscenario,
        vscenario,
    )
    padded_vsscenario = jax.tree.map(
        lambda x, y: x.at[max_n_ally : max_n_ally + min(n_enemy, max_n_enemy)].set(
            y[n_ally : n_ally + min(n_enemy, max_n_enemy)]
        ),
        padded_vsscenario,
        vscenario,
    )
    return padded_vsscenario


def generate_padded_zone_scenario(zone_scenario: ZoneScenario, max_n_zone: int):
    padded_zone_scenario = ZoneScenario(
        n_zone=jnp.array([[max_n_zone]], dtype=jnp.int32),
        zone_type=jnp.zeros((max_n_zone, 1), dtype=jnp.int32),
        position=jnp.zeros((max_n_zone, 2)),
        axes=jnp.zeros((max_n_zone, 2)),
        effect_value=jnp.zeros((max_n_zone, 1)),
    )

    n_zone = min(int(zone_scenario.position.shape[0]), max_n_zone)
    padded_zone_scenario = jax.tree.map(
        lambda x, y: x.at[:n_zone].set(y[:n_zone]), padded_zone_scenario, zone_scenario
    )
    return padded_zone_scenario


def build_batched_scenarios(
    scenario_names: List[str] | str,
    n_repeat: int = 1,
    max_n_ally: Optional[int] = None,
    max_n_enemy: Optional[int] = None,
    max_n_zone: Optional[int] = None,
) -> Tuple[VectorizedScenario, ZoneScenario, TABSConfig]:
    if isinstance(scenario_names, str):
        scenario_names = [scenario_names]
    vscenarios, zone_scenarios = zip(
        *[load_scenario_from_json(scenario_name) for scenario_name in scenario_names]
    )

    if max_n_ally is None:
        max_n_ally = max([(vscenario.teams == 0).sum().item() for vscenario in vscenarios])
    if max_n_enemy is None:
        max_n_enemy = max([(vscenario.teams == 1).sum().item() for vscenario in vscenarios])
    if max_n_zone is None:
        max_n_zone = max([zone_scenario.n_zone.item() for zone_scenario in zone_scenarios])

    vscenarios = [
        generate_padded_unit_scenario(vscenario, max_n_ally, max_n_enemy, max_n_zone)
        for vscenario in vscenarios
    ]
    zone_scenarios = [
        generate_padded_zone_scenario(zone_scenario, max_n_zone) for zone_scenario in zone_scenarios
    ]

    stacked_vscenario = jax.tree.map(
        lambda *args: jnp.repeat(jnp.stack(args), axis=0, repeats=n_repeat), *vscenarios
    )
    stacked_zone_scenario = jax.tree.map(
        lambda *args: jnp.repeat(jnp.stack(args), axis=0, repeats=n_repeat),
        *zone_scenarios,
    )

    # TABS Configuration
    tabs_config = TABSConfig(
        max_n_ally=max_n_ally,
        max_n_enemy=max_n_enemy,
        max_n_zone=max_n_zone,
    )
    return stacked_vscenario, stacked_zone_scenario, tabs_config
