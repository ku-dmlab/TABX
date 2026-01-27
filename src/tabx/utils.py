import warnings
from typing import Dict, List, NamedTuple, Optional, Tuple

import chex

from src.tabx.config import TABXConfig
from src.tabx.heuristic_policy import build_batched_heuristic_params
from src.tabx.physics import build_batched_physics_params
from src.tabx.scenarios import build_batched_scenarios


class Transition(NamedTuple):
    global_done: chex.Array
    done: chex.Array
    action: chex.Array
    value: chex.Array
    reward: chex.Array
    log_prob: chex.Array
    obs: chex.Array
    world_state: chex.Array
    info: chex.Array
    avail_actions: chex.Array


def notify(sprites, event, info):
    for key, sprite in sprites.items():
        if hasattr(sprite, "on_" + event):
            sprites[key] = getattr(sprite, "on_" + event)(sprites, info)
    return sprites


EnvParameters = Dict[str, chex.Array]


def build_batched_env_params_and_config(
    scenario_names: List[str] | str = "default",
    physics_param_names: List[str] | str = "default",
    heuristic_param_names: List[str] | str = "easy",
    n_repeat: int = 1,
    squeeze_when_single_scenario: bool = True,
    max_n_ally: Optional[int] = None,
    max_n_enemy: Optional[int] = None,
    max_n_zone: Optional[int] = None,
) -> Tuple[EnvParameters, TABXConfig]:
    n_scenario = 1 if isinstance(scenario_names, str) else len(scenario_names)
    n_physics = 1 if isinstance(physics_param_names, str) else len(physics_param_names)
    n_heuristic = 1 if isinstance(heuristic_param_names, str) else len(heuristic_param_names)

    if not (n_scenario == n_physics == n_heuristic):
        warnings.warn(
            "The number of scenarios, physics parameters, and heuristic parameters should be the same.",
            category=UserWarning,
        )

    vscenario, zone_scenario, tabx_config = build_batched_scenarios(
        scenario_names=scenario_names,
        n_repeat=n_repeat,
        squeeze_when_single_scenario=squeeze_when_single_scenario,
        max_n_ally=max_n_ally,
        max_n_enemy=max_n_enemy,
        max_n_zone=max_n_zone,
    )
    physics_params = build_batched_physics_params(
        physics_param_names=physics_param_names,
        n_repeat=n_repeat,
        squeeze_when_single_physics=squeeze_when_single_scenario,
    )
    heuristic_params = build_batched_heuristic_params(
        heuristic_param_names=heuristic_param_names,
        n_repeat=n_repeat,
        squeeze_when_single_heuristic=squeeze_when_single_scenario,
    )
    return {
        "scenario": vscenario,
        "zone_scenario": zone_scenario,
        "physics_params": physics_params,
        "heuristic_params": heuristic_params,
    }, tabx_config
