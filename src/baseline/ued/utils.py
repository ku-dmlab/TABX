from itertools import product
from typing import List

from src.tabs.heuristic_policy.constants import HEURISTIC_PARAMS
from src.tabs.scenarios.constants import CHALLENGES, EVAL_UNIT_SCENARIOS, EVAL_ZONE_SCENARIOS


def get_evaluation_scenarios(scenario_name: str, free_param_type: List) -> List:
    if scenario_name in CHALLENGES:
        raise ValueError(f"{scenario_name} is not supported in the UED setting.")

    splited_scenario_name = scenario_name.split("_")
    enemy_comp = splited_scenario_name[0].split("vs")[1]
    if len(splited_scenario_name) == 1:
        zone_name = "void"
    else:
        zone_name = splited_scenario_name[1]

    if set(["zone", "unit_spec"]) <= set(list(free_param_type)):  # both zone & unit spec
        """Return eval unit scenarios with eval zone scenarios."""
        unit_scenarios = [s for s in EVAL_UNIT_SCENARIOS if enemy_comp in s] + [
            splited_scenario_name[0]
        ]
        zone_scenarios = [s for s in EVAL_ZONE_SCENARIOS if f"{zone_name}-" in s] + [zone_name]

        return [
            (s1 + "_" + s2).replace("_void", "")
            for s1, s2 in product(unit_scenarios, zone_scenarios)
        ]
    elif "zone" in free_param_type:  # zone only
        """Return the given unit scenario with eval zone scenarios."""
        return [scenario_name] + [
            splited_scenario_name[0] + "_" + s for s in EVAL_ZONE_SCENARIOS if f"{zone_name}-" in s
        ]
    elif "unit_spec" in free_param_type:  # unit spec only
        """Return eval unit scenario with the given zone scenario."""
        return [scenario_name] + [
            (s1 + "_" + s2).replace("_void", "")
            for s1, s2 in product(EVAL_UNIT_SCENARIOS, [zone_name])
            if enemy_comp in s1.split("vs")[-1]
        ]
    else:  # heuristic config
        return [scenario_name]


def get_evaluation_heuristic_params(heuristic_param_name: str, free_param_type: List) -> List:
    if "heuristic_config" in free_param_type:
        return list(set(HEURISTIC_PARAMS) - set(list(heuristic_param_name)))
    return [heuristic_param_name]
