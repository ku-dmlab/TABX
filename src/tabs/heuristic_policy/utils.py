import json
from pathlib import Path
from typing import List

import jax
import jax.numpy as jnp

from src.tabs.heuristic_policy.constants import HEURISTIC_PARAMS
from src.tabs.heuristic_policy.params import TABSHeuristicConfig


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

    physics_params = to_jnp_array(loaded)

    return TABSHeuristicConfig(**physics_params)


def load_heuristic_params_from_json(heuristic_param_name: str = "default"):
    base_path = Path(__file__).resolve().parent
    if heuristic_param_name not in HEURISTIC_PARAMS:
        raise ValueError(
            f"Heuristic param name {heuristic_param_name} not found in {HEURISTIC_PARAMS}"
        )
    return load_json_to_jnp(str(base_path / "parameters" / f"{heuristic_param_name}.json"))


def build_batched_heuristic_params(
    heuristic_param_names: List[str] | str = "easy",
    n_repeat: int = 1,
    squeeze_when_single_heuristic: bool = True,
):
    if isinstance(heuristic_param_names, str):
        heuristic_param_names = [heuristic_param_names]
    heuristic_params = [
        load_heuristic_params_from_json(heuristic_param_name)
        for heuristic_param_name in heuristic_param_names
    ]
    stacked_heuristic_params = jax.tree.map(
        lambda *args: jnp.repeat(jnp.stack(args), axis=0, repeats=n_repeat),
        *heuristic_params,
    )
    if squeeze_when_single_heuristic and (len(heuristic_param_names) * n_repeat == 1):
        stacked_heuristic_params = jax.tree.map(
            lambda x: x.squeeze(axis=0), stacked_heuristic_params
        )
    return stacked_heuristic_params
