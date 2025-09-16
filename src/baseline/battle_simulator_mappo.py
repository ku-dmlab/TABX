import hashlib
from dataclasses import dataclass, replace
import json

from tqdm import tqdm
import tyro
import wandb
import numpy as np

from src.baseline.configs.config import PPOConfig
from src.tabs.scenarios import TABSConfig


@dataclass
class Config:
    seed: int = 42
    n_env: int = 32  # the number of environments to run in parallel
    tabs: TABSConfig = TABSConfig()
    mappo: PPOConfig = PPOConfig(n_env=n_env, seed=seed)
    save_path: str = "/save"
    gpu_id: int = 3

    total_iter: int = 40
    iter_per_train_step: int = 100


if __name__ == "__main__":
    config = tyro.cli(Config)

    import os
    from functools import partial
    import datetime

    os.environ["CUDA_VISIBLE_DEVICES"] = str(config.gpu_id)
    import jax
    import jax.numpy as jnp

    from src.baseline.algorithm import MAPPO
    from src.tabs.wrappers import (
        TABSBattleSimulatorAutoResetWrapper,
        TABSBattleSimulatorLogWrapper,
        TABSBattleSimulatorHeuristicWrapper,
    )
    from src.tabs.constants import ALL_UNIT_NAMES
    from src.tabs import TABSBattleSimulator
    from src.tabs.scenarios import (
        TABSConfig,
        generate_scenario,
        get_vectorized_scenario,
        VectorizedScenario,
    )
    from src.baseline.utils import dataclass_to_dict, get_abs_path

    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    config_dict = dataclass_to_dict(config)
    config_hash = hashlib.md5(str(config).encode()).hexdigest()[:8]
    config.save_path = f"/save/{config.tabs.scenario_name}_{current_time}_{config_hash}"
    wandb.init(project="battle_simulator_mappo", config=config)

    logs_dir = get_abs_path(config.save_path)
    os.makedirs(logs_dir, exist_ok=True)

    # Save config to logs directory
    with open(os.path.join(logs_dir, "config.json"), "w") as f:
        json.dump(config_dict, f, indent=2)

    tabs_conf = config.tabs
    scenario = generate_scenario(tabs_conf)
    config.tabs = replace(
        config.tabs,
        max_n_ally=int(scenario.ally_unit_comp.sum().item()),
        max_n_enemy=int(scenario.enemy_unit_comp.sum().item()),
    )
    repeated_scenarios = jax.tree.map(lambda x: jnp.repeat(x[None], config.n_env, axis=0), scenario)

    env = TABSBattleSimulator(tabs_conf)
    env = TABSBattleSimulatorHeuristicWrapper(env, "enemy")
    env = TABSBattleSimulatorAutoResetWrapper(env)
    env = TABSBattleSimulatorLogWrapper(env)

    num_iterations = config.total_env_step // (
        config.n_env * config.mappo.rollout_step * config.log_interval
    )

    mappo = MAPPO(config.mappo, env)
    train_state = mappo.init_train_state(
        jax.random.key(config.seed), num_iterations * config.log_interval
    )

    v_vectorize_scenario = jax.vmap(
        partial(
            get_vectorized_scenario,
            n_ally=config.tabs.max_n_ally,
            n_enemy=config.tabs.max_n_enemy,
        )
    )
    vectorized_scenario: VectorizedScenario = v_vectorize_scenario(repeated_scenarios)

    def train_body(carry, _):
        train_state = carry
        train_state, rollout_result = mappo.rollout(train_state, repeated_scenarios)
        train_state, train_info = mappo.train(train_state, rollout_result)

        last_state = rollout_result["last_state"]

        battle_metric = {
            "returned_cumulative_is_attackings": last_state.returned_cumulative_is_attackings,
            "returned_cumulative_damage_dealts": last_state.returned_cumulative_damage_dealts,
            "returned_cumulative_attack_success": last_state.returned_cumulative_attack_success,
        }

        ally_battle_metric = jax.tree.map(lambda x: x[:, : config.tabs.max_n_ally], battle_metric)
        ally_is_disabled = vectorized_scenario.is_disabled[:, : config.tabs.max_n_ally]
        ally_unit_ids = vectorized_scenario.unit_ids[:, : config.tabs.max_n_ally]

        def unit_condition_sum(target_unit_id, unit_ids, is_disabled, values):
            return ((unit_ids == target_unit_id) * (1 - is_disabled) * values).sum()

        unit_battle_metric = jax.tree.map(
            lambda x: jax.vmap(unit_condition_sum, in_axes=(0, None, None, None))(
                jnp.arange(config.tabs.max_num_units), ally_unit_ids, ally_is_disabled, x
            ),
            ally_battle_metric,
        )

        unit_counts = jax.vmap(unit_condition_sum, in_axes=(0, None, None, None))(
            jnp.arange(config.tabs.max_num_units),
            ally_unit_ids,
            ally_is_disabled,
            jnp.ones_like(ally_is_disabled),
        )

        unit_battle_metric["attack_success_rate"] = (
            unit_battle_metric["returned_cumulative_attack_success"]
            / unit_battle_metric["returned_cumulative_is_attackings"]
        )

        unit_specific_metric = {}
        for key, value in unit_battle_metric.items():
            for i, name in enumerate(ALL_UNIT_NAMES):
                unit_specific_metric[f"{key}/{name}"] = value[i] / (
                    unit_counts[i] if key != "attack_success_rate" else 1
                )

        team_fight_metric = {
            "cumulative_is_attackings": unit_battle_metric[
                "returned_cumulative_is_attackings"
            ].sum()
            / config.n_env,
            "cumulative_damage_dealts": jnp.nansum(
                unit_battle_metric["returned_cumulative_damage_dealts"]
                * (unit_battle_metric["returned_cumulative_damage_dealts"] > 0)
            )
            / config.n_env,
            "cumulative_heal_amount": jnp.nansum(
                unit_battle_metric["returned_cumulative_damage_dealts"]
                * (unit_battle_metric["returned_cumulative_damage_dealts"] < 0)
            )
            / config.n_env,
            "attack_success_rate": jnp.nansum(
                unit_battle_metric["returned_cumulative_attack_success"]
            )
            / jnp.nansum(unit_battle_metric["returned_cumulative_is_attackings"]),
            "first_kill_rate": last_state.returned_first_kills[:, 0].mean(),
        }

        train_info.update(unit_specific_metric)
        train_info.update(team_fight_metric)

        return train_state, train_info

    for step in tqdm(range(num_iterations)):
        result = jax.lax.scan(train_body, train_state, None, config.log_interval)
        train_state, train_info = result
        mappo.save_state(train_state, config.save_path + f"/{step}")

        train_info["returned_episode_returns"] = (
            train_info["returned_episode_returns"][:, :, 0].mean(axis=1).flatten()
        )
        train_info["returned_episode_lengths"] = (
            train_info["returned_episode_lengths"][:, :, 0].mean(axis=1).flatten()
        )
        train_info["returned_episode_wins"] = (
            train_info["returned_episode_wins"][:, :, 0].mean(axis=1).flatten()
        )

        for i in range(config.log_interval):
            wandb.log(jax.tree.map(lambda x: x[i], train_info))

        np_result = jax.tree.map(lambda x: np.array(x).tolist(), train_info)
        with open(os.path.join(logs_dir, f"result_{step}.json"), "w") as f:
            json.dump(np_result, f)
