import os
import json
import datetime
from dataclasses import dataclass, replace

from tqdm import tqdm
import tyro
import wandb
import hashlib
import numpy as np

from src.baseline.configs.config import PPOConfig
from src.tabs.scenarios import TABSConfig


@dataclass
class Config:
    seed: int = 42
    n_env: int = 64  # the number of environments to run in parallel
    tabs: TABSConfig = TABSConfig()
    battle: PPOConfig = PPOConfig(rollout_step=512, n_env=n_env, batch_size=n_env)
    base_path: str = "./ckpt/tabs_bs_mappo"
    project_name: str = "tabs_bs_mappo"
    gpu_id: int = 0
    total_iter: int = 50
    iter_per_train_step: int = 10
    debug: bool = False


if __name__ == "__main__":
    config = tyro.cli(Config)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(config.gpu_id)

    import jax
    import jax.numpy as jnp

    from src.baseline.algorithm import MAPPO
    from src.tabs.wrappers import (
        TABSBattleSimulatorLogWrapper,
        TABSBattleSimulatorHeuristicWrapper,
        TABSBattleSimulatorAutoResetWrapper,
    )
    from src.tabs import TABSBattleSimulator
    from src.tabs.scenarios import generate_scenario
    from src.baseline.utils import dataclass_to_dict, get_abs_path
    from src.baseline.train_utils import get_battle_metric

    # Create a hash of the config for unique folder naming
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    config_dict = dataclass_to_dict(config)
    config_str = json.dumps(config_dict, sort_keys=True)
    config_hash = hashlib.md5(config_str.encode()).hexdigest()[:8]
    config.save_path = os.path.join(
        config.base_path, f"{config.tabs.scenario_name}_{current_time}_{config_hash}"
    )

    logs_dir = get_abs_path(config.save_path)
    os.makedirs(logs_dir, exist_ok=True)

    # Save config to logs directory
    with open(os.path.join(logs_dir, "config.json"), "w") as f:
        json.dump(config_dict, f, indent=2)

    wandb.init(
        project=config.project_name,
        config=config,
        mode="online" if not config.debug else "disabled",
    )

    scenario = generate_scenario(config.tabs)
    config.tabs = replace(
        config.tabs,
        max_n_ally=int(scenario.ally_unit_comp.sum().item()),
        max_n_enemy=int(scenario.enemy_unit_comp.sum().item()),
    )
    # Duplicate the scenario for parallel runs
    repeated_scenarios = jax.tree.map(lambda x: jnp.repeat(x[None], config.n_env, axis=0), scenario)

    # Environments
    env = TABSBattleSimulator(config.tabs)
    env = TABSBattleSimulatorHeuristicWrapper(env)
    env = TABSBattleSimulatorAutoResetWrapper(env)
    env = TABSBattleSimulatorLogWrapper(env)

    # Agent
    mappo = MAPPO(config.battle, env)
    train_state = mappo.init_train_state(
        jax.random.key(config.seed), config.total_iter * config.iter_per_train_step
    )

    # Create logs directory if it doesn't exist
    logs_dir = get_abs_path(config.save_path + "/logs")
    os.makedirs(logs_dir, exist_ok=True)

    @jax.jit
    def train_fn(carry):
        def train_body(carry, _):
            train_state = carry
            train_state, rollout_result = mappo.rollout(train_state, repeated_scenarios)
            train_state, train_info = mappo.train(train_state, rollout_result)

            metric = get_battle_metric(config, rollout_result["last_state"], repeated_scenarios)
            metric.update(
                {
                    "episode_returns": rollout_result["returned_episode_returns"][:, 0].mean(),
                    "episode_lengths": rollout_result["returned_episode_lengths"][:, 0].mean(),
                    "episode_wins": rollout_result["returned_episode_wins"][:, 0].mean(),
                    "reward_sum": rollout_result["common_reward"].sum() / config.n_env,
                }
            )

            result = {"train_info": train_info, "metric": metric}

            return train_state, result

        carry, result = jax.lax.scan(train_body, carry, None, config.iter_per_train_step)

        return carry, result

    carry = train_state
    for step in tqdm(range(config.total_iter)):
        carry, result = train_fn(carry)

        # Log metrics
        for i in range(config.iter_per_step):
            # Log comb metrics
            wandb_log = {}
            for result_key, result_value in result.items():
                for key_, value in result_value.items():
                    wandb_log[f"{result_key}/{key_}"] = jax.tree.map(lambda x: x[i], value)
            wandb.log(wandb_log)

        mappo.save_state(carry[0], config.save_path + f"bs/mappo/{step}")

        np_result = jax.tree.map(lambda x: np.array(x).tolist(), result)
        with open(os.path.join(logs_dir, f"result_{step}.json"), "w") as f:
            json.dump(np_result, f)
