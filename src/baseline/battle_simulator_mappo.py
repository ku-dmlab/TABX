from dataclasses import dataclass, replace
from tqdm import tqdm
import tyro
import wandb
import hashlib
from src.baseline.configs.config import PPOConfig
from src.tabs.scenarios import TABSConfig


@dataclass
class Config:
    seed: int = 42
    n_env: int = 128  # the number of environments to run in parallel
    tabs: TABSConfig = TABSConfig()
    mappo: PPOConfig = PPOConfig(n_env=n_env, seed=seed)
    save_path: str = "/save"
    gpu_id: int = 3
    total_env_step: int = int(2e8)
    log_interval: int = 100


if __name__ == "__main__":
    config = tyro.cli(Config)
    import os

    os.environ["CUDA_VISIBLE_DEVICES"] = str(config.gpu_id)
    import jax
    import jax.numpy as jnp
    from src.baseline.algorithm import MAPPO
    from src.tabs.wrappers import (
        TABSBattleSimulatorAutoResetWrapper,
        TABSBattleSimulatorLogWrapper,
        TABSBattleSimulatorHeuristicWrapper,
    )
    from src.tabs import TABSBattleSimulator
    from src.tabs.scenarios import TABSConfig, generate_scenario
    import datetime

    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    config_hash = hashlib.md5(str(config).encode()).hexdigest()[:8]
    config.save_path = f"/save/{config.tabs.scenario_name}_{current_time}_{config_hash}"
    wandb.init(project="battle_simulator_mappo", config=config)

    tabs_conf = config.tabs
    scenario = generate_scenario(tabs_conf)
    repeated_scenarios = jax.tree.map(lambda x: jnp.repeat(x[None], config.n_env, axis=0), scenario)
    tabs_conf = replace(
        tabs_conf,
        max_n_ally=int(scenario.ally_unit_comp.sum().item()),
        max_n_enemy=int(scenario.enemy_unit_comp.sum().item()),
    )  # For avoiding inefficient instantiation of units

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

    def train_body(carry, _):
        train_state = carry
        train_state, rollout_result = mappo.rollout(train_state, repeated_scenarios)
        train_state, train_info = mappo.train(train_state, rollout_result)
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
