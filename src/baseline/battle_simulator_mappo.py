from dataclasses import dataclass
from tqdm import tqdm
import tyro
import json
import wandb
import hashlib


@dataclass
class Config:
    seed: int = 42
    n_env: int = 64  # the number of environments to run in parallel
    rollout_step: int = 1024  # the number of rollouts to run in parallel
    gamma: float = 0.99  # the discount factor
    lamda: float = 0.95  # the lambda for GAE
    clip_value: float = 1.0  # the clip value for PPO
    clip_ratio: float = 0.05  # the clip ratio for PPO
    entropy_coef: float = 0.01  # the entropy coefficient
    ppo_epochs: int = 10  # the number of epochs to update the policy and value function
    max_grad_norm: float = 0.5  # the maximum gradient norm
    lr: float = 1e-3  # the learning rate
    layer_dim = 256

    scenario: str = "8A_vs_1A1M1H"

    save_path: str = "/save"
    gpu_id: int = 3

    total_env_step: int = int(2e8)
    log_step: int = 100


if __name__ == "__main__":
    config = tyro.cli(Config)

    import os
    from functools import partial

    os.environ["CUDA_VISIBLE_DEVICES"] = str(config.gpu_id)

    import jax

    from src.baseline.algorithm import MAPPO
    from src.tabs.wrappers import (
        TABSBattleSimulatorAutoResetWrapper,
        TABSBattleSimulatorLogWrapper,
        TABSBattleSimulatorHeuristicWrapper,
    )
    from src.tabs import TABSBattleSimulator
    from src.tabs.scenarios import TABSConf, generate_scenario

    # Create a hash of the config for unique folder naming
    config_dict = {k: v for k, v in vars(config).items() if not k.startswith("_")}
    config_str = json.dumps(config_dict, sort_keys=True)
    config_hash = hashlib.md5(config_str.encode()).hexdigest()[:8]
    config.save_path = f"/save/{config.scenario}_{config_hash}"

    wandb.init(project="battle_simulator_mappo", config=config)

    tabs_conf = TABSConf().replace(scenario_name=config.scenario)
    scenario = generate_scenario(tabs_conf)
    tabs_conf = tabs_conf.replace(
        max_n_ally=int(scenario.ally_unit_comp.sum().item()),
        max_n_enemy=int(scenario.enemy_unit_comp.sum().item()),
    )  # For avoiding inefficient instantiation of units

    env = TABSBattleSimulator(tabs_conf)
    env = TABSBattleSimulatorHeuristicWrapper(env, "enemy")
    env = TABSBattleSimulatorAutoResetWrapper(env, scenario)
    env = TABSBattleSimulatorLogWrapper(env)

    mappo = MAPPO(config, env)
    train_state = mappo.init_train_state()
    train_fn = jax.jit(partial(mappo.train, step=config.log_step))

    for step in tqdm(
        range(config.total_env_step // (config.n_env * config.rollout_step * config.log_step))
    ):
        result = train_fn(train_state)
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

        for i in range(config.log_step):
            wandb.log(jax.tree.map(lambda x: x[i], train_info))
