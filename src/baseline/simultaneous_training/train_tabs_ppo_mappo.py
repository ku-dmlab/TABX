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
from src.tabs.config import TABSConfig


@dataclass
class Config:
    seed: int = 42
    n_env: int = 64  # the number of environments to run in parallel
    tabs: TABSConfig = TABSConfig()
    comb: PPOConfig = PPOConfig(
        rollout_step=tabs.max_n_ally, n_env=n_env, entropy_coef=0.1, batch_size=32
    )
    deploy: PPOConfig = PPOConfig(
        rollout_step=tabs.max_n_ally, n_env=n_env, entropy_coef=0.1, batch_size=32
    )
    battle: PPOConfig = PPOConfig(rollout_step=512, n_env=n_env, batch_size=n_env)
    base_path: str = "./ckpt/tabs_st_ppo_mappo"
    project_name: str = "tabs_st_ppo_mappo"
    gpu_id: int = 0
    iter_per_step: int = 100
    total_train_iter: int = 10
    debug: bool = False


if __name__ == "__main__":
    config = tyro.cli(Config)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(config.gpu_id)

    import jax
    import jax.numpy as jnp

    from src.baseline.algorithm import PPO, MAPPO
    from src.tabs.wrappers import (
        TABSBattleSimulatorLogWrapper,
        TABSBattleSimulatorHeuristicWrapper,
        TABSBattleSimulatorAutoResetWrapper,
    )
    from src.tabs.scenarios import generate_scenario, TABSConfig, pprint_grid_with_units
    from src.tabs import TABSUnitComb, TABSUnitDeploy, TABSBattleSimulator
    from src.baseline.utils import (
        get_abs_path,
        dataclass_to_dict,
    )
    from src.baseline.train_utils import rollout_tabs

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
    config.tabs = replace(config.tabs, max_n_enemy=int(scenario.enemy_unit_comp.sum().item()))
    # Duplicate the scenario for parallel runs
    repeated_scenarios = jax.tree.map(lambda x: jnp.repeat(x[None], config.n_env, axis=0), scenario)

    # Envrionments
    env_unit_comb = TABSUnitComb(config.tabs)
    env_unit_deploy = TABSUnitDeploy(config.tabs)
    env_bs = TABSBattleSimulator(config.tabs)
    env_bs = TABSBattleSimulatorHeuristicWrapper(env_bs)
    env_bs = TABSBattleSimulatorAutoResetWrapper(env_bs)
    env_bs = TABSBattleSimulatorLogWrapper(env_bs)

    # Agents
    unit_comb_agent = PPO(config.comb, env_unit_comb)
    unit_deploy_agent = PPO(config.deploy, env_unit_deploy)
    battle_agent = MAPPO(config.battle, env_bs)

    key = jax.random.key(config.seed)
    key_comb, key_deploy, key_bs = jax.random.split(key, 3)
    train_state_comb = unit_comb_agent.init_train_state(
        key_comb, config.iter_per_step * config.total_train_iter
    )
    train_state_deploy = unit_deploy_agent.init_train_state(
        key_deploy, config.iter_per_step * config.total_train_iter
    )
    train_state_bs = battle_agent.init_train_state(
        key_bs, config.iter_per_step * config.total_train_iter
    )

    @jax.jit
    def train_fn(carry):
        def train_body(carry, _):
            train_state_comb, train_state_deploy, train_state_bs = carry

            # Rollout
            (train_state_comb, train_state_deploy, train_state_bs), rollout_result = rollout_tabs(
                unit_comb_agent,
                unit_deploy_agent,
                battle_agent,
                train_state_comb,
                train_state_deploy,
                train_state_bs,
                repeated_scenarios,
                config,
            )

            # Update policies
            train_state_comb, train_info_comb = unit_comb_agent.train(
                train_state_comb, rollout_result["rollout_result_comb"]
            )

            train_state_deploy, train_info_deploy = unit_deploy_agent.train(
                train_state_deploy, rollout_result["rollout_result_deploy"]
            )

            train_state_bs, train_info_bs = battle_agent.train(
                train_state_bs, rollout_result["rollout_result_bs"]
            )

            result = {
                "train_info_comb": train_info_comb,
                "train_info_deploy": train_info_deploy,
                "train_info_bs": train_info_bs,
                "metric": rollout_result["metric"],
            }

            return (train_state_comb, train_state_deploy, train_state_bs), result

        carry, result = jax.lax.scan(train_body, carry, None, config.iter_per_step)

        return carry, result

    # Create logs directory if it doesn't exist
    logs_dir = get_abs_path(config.save_path + "/logs")
    os.makedirs(logs_dir, exist_ok=True)

    # Simultaneous training for the end-to-end agent
    carry = (train_state_comb, train_state_deploy, train_state_bs)
    for step in tqdm(range(config.total_train_iter)):
        # Train on TABS
        carry, result = train_fn(carry)

        # Log metrics
        for i in range(config.iter_per_step):
            # Log comb metrics
            wandb_log = {}
            for result_key, result_value in result.items():
                for key_, value in result_value.items():
                    if key_ == "unit_deploy":
                        wandb_log[f"{result_key}/{key_}"] = wandb.Html(
                            f"<pre>{pprint_grid_with_units(value[i])}</pre>"
                        )
                        continue
                    wandb_log[f"{result_key}/{key_}"] = jax.tree.map(lambda x: x[i], value)
            wandb.log(wandb_log)

        # Save models
        unit_comb_agent.save_state(carry[0], os.path.join(config.save_path, f"comb/ppo/{step}"))
        unit_deploy_agent.save_state(carry[1], os.path.join(config.save_path, f"deploy/ppo/{step}"))
        battle_agent.save_state(carry[2], os.path.join(config.save_path, f"bs/mappo/{step}"))

        np_result = jax.tree.map(lambda x: np.array(x).tolist(), result)
        with open(os.path.join(logs_dir, f"result_{step}.json"), "w") as f:
            json.dump(np_result, f)
