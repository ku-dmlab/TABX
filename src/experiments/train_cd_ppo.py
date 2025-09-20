import os
import json
import datetime
from functools import partial
from dataclasses import dataclass, replace

from tqdm import tqdm
import tyro
import wandb
import hashlib
import numpy as np

from src.baseline.configs.config import PPOConfig
from src.tabs.config import TABSConfig, TABSHeuristicConfig
from src.tabs.constants import ALL_UNIT_NAMES


@dataclass
class Config:
    seed: int = 42
    n_env: int = 64  # the number of environments to run in parallel
    tabs: TABSConfig = TABSConfig()
    enemy_heuristic_config: TABSHeuristicConfig = TABSHeuristicConfig()
    initial_ally_heuristic_config: TABSHeuristicConfig = TABSHeuristicConfig(
        epsilon=0.5,
        aggressive_threshold=0.1,
        rotate_noise_scale=1.0,
        healer_rotate_noise_scale=0.5,
        healer_aggressive_threshold=0.0,
    )
    end_ally_heuristic_config: TABSHeuristicConfig = TABSHeuristicConfig(
        epsilon=0.0,
        aggressive_threshold=0.6,
        rotate_noise_scale=0.0,
        healer_rotate_noise_scale=0.0,
        healer_aggressive_threshold=1.0,
    )
    comb_ppo: PPOConfig = PPOConfig(
        rollout_step=tabs.max_n_ally, n_env=n_env, entropy_coef=1.0, batch_size=32
    )
    deploy_ppo: PPOConfig = PPOConfig(
        rollout_step=tabs.max_n_ally, n_env=n_env, entropy_coef=1.0, batch_size=32
    )
    base_path: str = "ckpt/tabs_cd_ppo"
    project_name: str = "tabs_cd_ppo"
    gpu_id: int = 0

    total_iter: int = 10
    iter_per_train_step: int = 150
    battle_simulator_rollout_step: int = 512

    debug: bool = False


if __name__ == "__main__":
    config = tyro.cli(Config)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(config.gpu_id)

    import jax
    import jax.numpy as jnp

    from src.baseline.algorithm import PPO
    from src.tabs.wrappers import (
        TABSBattleSimulatorHeuristicWrapper,
        TABSBattleSimulatorLogWrapper,
    )
    from src.tabs import TABSUnitComb, TABSUnitDeploy, TABSBattleSimulator
    from src.tabs.scenarios import generate_scenario, TABSConfig
    from src.tabs.scenarios import pprint_grid_with_units
    from src.baseline.utils import get_abs_path, dataclass_to_dict
    from src.tabs.tabs_battle_simulator.heuristic_policy import heuristic_policy

    # Create a hash of the config for unique folder naming
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    config_dict = dataclass_to_dict(config)
    config_str = json.dumps(config_dict, sort_keys=True)
    config_hash = hashlib.md5(config_str.encode()).hexdigest()[:8]
    config.save_path = os.path.join(
        config.base_path, f"{config.tabs.scenario_name}_{current_time}_{config_hash}"
    )

    wandb.init(
        project=config.project_name,
        config=config,
        mode="online" if not config.debug else "disabled",
    )

    scenario = generate_scenario(config.tabs)
    config.tabs = replace(config.tabs, max_n_enemy=int(scenario.enemy_unit_comp.sum().item()))
    repeated_scenario = jax.tree.map(lambda x: jnp.repeat(x[None], config.n_env, axis=0), scenario)

    # Environments
    env_unit_deploy = TABSUnitDeploy(config.tabs)
    env_unit_comb = TABSUnitComb(config.tabs)

    # Agents
    ppo_unit_deploy = PPO(config.deploy_ppo, env_unit_deploy)
    ppo_unit_comb = PPO(config.comb_ppo, env_unit_comb)

    deploy_key, comb_key = jax.random.split(jax.random.key(config.seed))
    unit_deploy_train_state = ppo_unit_deploy.init_train_state(
        deploy_key, config.iter_per_train_step * config.total_iter
    )
    unit_comb_train_state = ppo_unit_comb.init_train_state(
        comb_key, config.iter_per_train_step * config.total_iter
    )

    battle_simulator = TABSBattleSimulator(config.tabs)
    battle_simulator = TABSBattleSimulatorHeuristicWrapper(
        battle_simulator, "enemy", config.enemy_heuristic_config
    )
    battle_simulator = TABSBattleSimulatorLogWrapper(battle_simulator, reset_when_done=False)

    bs_v_reset = jax.vmap(battle_simulator.reset, in_axes=(0, 0))
    bs_v_step = jax.vmap(partial(battle_simulator.step), in_axes=(0, 0, 0))

    def get_ally_heuristic_config(step):
        weight = jnp.clip(1 - step / (config.total_iter * config.iter_per_train_step), 0.0, 1.0)

        return TABSHeuristicConfig(
            epsilon=weight * config.initial_ally_heuristic_config.epsilon
            + (1 - weight) * config.end_ally_heuristic_config.epsilon,
            aggressive_threshold=weight * config.initial_ally_heuristic_config.aggressive_threshold
            + (1 - weight) * config.end_ally_heuristic_config.aggressive_threshold,
            rotate_noise_scale=weight * config.initial_ally_heuristic_config.rotate_noise_scale
            + (1 - weight) * config.end_ally_heuristic_config.rotate_noise_scale,
            healer_rotate_noise_scale=weight
            * config.initial_ally_heuristic_config.healer_rotate_noise_scale
            + (1 - weight) * config.end_ally_heuristic_config.healer_rotate_noise_scale,
            healer_aggressive_threshold=weight
            * config.initial_ally_heuristic_config.healer_aggressive_threshold
            + (1 - weight) * config.end_ally_heuristic_config.healer_aggressive_threshold,
        )

    def battle_simulator_rollout(key, deployed_scenario, step):
        key, reset_key = jax.random.split(key)
        obs, state = bs_v_reset(jax.random.split(reset_key, config.n_env), deployed_scenario)

        def rollout_body(carry, key):
            key, obs, state = carry

            step_key, action_key, key = jax.random.split(key, 3)
            ally_actions = jax.tree.map(
                lambda obs: jax.vmap(
                    partial(
                        heuristic_policy,
                        num_agents=battle_simulator.num_agents,
                        heuristic_config=get_ally_heuristic_config(step),
                    ),
                    in_axes=(0, 0),
                )(jax.random.split(action_key, config.n_env), obs),
                obs,
            )
            next_obs, next_state, reward, done, info = bs_v_step(
                jax.random.split(step_key, config.n_env), state, ally_actions
            )

            return (key, next_obs, next_state), info

        carry = (key, obs, state)
        _, info = jax.lax.scan(rollout_body, carry, None, config.battle_simulator_rollout_step)

        result = {
            "episode_lengths": info["episode_lengths"][-1],
            "episode_returns": info["episode_returns"][-1] + info["returned_episode_wins"][-1],
            "episode_wins": info["returned_episode_wins"][-1],
        }

        return result

    def train_body(carry, _):
        (unit_deploy_train_state, unit_comb_train_state, key, step) = carry

        key, new_key = jax.random.split(key)

        # Combine and deploy units
        unit_comb_train_state, comb_rollout_result = ppo_unit_comb.rollout(
            unit_comb_train_state, repeated_scenario
        )
        comb_scenario = repeated_scenario.replace(
            ally_unit_comp=comb_rollout_result["last_state"].current_unit_list
        )
        unit_deploy_train_state, deploy_rollout_result = ppo_unit_deploy.rollout(
            unit_deploy_train_state, comb_scenario
        )
        deployed_scenario = comb_scenario.replace(
            battle_field=deploy_rollout_result["last_state"].battle_field
        )

        # Rollout battle simulator using the deployed scenario
        bs_rollout_result = battle_simulator_rollout(key, deployed_scenario, step)

        # Set rewards for deploy and comb based on the battle simulator rollout
        deploy_rewards = (
            deploy_rollout_result["dones"] * bs_rollout_result["episode_returns"][None, :, 0]
        )
        deploy_rollout_result["rewards"] = deploy_rewards
        comb_rewards = (
            comb_rollout_result["dones"] * bs_rollout_result["episode_returns"][None, :, 0]
        )
        comb_rollout_result["rewards"] = comb_rewards

        bs_rollout_result["episode_lengths"] = bs_rollout_result["episode_lengths"][:, 0].mean()
        bs_rollout_result["episode_wins"] = bs_rollout_result["episode_wins"][:, 0].mean()
        bs_rollout_result["episode_returns"] = bs_rollout_result["episode_returns"][:, 0].mean()
        bs_rollout_result["episode_returns"] += bs_rollout_result["episode_wins"]

        unit_deploy_train_state, deploy_train_info = ppo_unit_deploy.train(
            unit_deploy_train_state, deploy_rollout_result
        )

        unit_comb_train_state, comb_train_info = ppo_unit_comb.train(
            unit_comb_train_state, comb_rollout_result
        )

        # Ror logging
        comb_train_info["budget"] = comb_rollout_result["last_state"].budget.mean()
        comb_train_info["unit_list"] = comb_rollout_result["last_state"].current_unit_list.mean(
            axis=0
        )

        for i, name in enumerate(ALL_UNIT_NAMES):
            comb_train_info[f"unit_count/{name}"] = comb_rollout_result[
                "last_state"
            ].current_unit_list.mean(axis=0)[i]

        deploy_train_info["unit_deploy"] = deploy_rollout_result["last_state"].battle_field[0]

        result = {
            "comb_train_info": comb_train_info,
            "deploy_train_info": deploy_train_info,
            "bs_rollout_result": bs_rollout_result,
        }

        return (unit_deploy_train_state, unit_comb_train_state, new_key, step + 1), result

    logs_dir = get_abs_path(config.save_path)
    os.makedirs(logs_dir, exist_ok=True)

    # Save config to logs directory
    with open(os.path.join(logs_dir, "config.json"), "w") as f:
        json.dump(config_dict, f, indent=2)

    @jax.jit
    def train_fn(unit_deploy_train_state, unit_comb_train_state, key, step):
        carry = (unit_deploy_train_state, unit_comb_train_state, key, step)
        (unit_deploy_train_state, unit_comb_train_state, key, step), result = jax.lax.scan(
            train_body, carry, None, config.iter_per_train_step
        )

        return (unit_deploy_train_state, unit_comb_train_state, key, step), result

    key = jax.random.key(config.seed)
    step = jnp.array(0)
    for train_step in tqdm(range(config.total_iter)):
        (unit_deploy_train_state, unit_comb_train_state, key, step), result = train_fn(
            unit_deploy_train_state, unit_comb_train_state, key, step
        )

        for i in range(config.iter_per_train_step):
            log_data = {}
            for key_, value in result["comb_train_info"].items():
                log_data[f"comb/{key_}"] = jax.tree.map(lambda x: x[i], value)
            for key_, value in result["deploy_train_info"].items():
                if key_ == "unit_deploy":
                    log_data[f"deploy/{key_}"] = wandb.Html(
                        f"<pre>{pprint_grid_with_units(value[i])}</pre>"
                    )
                    continue
                log_data[f"deploy/{key_}"] = jax.tree.map(lambda x: x[i], value)

            for key_, value in result["bs_rollout_result"].items():
                log_data[f"bs/{key_}"] = jax.tree.map(lambda x: x[i], value)

            wandb.log(log_data)

        ppo_unit_comb.save_state(
            unit_comb_train_state, os.path.join(config.save_path, f"comb/ppo/{train_step}")
        )
        ppo_unit_deploy.save_state(
            unit_deploy_train_state, os.path.join(config.save_path, f"deploy/ppo/{train_step}")
        )

        np_result = jax.tree.map(lambda x: np.array(x).tolist(), result)
        with open(os.path.join(logs_dir, f"result_{train_step}.json"), "w") as f:
            json.dump(np_result, f)
