import hashlib
import json
from dataclasses import dataclass, replace

import numpy as np
import tyro
from src.baseline.configs.config import PQNConfig
from tqdm import tqdm

import wandb
from src.tabs.config import TABSConfig
from src.tabs.constants import ALL_UNIT_NAMES


@dataclass
class Config:
    seed: int = 42
    n_env: int = 64  # the number of environments to run in parallel
    tabs: TABSConfig = TABSConfig()
    comb_pqn: PQNConfig = PQNConfig(rollout_step=tabs.max_n_ally, n_env=n_env, batch_size=32)
    deploy_pqn: PQNConfig = PQNConfig(rollout_step=tabs.max_n_ally, n_env=n_env, batch_size=32)
    save_path: str = "/save"
    gpu_id: int = 3

    total_iter: int = 10
    iter_per_train_step: int = 100
    battle_simulator_rollout_step: int = 512


if __name__ == "__main__":
    config = tyro.cli(Config)

    import os
    from functools import partial

    os.environ["CUDA_VISIBLE_DEVICES"] = str(config.gpu_id)

    import datetime

    import jax
    import jax.numpy as jnp
    from src.baseline.algorithm import PQN

    from src.baseline.utils import dataclass_to_dict, get_abs_path
    from src.tabs import TABSBattleSimulator, TABSUnitComb, TABSUnitDeploy
    from src.tabs.scenarios import TABSConfig, generate_scenario, pprint_grid_with_units
    from src.tabs.wrappers import (
        TABSBattleSimulatorHeuristicWrapper,
        TABSBattleSimulatorLogWrapper,
    )

    # Create a hash of the config for unique folder naming
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    config_dict = dataclass_to_dict(config)
    config_str = json.dumps(config_dict, sort_keys=True)
    config_hash = hashlib.md5(config_str.encode()).hexdigest()[:8]
    config.save_path = f"/save/{config.tabs.scenario_name}_{current_time}_{config_hash}"

    wandb.init(project="comb_deploy_pqn", config=config)

    scenario = generate_scenario(config.tabs)
    config.tabs = replace(config.tabs, max_n_enemy=int(scenario.enemy_unit_comp.sum().item()))
    repeated_scenario = jax.tree.map(lambda x: jnp.repeat(x[None], config.n_env, axis=0), scenario)

    env_unit_deploy = TABSUnitDeploy(config.tabs)
    env_unit_comb = TABSUnitComb(config.tabs)

    pqn_unit_deploy = PQN(config.deploy_pqn, env_unit_deploy)
    pqn_unit_comb = PQN(config.comb_pqn, env_unit_comb)

    deploy_key, comb_key = jax.random.split(jax.random.key(config.seed))

    unit_deploy_train_state = pqn_unit_deploy.init_train_state(
        deploy_key, config.iter_per_train_step * config.total_iter
    )
    unit_comb_train_state = pqn_unit_comb.init_train_state(
        comb_key, config.iter_per_train_step * config.total_iter
    )

    battle_simulator = TABSBattleSimulator(config.tabs)
    battle_simulator = TABSBattleSimulatorHeuristicWrapper(battle_simulator, "all")
    battle_simulator = TABSBattleSimulatorLogWrapper(battle_simulator, reset_when_done=False)

    bs_v_reset = jax.vmap(battle_simulator.reset, in_axes=(0, 0))
    bs_v_step = jax.vmap(partial(battle_simulator.step, action={}), in_axes=(0, 0))

    def battle_simulator_rollout(key, deployed_scenario):
        key, reset_key = jax.random.split(key)
        obs, state = bs_v_reset(jax.random.split(reset_key, config.n_env), deployed_scenario)

        def rollout_body(carry, key):
            key, state = carry

            step_key, key = jax.random.split(key)
            obs, state, reward, done, info = bs_v_step(
                jax.random.split(step_key, config.n_env), state
            )

            return (key, state), info

        carry = (key, state)
        _, info = jax.lax.scan(rollout_body, carry, None, config.battle_simulator_rollout_step)

        result = {
            "episode_lengths": info["episode_lengths"][-1],
            "episode_returns": info["episode_returns"][-1] + info["returned_episode_wins"][-1],
            "episode_wins": info["returned_episode_wins"][-1],
        }

        return result

    def rollout(
        unit_deploy_train_state, unit_comb_train_state, key, repeated_scenario, greedy=False
    ):
        unit_comb_train_state, comb_rollout_result = pqn_unit_comb.rollout(
            unit_comb_train_state, repeated_scenario, greedy=greedy
        )
        comb_scenario = repeated_scenario.replace(
            ally_unit_comp=comb_rollout_result["last_state"].current_unit_list
        )
        unit_deploy_train_state, deploy_rollout_result = pqn_unit_deploy.rollout(
            unit_deploy_train_state, comb_scenario, greedy=greedy
        )
        deployed_scenario = comb_scenario.replace(
            battle_field=deploy_rollout_result["last_state"].battle_field
        )

        # Rollout battle simulator using the deployed scenario
        bs_rollout_result = battle_simulator_rollout(key, deployed_scenario)

        # Set rewards for deploy and comb based on the battle simulator rollout
        deploy_rewards = (
            deploy_rollout_result["dones"] * bs_rollout_result["episode_returns"][None, :, 0]
        )
        deploy_rollout_result["rewards"] = deploy_rewards
        comb_rewards = (
            comb_rollout_result["dones"] * bs_rollout_result["episode_returns"][None, :, 0]
        )
        comb_rollout_result["rewards"] = comb_rewards

        metric = {
            "episode_lengths": bs_rollout_result["episode_lengths"][:, 0].mean(),
            "episode_wins": bs_rollout_result["episode_wins"][:, 0].mean(),
            "episode_returns": bs_rollout_result["episode_returns"][:, 0].mean(),
        }
        metric["budget"] = comb_rollout_result["last_state"].budget.mean()
        metric["unit_list"] = comb_rollout_result["last_state"].current_unit_list.mean(axis=0)

        for i, name in enumerate(ALL_UNIT_NAMES):
            metric[f"unit_count/{name}"] = comb_rollout_result["last_state"].current_unit_list.mean(
                axis=0
            )[i]

        metric["unit_deploy"] = deploy_rollout_result["last_state"].battle_field[0]

        return (
            (unit_deploy_train_state, unit_comb_train_state, key),
            deploy_rollout_result,
            comb_rollout_result,
            bs_rollout_result,
            metric,
        )

    def train_body(carry, _):
        (unit_deploy_train_state, unit_comb_train_state, key) = carry

        key, new_key = jax.random.split(key)

        # Combine and deploy units
        (
            (unit_deploy_train_state, unit_comb_train_state, key),
            deploy_rollout_result,
            comb_rollout_result,
            bs_rollout_result,
            metric,
        ) = rollout(unit_deploy_train_state, unit_comb_train_state, key, repeated_scenario)

        ((unit_deploy_train_state, unit_comb_train_state, key), _, _, _, test_metric) = rollout(
            unit_deploy_train_state, unit_comb_train_state, key, repeated_scenario, greedy=True
        )

        unit_deploy_train_state, deploy_train_info = pqn_unit_deploy.train(
            unit_deploy_train_state, deploy_rollout_result
        )

        unit_comb_train_state, comb_train_info = pqn_unit_comb.train(
            unit_comb_train_state, comb_rollout_result
        )

        result = {
            "comb_train_info": comb_train_info,
            "deploy_train_info": deploy_train_info,
            "bs_rollout_result": bs_rollout_result,
            "metric": metric,
            "test_metric": test_metric,
        }

        return (unit_deploy_train_state, unit_comb_train_state, new_key), result

    logs_dir = get_abs_path(config.save_path)
    os.makedirs(logs_dir, exist_ok=True)

    # Save config to logs directory
    with open(os.path.join(logs_dir, "config.json"), "w") as f:
        json.dump(config_dict, f, indent=2)

    @jax.jit
    def train_fn(unit_deploy_train_state, unit_comb_train_state, key):
        carry = (unit_deploy_train_state, unit_comb_train_state, key)
        (unit_deploy_train_state, unit_comb_train_state, key), result = jax.lax.scan(
            train_body, carry, None, config.iter_per_train_step
        )

        return (unit_deploy_train_state, unit_comb_train_state, key), result

    key = jax.random.key(config.seed)
    for step in tqdm(range(config.total_iter)):
        (unit_deploy_train_state, unit_comb_train_state, key), result = train_fn(
            unit_deploy_train_state, unit_comb_train_state, key
        )

        for i in range(config.iter_per_train_step):
            log_data = {}
            for key_, value in result["comb_train_info"].items():
                log_data[f"comb/{key_}"] = jax.tree.map(lambda x: x[i], value)
            for key_, value in result["deploy_train_info"].items():
                log_data[f"deploy/{key_}"] = jax.tree.map(lambda x: x[i], value)

            for key_, value in result["bs_rollout_result"].items():
                log_data[f"bs/{key_}"] = jax.tree.map(lambda x: x[i], value)

            for key_, value in result["metric"].items():
                if key_ == "unit_deploy":
                    log_data[f"metric/{key_}"] = wandb.Html(
                        f"<pre>{pprint_grid_with_units(value[i])}</pre>"
                    )
                    continue
                log_data[f"metric/{key_}"] = jax.tree.map(lambda x: x[i], value)
            for key_, value in result["test_metric"].items():
                if key_ == "unit_deploy":
                    log_data[f"test_metric/{key_}"] = wandb.Html(
                        f"<pre>{pprint_grid_with_units(value[i])}</pre>"
                    )
                    continue
                log_data[f"test_metric/{key_}"] = jax.tree.map(lambda x: x[i], value)

            wandb.log(log_data)

        pqn_unit_comb.save_state(
            unit_comb_train_state, os.path.join(config.save_path, f"comb/pqn/{step}")
        )
        pqn_unit_deploy.save_state(
            unit_deploy_train_state, os.path.join(config.save_path, f"deploy/pqn/{step}")
        )

        np_result = jax.tree.map(lambda x: np.array(x).tolist(), result)
        with open(os.path.join(logs_dir, f"result_{step}.json"), "w") as f:
            json.dump(np_result, f)
