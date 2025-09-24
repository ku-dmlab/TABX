from dataclasses import dataclass, replace
from tqdm import tqdm
import json
import tyro
import wandb
import hashlib
import numpy as np

from src.baseline.configs.config import PPOConfig
from src.tabs.config import TABSConfig
from src.tabs.constants import ALL_UNIT_NAMES


@dataclass
class Config:
    seed: int = 42
    n_env: int = 64  # the number of environments to run in parallel
    tabs: TABSConfig = TABSConfig()
    comb_ppo: PPOConfig = PPOConfig(
        rollout_step=tabs.max_n_ally, n_env=n_env, entropy_coef=0.01, batch_size=32
    )
    deploy_ppo: PPOConfig = PPOConfig(
        rollout_step=tabs.max_n_ally, n_env=n_env, entropy_coef=0.01, batch_size=32
    )
    mappo: PPOConfig = PPOConfig(rollout_step=512, n_env=n_env, batch_size=n_env)
    save_path: str = "/save"
    gpu_id: int = 3
    total_iter: int = 10
    iter_per_comb_deploy_train_step: int = 50
    iter_per_bs_train_step: int = 100
    battle_simulator_rollout_step: int = 512


if __name__ == "__main__":
    config = tyro.cli(Config)

    import os
    from functools import partial

    os.environ["CUDA_VISIBLE_DEVICES"] = str(config.gpu_id)

    import jax
    import jax.numpy as jnp

    from src.baseline.algorithm import PPO, MAPPO
    from src.tabs.wrappers import (
        TABSBattleSimulatorHeuristicWrapper,
        TABSBattleSimulatorLogWrapper,
        TABSBattleSimulatorAutoResetWrapper,
    )
    from src.tabs import TABSUnitComb, TABSUnitDeploy, TABSBattleSimulator
    from src.tabs.scenarios import (
        generate_scenario,
        TABSConfig,
        get_vectorized_scenario,
        VectorizedScenario,
    )
    from src.tabs.scenarios import pprint_grid_with_units
    from src.baseline.utils import get_abs_path, dataclass_to_dict
    import datetime

    # Create a hash of the config for unique folder naming
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    config_dict = dataclass_to_dict(config)
    config_str = json.dumps(config_dict, sort_keys=True)
    config_hash = hashlib.md5(config_str.encode()).hexdigest()[:8]
    config.save_path = f"/save/{config.tabs.scenario_name}_{current_time}_{config_hash}"

    wandb.init(project="comb_deploy_ppo", config=config)

    scenario = generate_scenario(config.tabs)
    config.tabs = replace(config.tabs, max_n_enemy=int(scenario.enemy_unit_comp.sum().item()))
    repeated_scenario = jax.tree.map(lambda x: jnp.repeat(x[None], config.n_env, axis=0), scenario)

    env_unit_deploy = TABSUnitDeploy(config.tabs)
    env_unit_comb = TABSUnitComb(config.tabs)

    ppo_unit_deploy = PPO(config.deploy_ppo, env_unit_deploy)
    ppo_unit_comb = PPO(config.comb_ppo, env_unit_comb)

    deploy_key, comb_key = jax.random.split(jax.random.key(config.seed))

    unit_deploy_train_state = ppo_unit_deploy.init_train_state(
        deploy_key, config.iter_per_comb_deploy_train_step * config.total_iter
    )
    unit_comb_train_state = ppo_unit_comb.init_train_state(
        comb_key, config.iter_per_comb_deploy_train_step * config.total_iter
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

    def train_body(carry, _):
        (unit_deploy_train_state, unit_comb_train_state, key) = carry

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
            train_body, carry, None, config.iter_per_comb_deploy_train_step
        )

        return (unit_deploy_train_state, unit_comb_train_state, key), result

    key = jax.random.key(config.seed)
    for step in tqdm(range(config.total_iter)):
        (unit_deploy_train_state, unit_comb_train_state, key), result = train_fn(
            unit_deploy_train_state, unit_comb_train_state, key
        )

        for i in range(config.iter_per_comb_deploy_train_step):
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
            unit_comb_train_state, os.path.join(config.save_path, f"comb/ppo/{step}")
        )
        ppo_unit_deploy.save_state(
            unit_deploy_train_state, os.path.join(config.save_path, f"deploy/ppo/{step}")
        )

        np_result = jax.tree.map(lambda x: np.array(x).tolist(), result)
        with open(os.path.join(logs_dir, f"result_{step}.json"), "w") as f:
            json.dump(np_result, f)

    env_bs = TABSBattleSimulator(config.tabs)
    env_bs = TABSBattleSimulatorHeuristicWrapper(env_bs)
    env_bs = TABSBattleSimulatorAutoResetWrapper(env_bs)
    env_bs = TABSBattleSimulatorLogWrapper(env_bs)

    mappo_bs = MAPPO(config.mappo, env_bs)
    bs_train_state = mappo_bs.init_train_state(
        key, config.iter_per_bs_train_step * config.total_iter
    )

    def train_bs(carry, _):
        (unit_deploy_train_state, unit_comb_train_state, bs_train_state) = carry

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
        bs_train_state, rollout_result = mappo_bs.rollout(bs_train_state, deployed_scenario)
        bs_train_state, train_info = mappo_bs.train(bs_train_state, rollout_result)

        # logging
        last_state = rollout_result["last_state"]

        battle_metric = {
            "returned_cumulative_is_attackings": last_state.returned_cumulative_is_attackings,
            "returned_cumulative_damage_dealts": last_state.returned_cumulative_damage_dealts,
            "returned_cumulative_attack_success": last_state.returned_cumulative_attack_success,
        }
        v_vectorize_scenario = jax.vmap(
            partial(
                get_vectorized_scenario,
                n_ally=config.tabs.max_n_ally,
                n_enemy=config.tabs.max_n_enemy,
            )
        )
        vectorized_scenario: VectorizedScenario = v_vectorize_scenario(deployed_scenario)

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
        train_info.update(
            {
                "episode_returns": last_state.returned_episode_returns[:, 0].mean(),
                "episode_lengths": last_state.returned_episode_lengths[:, 0].mean(),
                "episode_wins": last_state.returned_episode_wins[:, 0].mean(),
            }
        )

        return (unit_deploy_train_state, unit_comb_train_state, bs_train_state), train_info

    for step in tqdm(range(config.total_iter)):
        (unit_deploy_train_state, unit_comb_train_state, bs_train_state), train_info = jax.lax.scan(
            train_bs,
            (unit_deploy_train_state, unit_comb_train_state, bs_train_state),
            None,
            config.iter_per_bs_train_step,
        )
        mappo_bs.save_state(bs_train_state, os.path.join(config.save_path, f"bs/mappo/{step}"))

        np_result = jax.tree.map(lambda x: np.array(x).tolist(), train_info)
        with open(os.path.join(logs_dir, f"bs_result_{step}.json"), "w") as f:
            json.dump(np_result, f)

        for i in range(config.iter_per_bs_train_step):
            log_data = {}
            for key_, value in train_info.items():
                log_data[f"bs/{key_}"] = jax.tree.map(lambda x: x[i], value)
            wandb.log(log_data)
