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
from src.tabs.constants import ALL_UNIT_NAMES


@dataclass
class Config:
    seed: int = 42
    n_env: int = 32  # the number of environments to run in parallel
    tabs: TABSConfig = TABSConfig()
    comb_ppo: PPOConfig = PPOConfig(
        rollout_step=tabs.max_n_ally, n_env=n_env, entropy_coef=0.1, batch_size=32
    )
    deploy_ppo: PPOConfig = PPOConfig(
        rollout_step=tabs.max_n_ally, n_env=n_env, entropy_coef=0.1, batch_size=32
    )
    mappo: PPOConfig = PPOConfig(rollout_step=512, n_env=n_env, batch_size=n_env)
    save_path: str = "/save"
    gpu_id: int = 0
    iter_per_comb_step: int = 100
    iter_per_deploy_step: int = 100
    iter_per_bs_step: int = 200
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
    from src.baseline.utils import get_abs_path
    from functools import partial
    from src.tabs.scenarios import get_vectorized_scenario, VectorizedScenario

    # Create a hash of the config for unique folder naming
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    config_hash = hashlib.md5(str(config).encode()).hexdigest()[:8]
    config.save_path = f"/save/{config.tabs.scenario_name}_{current_time}_{config_hash}"

    wandb.init(
        project="tabs_ppo_mappo", config=config, mode="online" if not config.debug else "disabled"
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
    ppo_unit_comb = PPO(config.comb_ppo, env_unit_comb)
    ppo_unit_deploy = PPO(config.deploy_ppo, env_unit_deploy)
    mappo_bs = MAPPO(config.mappo, env_bs)

    key = jax.random.key(config.seed)
    key_comb, key_deploy, key_bs = jax.random.split(key, 3)
    train_state_comb = ppo_unit_comb.init_train_state(
        key_comb, config.iter_per_comb_step * config.total_train_iter
    )
    train_state_deploy = ppo_unit_deploy.init_train_state(
        key_deploy, config.iter_per_deploy_step * config.total_train_iter
    )
    train_state_bs = mappo_bs.init_train_state(
        key_bs, config.iter_per_bs_step * config.total_train_iter
    )

    # Rollout and get samples
    def rollout_tabs(train_state_comb, train_state_deploy, train_state_bs, scenarios_comb):
        # Combine allies - TABSUnitComb
        train_state_comb, rollout_result_comb = ppo_unit_comb.rollout(
            train_state_comb, scenarios_comb
        )
        # Update the scenario with the combination results for TABSUnitDeploy
        scenarios_deploy = scenarios_comb.replace(
            ally_unit_comp=rollout_result_comb["last_state"].current_unit_list
        )

        # Deploy allies - TABSUnitDeploy
        train_state_deploy, rollout_result_deploy = ppo_unit_deploy.rollout(
            train_state_deploy, scenarios_deploy
        )
        # Update the scenario with the deployment results for TABSBattltSimulate
        scenarios_bs = scenarios_deploy.replace(
            battle_field=rollout_result_deploy["last_state"].battle_field
        )

        # Battle - TABSBattleSimulate
        train_state_bs, rollout_result_bs = mappo_bs.rollout(train_state_bs, scenarios_bs)

        # Set rewards for comb and deploy agents based on the battle simulation result
        rollout_result_comb["rewards"] = (
            rollout_result_comb["dones"] * rollout_result_bs["returned_episode_returns"][None, :, 0]
        )
        rollout_result_comb["unit_list"] = rollout_result_comb["last_state"].current_unit_list
        rollout_result_deploy["rewards"] = (
            rollout_result_deploy["dones"]
            * rollout_result_bs["returned_episode_returns"][None, :, 0]
        )
        rollout_result_bs["rewards"] = rollout_result_bs["common_reward"].sum() / config.n_env

        rollout_result = {"bs_rollout_result": rollout_result_bs}
        rollout_result["bs_rollout_result"]["returned_episode_returns"] = rollout_result[
            "bs_rollout_result"
        ]["returned_episode_returns"][:, 0].mean()
        rollout_result["bs_rollout_result"]["returned_episode_lengths"] = rollout_result[
            "bs_rollout_result"
        ]["returned_episode_lengths"][:, 0].mean()
        rollout_result["bs_rollout_result"]["returned_episode_wins"] = rollout_result[
            "bs_rollout_result"
        ]["returned_episode_wins"][:, 0].mean()
        rollout_result["bs_rollout_result"]["returned_episode_returns"] += rollout_result[
            "bs_rollout_result"
        ]["returned_episode_wins"]

        rollout_result["comb_evaluation"] = {
            "budget": rollout_result_comb["last_state"].budget.mean(),
        }

        for i, name in enumerate(ALL_UNIT_NAMES):
            rollout_result["comb_evaluation"][f"unit_count/{name}"] = rollout_result_comb[
                "last_state"
            ].current_unit_list.mean(axis=0)[i]

        rollout_result["unit_deploy"] = rollout_result_deploy["last_state"].battle_field[0]

        # attack success rate, damage dealt
        last_state = rollout_result_bs["last_state"]

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
        vectorized_scenario: VectorizedScenario = v_vectorize_scenario(scenarios_bs)

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

        unit_battle_metric["attack_success_rate"] = (
            unit_battle_metric["returned_cumulative_attack_success"]
            / unit_battle_metric["returned_cumulative_is_attackings"]
        )

        unit_specific_metric = {}
        for key, value in unit_battle_metric.items():
            for i, name in enumerate(ALL_UNIT_NAMES):
                unit_specific_metric[f"{key}/{name}"] = value[i]

        team_fight_metric = {
            "cumulative_is_attackings": unit_battle_metric[
                "returned_cumulative_is_attackings"
            ].sum(),
            "cumulative_damage_dealts": jnp.nansum(
                unit_battle_metric["returned_cumulative_damage_dealts"]
                * (unit_battle_metric["returned_cumulative_damage_dealts"] > 0)
            ),
            "cumulative_heal_amount": jnp.nansum(
                unit_battle_metric["returned_cumulative_damage_dealts"]
                * (unit_battle_metric["returned_cumulative_damage_dealts"] < 0)
            ),
            "attack_success_rate": jnp.nansum(
                unit_battle_metric["returned_cumulative_attack_success"]
                / unit_battle_metric["returned_cumulative_is_attackings"]
            ),
            "first_kill_rate": last_state.returned_first_kills[:, 0].mean(),
        }

        return {
            "train_state_comb": train_state_comb,
            "train_state_deploy": train_state_deploy,
            "train_state_bs": train_state_bs,
            "rollout_result_comb": rollout_result_comb,
            "rollout_result_deploy": rollout_result_deploy,
            "rollout_result_bs": rollout_result_bs,
            "rollout_result": rollout_result,
            "episode_returns": rollout_result_bs["returned_episode_returns"],
            "episode_lengths": rollout_result_bs["returned_episode_lengths"],
            "episode_wins": rollout_result_bs["returned_episode_wins"],
            "unit_battle_metric": unit_specific_metric,
            "team_fight_metric": team_fight_metric,
        }

    def train_comb(carry, _):
        (train_state_comb, train_state_deploy, train_state_bs) = carry
        rollout_result = rollout_tabs(
            train_state_comb, train_state_deploy, train_state_bs, repeated_scenarios
        )

        train_state_comb, train_info_comb = ppo_unit_comb.train(
            train_state_comb, rollout_result["rollout_result_comb"]
        )

        train_info_comb["rollout_result"] = rollout_result["rollout_result"]
        train_info_comb["episode_returns"] = rollout_result["episode_returns"]
        train_info_comb["episode_lengths"] = rollout_result["episode_lengths"]
        train_info_comb["episode_wins"] = rollout_result["episode_wins"]

        return (
            train_state_comb,
            train_state_deploy,
            train_state_bs,
        ), train_info_comb

    def train_deploy(carry, _):
        (train_state_comb, train_state_deploy, train_state_bs) = carry
        rollout_result = rollout_tabs(
            train_state_comb, train_state_deploy, train_state_bs, repeated_scenarios
        )

        train_state_deploy, train_info_deploy = ppo_unit_deploy.train(
            train_state_deploy, rollout_result["rollout_result_deploy"]
        )

        train_info_deploy["rollout_result"] = rollout_result["rollout_result"]
        train_info_deploy["episode_returns"] = rollout_result["episode_returns"]
        train_info_deploy["episode_lengths"] = rollout_result["episode_lengths"]
        train_info_deploy["episode_wins"] = rollout_result["episode_wins"]

        return (
            train_state_comb,
            train_state_deploy,
            train_state_bs,
        ), train_info_deploy

    def train_bs(carry, _):
        (train_state_comb, train_state_deploy, train_state_bs) = carry
        rollout_result = rollout_tabs(
            train_state_comb, train_state_deploy, train_state_bs, repeated_scenarios
        )

        train_state_bs, train_info_bs = mappo_bs.train(
            train_state_bs, rollout_result["rollout_result_bs"]
        )

        train_info_bs["rollout_result"] = rollout_result["rollout_result"]
        train_info_bs["episode_returns"] = rollout_result["episode_returns"]
        train_info_bs["episode_lengths"] = rollout_result["episode_lengths"]
        train_info_bs["episode_wins"] = rollout_result["episode_wins"]
        train_info_bs.update(rollout_result["team_fight_metric"])
        train_info_bs.update(rollout_result["unit_battle_metric"])

        return (
            train_state_comb,
            train_state_deploy,
            train_state_bs,
        ), train_info_bs

    @jax.jit
    def train_fn(carry):
        carry, train_info_comb = jax.lax.scan(train_comb, carry, None, config.iter_per_comb_step)
        carry, train_info_deploy = jax.lax.scan(
            train_deploy, carry, None, config.iter_per_deploy_step
        )
        carry, train_info_bs = jax.lax.scan(train_bs, carry, None, config.iter_per_bs_step)

        result = {}
        result["comb_evaluation"] = train_info_comb["rollout_result"]["comb_evaluation"]
        result["unit_deploy"] = train_info_deploy["rollout_result"]["unit_deploy"]
        result["bs_rollout_result"] = {
            "returned_episode_returns": train_info_bs["rollout_result"]["bs_rollout_result"][
                "returned_episode_returns"
            ],
            "returned_episode_lengths": train_info_bs["rollout_result"]["bs_rollout_result"][
                "returned_episode_lengths"
            ],
            "returned_episode_wins": train_info_bs["rollout_result"]["bs_rollout_result"][
                "returned_episode_wins"
            ],
        }

        del train_info_comb["rollout_result"]
        del train_info_deploy["rollout_result"]
        del train_info_bs["rollout_result"]

        result.update(
            {
                "train_info_comb": train_info_comb,
                "train_info_deploy": train_info_deploy,
                "train_info_bs": train_info_bs,
            }
        )

        return carry, result

    # Define custom step metrics for each training phase
    wandb.define_metric("comb_step")
    wandb.define_metric("deploy_step")
    wandb.define_metric("bs_step")

    # Define all comb metrics to use comb_step as x-axis
    wandb.define_metric("comb/*", step_metric="comb_step")

    # Define all deploy metrics to use deploy_step as x-axis
    wandb.define_metric("deploy/*", step_metric="deploy_step")

    # Define all bs metrics to use bs_step as x-axis
    wandb.define_metric("bs/*", step_metric="bs_step")

    # Create logs directory if it doesn't exist
    logs_dir = get_abs_path(config.save_path + "/logs")
    os.makedirs(logs_dir, exist_ok=True)

    # Alternating training for the end-to-end agent
    carry = (train_state_comb, train_state_deploy, train_state_bs)
    for step in tqdm(range(config.total_train_iter)):
        # Train on TABSUnitComb
        carry, result = train_fn(carry)

        # Log comb metrics
        for i in range(config.iter_per_comb_step):
            comb_log_data = {}
            for key_, value in result["comb_evaluation"].items():
                comb_log_data[f"comb/{key_}"] = jax.tree.map(lambda x: x[i], value)
            for key_, value in result["train_info_comb"].items():
                comb_log_data[f"comb/train_info/{key_}"] = jax.tree.map(lambda x: x[i], value)
            # Custom step for comb training
            comb_step = step * (config.iter_per_comb_step) + i
            comb_log_data["comb_step"] = comb_step
            wandb.log(comb_log_data)

        # Log deploy metrics
        for i in range(config.iter_per_deploy_step):
            deploy_log_data = {}
            if "unit_deploy" in result:
                deploy_log_data["deploy/unit_deploy"] = wandb.Html(
                    f"<pre>{pprint_grid_with_units(result['unit_deploy'][i])}</pre>"
                )
            for key_, value in result["train_info_deploy"].items():
                deploy_log_data[f"deploy/train_info/{key_}"] = jax.tree.map(lambda x: x[i], value)
            # Custom step for deploy training
            deploy_step = step * (config.iter_per_deploy_step) + i
            deploy_log_data["deploy_step"] = deploy_step
            wandb.log(deploy_log_data)

        # Log bs metrics
        for i in range(config.iter_per_bs_step):
            bs_log_data = {}
            for key_, value in result["bs_rollout_result"].items():
                bs_log_data[f"bs/{key_}"] = jax.tree.map(lambda x: x[i], value)
            for key_, value in result["train_info_bs"].items():
                bs_log_data[f"bs/train_info/{key_}"] = jax.tree.map(lambda x: x[i], value)

            # Custom step for bs training
            bs_step = step * (config.iter_per_bs_step) + i
            bs_log_data["bs_step"] = bs_step
            wandb.log(bs_log_data)

        # Save models
        ppo_unit_comb.save_state(carry[0], os.path.join(config.save_path, f"comb/ppo/{step}"))
        ppo_unit_deploy.save_state(carry[1], os.path.join(config.save_path, f"deploy/ppo/{step}"))
        mappo_bs.save_state(carry[2], os.path.join(config.save_path, f"bs/mappo/{step}"))

        np_result = jax.tree.map(lambda x: np.array(x).tolist(), result)
        with open(os.path.join(logs_dir, f"result_{step}.json"), "w") as f:
            json.dump(np_result, f)
