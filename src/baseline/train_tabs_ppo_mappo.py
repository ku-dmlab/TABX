from dataclasses import dataclass
from tqdm import tqdm
import json
import tyro
import wandb
import hashlib


@dataclass
class Config:
    # Algorithm configuration
    gamma: float = 0.99  # The discount factor
    lamda: float = 0.95  # The lambda for GAE
    clip_value: float = 1.0  # The clip value for PPO
    clip_ratio: float = 0.2  # The clip ratio for PPO
    entropy_coef: float = 0.01  # The entropy coefficient
    ppo_epochs: int = 5  # The number of epochs to update the policy and value function
    mappo_epochs: int = 5
    max_grad_norm: float = 0.5  # The maximum gradient norm
    lr: float = 3e-4  # The learning rate
    ppo_layer_dim = 128  # The layer dimension size
    mappo_layer_dim = 256

    # Env configuration

    # Training configuration
    n_env: int = 32  # The number of environments to run in parallel
    rollout_step_bs: int = 512  # The number of rollouts to run in parallel for TABSBattleSimulator
    train_step: int = 100000
    log_step: int = 10

    seed: int = 42
    save_path: str = "/save"
    gpu_id: int = 0
    debug: bool = False


if __name__ == "__main__":
    config = tyro.cli(Config)

    import os

    os.environ["CUDA_VISIBLE_DEVICES"] = str(config.gpu_id)

    import jax
    import jax.numpy as jnp

    from src.baseline.algorithm import PPO, MAPPO
    from src.tabs.wrappers import (
        TABSBattleSimulatorLogWrapper,
        TABSBattleSimulatorHeuristicWrapper,
        TABSBattleSimulatorAutoResetWrapper,
    )
    from src.tabs.units import get_all_unit_names
    from src.tabs.scenarios import generate_scenario, TABSConfig, pprint_grid_with_units
    from src.tabs import TABSUnitComb, TABSUnitDeploy, TABSBattleSimulator

    # Create a hash of the config for unique folder naming
    config_dict = {k: v for k, v in vars(config).items() if not k.startswith("_")}
    config_str = json.dumps(config_dict, sort_keys=True)
    config_hash = hashlib.md5(config_str.encode()).hexdigest()[:8]
    config.save_path = f"/save/{config.scenario}_{config_hash}"
    config.rollout_step = config.max_n_ally

    wandb.init(
        project="tabs_ppo_mappo", config=config, mode="online" if not config.debug else "disabled"
    )

    config.max_num_units = len(get_all_unit_names())

    tabs_conf = TABSConfig(
        scenario_name=config.scenario,
        max_n_ally=config.max_n_ally,
        max_n_enemy=config.max_n_enemy,
        max_field_height=config.max_field_height,
        max_field_width=config.max_field_width,
        max_num_units=config.max_num_units,
    )
    scenario = generate_scenario(tabs_conf)
    # Duplicate the scenario for parallel runs
    repeated_scenarios = jax.tree.map(lambda x: jnp.repeat(x[None], config.n_env, axis=0), scenario)

    # Envrionments
    env_unit_comb = TABSUnitComb(tabs_conf)
    env_unit_deploy = TABSUnitDeploy(tabs_conf)
    env_bs = TABSBattleSimulator(tabs_conf)
    env_bs = TABSBattleSimulatorHeuristicWrapper(env_bs)
    env_bs = TABSBattleSimulatorAutoResetWrapper(env_bs)
    env_bs = TABSBattleSimulatorLogWrapper(env_bs)

    # Agents
    ppo_unit_comb = PPO(config, env_unit_comb)
    ppo_unit_deploy = PPO(config, env_unit_deploy)
    mappo_bs = MAPPO(config, env_bs)

    key = jax.random.key(config.seed)
    key_comb, key_deploy, key_bs = jax.random.split(key, 3)
    train_state_comb = ppo_unit_comb.init_train_state(key_comb)
    train_state_deploy = ppo_unit_deploy.init_train_state(key_deploy)
    train_state_bs = mappo_bs.init_train_state(key_comb)

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

        for i, name in enumerate(get_all_unit_names()):
            rollout_result["comb_evaluation"][f"unit_count/{name}"] = rollout_result_comb[
                "last_state"
            ].current_unit_list.mean(axis=0)[i]

        rollout_result["unit_deploy"] = rollout_result_deploy["last_state"].battle_field[0]

        return {
            "train_state_comb": train_state_comb,
            "train_state_deploy": train_state_deploy,
            "train_state_bs": train_state_bs,
            "rollout_result_comb": rollout_result_comb,
            "rollout_result_deploy": rollout_result_deploy,
            "rollout_result_bs": rollout_result_bs,
            "rollout_result": rollout_result,
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

        return (
            train_state_comb,
            train_state_deploy,
            train_state_bs,
        ), train_info_bs

    @jax.jit
    def train_fn(carry):
        carry, train_info_comb = jax.lax.scan(train_comb, carry, None, config.log_step)
        carry, train_info_deploy = jax.lax.scan(train_deploy, carry, None, config.log_step)
        carry, train_info_bs = jax.lax.scan(train_bs, carry, None, config.log_step)

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

    # Alternating traininig for the end-to-end agent
    carry = (train_state_comb, train_state_deploy, train_state_bs)
    for step in tqdm(range(config.train_step // (config.log_step * config.n_env))):
        # Train on TABSUnitComb
        carry, result = train_fn(carry)
        for i in range(config.log_step):
            for key_, value in result.items():
                if key_ == "unit_deploy":
                    log_data = {f"{key_}": pprint_grid_with_units(value[i])}
                    wandb.log(
                        {f"{key_}": wandb.Html(f"<pre>{log_data[f'{key_}']}</pre>")},
                        step=step * config.log_step + i,
                    )
                    continue

                log_data = {
                    f"{key_}/{k}": v for k, v in jax.tree.map(lambda x: x[i], value).items()
                }
                wandb.log(log_data, step=step * config.log_step + i)

        # Save models
        ppo_unit_comb.save_state(carry[0], os.path.join(config.save_path, f"comb/ppo/{step}"))
        ppo_unit_deploy.save_state(carry[1], os.path.join(config.save_path, f"deploy/ppo/{step}"))
        mappo_bs.save_state(carry[2], os.path.join(config.save_path, f"bs/mappo/{step}"))
