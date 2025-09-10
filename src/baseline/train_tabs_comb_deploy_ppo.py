from dataclasses import dataclass
from tqdm import tqdm
import json
import tyro
import wandb
import hashlib


@dataclass
class Config:
    seed: int = 42
    n_env: int = 256  # the number of environments to run in parallel

    battle_simulator_rollout_step: int = 512

    gamma: float = 0.99  # the discount factor
    lamda: float = 0.95  # the lambda for GAE
    clip_value: float = 1.0  # the clip value for PPO
    clip_ratio: float = 0.2  # the clip ratio for PPO
    entropy_coef: float = 0.01  # the entropy coefficient
    ppo_epochs: int = 5  # the number of epochs to update the policy and value function
    max_grad_norm: float = 0.5  # the maximum gradient norm
    lr: float = 3e-4  # the learning rate
    ppo_layer_dim = 128

    # tabs configuration
    scenario: str = "2F1K2A1H"
    max_n_ally: int = 10
    max_n_enemy: int = 6
    max_field_height: int = 4
    max_field_width: int = 5k

    save_path: str = "/save"
    gpu_id: int = 0

    train_step: int = 10000
    log_step: int = 100


if __name__ == "__main__":
    config = tyro.cli(Config)

    import os
    from functools import partial

    os.environ["CUDA_VISIBLE_DEVICES"] = str(config.gpu_id)

    import jax
    import jax.numpy as jnp

    from src.baseline.algorithm import PPO
    from src.baseline.utils import get_gae
    from src.tabs.wrappers import (
        TABSBattleSimulatorHeuristicWrapper,
        TABSBattleSimulatorLogWrapper,
    )
    from src.tabs import TABSUnitComb, TABSUnitDeploy, TABSBattleSimulator
    from src.tabs.scenarios import generate_scenario, TABSConf
    from src.tabs.units import get_all_unit_names
    from src.tabs.scenarios import pprint_grid_with_units

    # Create a hash of the config for unique folder naming
    config_dict = {k: v for k, v in vars(config).items() if not k.startswith("_")}
    config_str = json.dumps(config_dict, sort_keys=True)
    config_hash = hashlib.md5(config_str.encode()).hexdigest()[:8]
    config.save_path = f"/save/{config.scenario}_{config_hash}"

    wandb.init(project="comb_deploy_ppo", config=config)

    config.max_num_units = len(get_all_unit_names())

    tabs_conf = TABSConf(
        scenario_name=config.scenario,
        max_n_ally=config.max_n_ally,
        max_n_enemy=config.max_n_enemy,
        max_field_height=config.max_field_height,
        max_field_width=config.max_field_width,
        max_num_units=config.max_num_units,
    )
    scenario = generate_scenario(tabs_conf)
    repeated_scenario = jax.tree.map(lambda x: jnp.repeat(x[None], config.n_env, axis=0), scenario)

    env_unit_deploy = TABSUnitDeploy(tabs_conf)
    env_unit_comb = TABSUnitComb(tabs_conf)

    config.rollout_step = config.max_n_ally
    ppo_unit_deploy = PPO(config, env_unit_deploy)
    ppo_unit_comb = PPO(config, env_unit_comb)

    unit_deploy_train_state = ppo_unit_deploy.init_train_state(jax.random.key(config.seed))
    unit_comb_train_state = ppo_unit_comb.init_train_state(jax.random.key(config.seed))

    battle_simulator = TABSBattleSimulator(tabs_conf)
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

    all_unit_names = get_all_unit_names()

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
        deploy_rewards = jnp.zeros_like(deploy_rollout_result["values"])
        deploy_rewards = (
            deploy_rollout_result["dones"] * bs_rollout_result["episode_returns"][None, :, 0]
        )
        deploy_rollout_result["rewards"] = deploy_rewards
        comb_rewards = jnp.zeros_like(comb_rollout_result["values"])
        comb_rewards = (
            comb_rollout_result["dones"] * bs_rollout_result["episode_returns"][None, :, 0]
        )
        comb_rollout_result["rewards"] = comb_rewards

        # Calculate advantages and returns
        advantages, returns = jax.vmap(get_gae, in_axes=(1, 1, 1, 0, None, None), out_axes=1)(
            comb_rollout_result["rewards"],
            comb_rollout_result["dones"],
            comb_rollout_result["values"],
            comb_rollout_result["last_value"],
            config.gamma,
            config.lamda,
        )
        comb_rollout_result["advantages"] = (
            advantages - advantages.mean(axis=1, keepdims=True)
        ) / (advantages.std(axis=1, keepdims=True) + 1e-8)  # is it need to consider nan?
        comb_rollout_result["returns"] = returns

        advantages, returns = jax.vmap(get_gae, in_axes=(1, 1, 1, 0, None, None), out_axes=1)(
            deploy_rollout_result["rewards"],
            deploy_rollout_result["dones"],
            deploy_rollout_result["values"],
            deploy_rollout_result["last_value"],
            config.gamma,
            config.lamda,
        )
        deploy_rollout_result["advantages"] = (
            advantages - advantages.mean(axis=1, keepdims=True)
        ) / (advantages.std(axis=1, keepdims=True) + 1e-8)
        # deploy_rollout_result['advantages'] = advantages
        deploy_rollout_result["returns"] = returns

        unit_deploy_train_state, deploy_train_info = ppo_unit_deploy.train(
            unit_deploy_train_state, deploy_rollout_result
        )

        unit_comb_train_state, comb_train_info = ppo_unit_comb.train(
            unit_comb_train_state, comb_rollout_result
        )

        # Ror logging
        result = {
            "comb_train_info": comb_train_info,
            "deploy_train_info": deploy_train_info,
            "bs_rollout_result": bs_rollout_result,
        }

        result["bs_rollout_result"]["episode_returns"] = result["bs_rollout_result"][
            "episode_returns"
        ][:, 0].mean()
        result["bs_rollout_result"]["episode_lengths"] = result["bs_rollout_result"][
            "episode_lengths"
        ][:, 0].mean()
        result["bs_rollout_result"]["episode_wins"] = result["bs_rollout_result"]["episode_wins"][
            :, 0
        ].mean()
        result["bs_rollout_result"]["episode_returns"] += result["bs_rollout_result"][
            "episode_wins"
        ]

        result["comb_evaluation"] = {
            "budget": comb_rollout_result["last_state"].budget.mean(),
        }

        for i, name in enumerate(all_unit_names):
            result["comb_evaluation"][f"unit_count/{name}"] = comb_rollout_result[
                "last_state"
            ].current_unit_list.mean(axis=0)[i]

        result["unit_deploy"] = deploy_rollout_result["last_state"].battle_field[0]

        return (unit_deploy_train_state, unit_comb_train_state, new_key), result

    @jax.jit
    def train_fn(unit_deploy_train_state, unit_comb_train_state, key):
        carry = (unit_deploy_train_state, unit_comb_train_state, key)
        (unit_deploy_train_state, unit_comb_train_state, key), result = jax.lax.scan(
            train_body, carry, None, config.log_step
        )

        return (unit_deploy_train_state, unit_comb_train_state, key), result

    key = jax.random.key(config.seed)
    for step in tqdm(range(config.train_step // config.log_step)):
        (unit_deploy_train_state, unit_comb_train_state, key), result = train_fn(
            unit_deploy_train_state, unit_comb_train_state, key
        )

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

        ppo_unit_deploy.save_state(unit_deploy_train_state, config.save_path + f"/deploy/{step}")
        ppo_unit_comb.save_state(unit_comb_train_state, config.save_path + f"/comb/{step}")
