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
    mappo_epochs: int = 10
    max_grad_norm: float = 0.5  # The maximum gradient norm
    lr: float = 1e-3  # The learning rate
    dqn_layer_dim = 128  # The layer dimension size
    mappo_layer_dim = 256
    eps: float = 0.1
    buffer_size: int = 500
    buffer_batch_size: int = 16

    # Env configuration
    scenario: str = "1K"
    max_n_ally: int = 10
    max_n_enemy: int = 1
    max_field_height: int = 4
    max_field_width: int = 5

    # Training configuration
    n_env: int = 64  # The number of environments to run in parallel
    rollout_step: int = (
        10  # The number of rollouts to run in parallel for TABSUnitComb and TABSUnitDeploy
    )
    rollout_step_bs: int = 1024  # The number of rollouts to run in parallel for TABSBattleSimulator
    train_step: int = 10000
    log_step: int = 100
    target_update_interval: int = 25

    seed: int = 42
    save_path: str = "/save/dqn_mappo"
    gpu_id: int = 0
    debug: bool = False


if __name__ == "__main__":
    config = tyro.cli(Config)

    import os

    os.environ["CUDA_VISIBLE_DEVICES"] = str(config.gpu_id)

    import jax
    import jax.numpy as jnp
    import flashbax as fbx
    from flashbax.buffers.trajectory_buffer import TrajectoryBuffer

    from src.baseline.dqn import DQN, TimeStep
    from src.baseline.mappo import MAPPO
    from src.maenv.tabs.wrappers.wrapper import (
        TABSBattleSimulatorAutoResetWrapper,
        TABSBattleSimulatorLogWrapper,
        TABSBattleSimulatorHeuristicWrapper,
    )
    from src.maenv.tabs.units import get_all_unit_names
    from src.maenv.tabs.scenarios import generate_scenario, TABSConf
    from src.maenv.tabs import TABSUnitComb, TABSUnitDeploy, TABSBattleSimulator

    # Create a hash of the config for unique folder naming
    config_dict = {k: v for k, v in vars(config).items() if not k.startswith("_")}
    config_str = json.dumps(config_dict, sort_keys=True)
    config_hash = hashlib.md5(config_str.encode()).hexdigest()[:8]
    config.save_path = f"/save/{config.scenario}_{config_hash}"

    wandb.init(
        project="tabs_dqn_mappo", config=config, mode="online" if not config.debug else "disabled"
    )

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
    # Duplicate the scenario for parallel runs
    repeated_scenarios = jax.tree.map(lambda x: jnp.repeat(x[None], config.n_env, axis=0), scenario)

    # Envrionments
    env_unit_comb = TABSUnitComb(tabs_conf)
    env_unit_deploy = TABSUnitDeploy(tabs_conf)
    env_bs = TABSBattleSimulator(tabs_conf)
    env_bs = TABSBattleSimulatorHeuristicWrapper(env_bs)
    env_bs = TABSBattleSimulatorAutoResetWrapper(env_bs, scenario)
    env_bs = TABSBattleSimulatorLogWrapper(env_bs)

    # Agents
    dqn_unit_comb = DQN(config, env_unit_comb)
    dqn_unit_deploy = DQN(config, env_unit_deploy)
    mappo_bs = MAPPO(config, env_bs)

    # Replay buffer
    buffer: TrajectoryBuffer = fbx.make_flat_buffer(
        max_length=config.buffer_size,
        min_length=config.buffer_batch_size,
        sample_batch_size=config.buffer_batch_size,
        add_sequences=False,
        add_batch_size=config.n_env,
    )

    # TrainState
    key = jax.random.key(config.seed)
    key_comb, key_deploy, key_bs = jax.random.split(key, 3)
    train_state_comb = dqn_unit_comb.init_train_state(key_comb, scenario)
    train_state_deploy = dqn_unit_deploy.init_train_state(key_deploy, scenario)
    train_state_bs = mappo_bs.init_train_state(key_comb)

    def rollout_tabs(train_state_comb, train_state_deploy, train_state_bs, scenarios_comb):
        # Combine allies - TABSUnitComb
        train_state_comb, rollout_result_comb = dqn_unit_comb.rollout(
            train_state_comb, scenarios_comb
        )
        # Update the scenario with the combination results for TABSUnitDeploy
        scenarios_deploy = scenarios_comb.replace(
            ally_unit_comp=rollout_result_comb["last_state"].current_unit_list
        )

        del rollout_result_comb["last_state"]

        # Deploy allies - TABSUnitDeploy
        train_state_deploy, rollout_result_deploy = dqn_unit_deploy.rollout(
            train_state_deploy, scenarios_deploy
        )
        # Update the scenario with the deployment results for TABSBattltSimulate
        scenarios_bs = scenarios_deploy.replace(
            battle_field=rollout_result_deploy["last_state"].battle_field
        )

        del rollout_result_deploy["last_state"]

        # Battle - TABSBattleSimulate
        train_state_bs, rollout_result_bs = mappo_bs.rollout(train_state_bs, scenarios_bs)

        # Set rewards for comb and deploy agents based on the battle simulation result
        rollout_result_comb["rewards"] = (
            rollout_result_comb["dones"] * rollout_result_bs["returned_episode_returns"][None, :, 0]
        )
        rollout_result_deploy["rewards"] = (
            rollout_result_deploy["dones"]
            * rollout_result_bs["returned_episode_returns"][None, :, 0]
        )
        rollout_result_bs["rewards"] = rollout_result_bs["common_reward"].sum() / config.n_env

        return (
            train_state_comb,
            train_state_deploy,
            train_state_bs,
            rollout_result_comb,
            rollout_result_deploy,
            rollout_result_bs,
        )

    def update_buffers(carry, x):
        train_state = carry

        timestep = TimeStep(
            obs=x["obs"],
            action=x["actions"],
            reward=x["rewards"],
            done=x["dones"],
            next_obs=x["next_obs"],
            unavail_action=x["unavail_actions"],
        )
        buffer_state = buffer.add(train_state.buffer_state, timestep)
        train_state = train_state.replace(buffer_state=buffer_state)

        return (train_state), x

    def train_comb(train_state_comb, train_state_deploy, train_state_bs, scenarios_comb):
        def train_body(carry, step):
            train_state_comb = carry

            # Update network
            train_state_comb, train_info_comb = dqn_unit_comb.train(train_state_comb)

            # Collect samples and update replay buffer
            results = rollout_tabs(
                train_state_comb, train_state_deploy, train_state_bs, scenarios_comb
            )
            (train_state_comb), _ = jax.lax.scan(update_buffers, (results[0]), results[3])

            # Update target network
            train_state_comb = dqn_unit_comb.update_target(train_state_comb, step)

            return (train_state_comb), train_info_comb

        train_state_comb, train_info_comb = jax.lax.scan(
            train_body, (train_state_comb), jnp.arange(config.log_step), config.log_step
        )

        train_states = (train_state_comb, train_state_deploy, train_state_bs)

        return train_states, train_info_comb

    def train_deploy(train_state_comb, train_state_deploy, train_state_bs, scenarios_deploy):
        def train_body(carry, step):
            train_state_deploy = carry

            # Update network
            train_state_deploy, train_info_deploy = dqn_unit_deploy.train(train_state_deploy)

            # Collect samples and update replay buffer
            results = rollout_tabs(
                train_state_comb, train_state_deploy, train_state_bs, scenarios_deploy
            )
            (train_state_deploy), _ = jax.lax.scan(update_buffers, (results[0]), results[3])

            # Update target network
            train_state_deploy = dqn_unit_deploy.update_target(train_state_deploy, step)

            return (train_state_deploy), train_info_deploy

        train_state_deploy, train_info_deploy = jax.lax.scan(
            train_body, (train_state_deploy), jnp.arange(config.log_step), config.log_step
        )

        train_states = (train_state_comb, train_state_deploy, train_state_bs)

        return train_states, train_info_deploy

    def train_bs(carry, _):
        (train_state_comb, train_state_deploy, train_state_bs) = carry
        train_state_comb, train_state_deploy, train_state_bs, _, _, rollout_result_bs = (
            rollout_tabs(train_state_comb, train_state_deploy, train_state_bs, repeated_scenarios)
        )

        train_state_bs, train_info_bs = mappo_bs.train(train_state_bs, rollout_result_bs)

        return (
            train_state_comb,
            train_state_deploy,
            train_state_bs,
        ), train_info_bs

    # Collect samples and update replay buffers
    can_sample = False
    jit_rollout = jax.jit(rollout_tabs)
    while not can_sample:
        results = jit_rollout(
            train_state_comb, train_state_deploy, train_state_bs, repeated_scenarios
        )
        (train_state_comb), _ = jax.lax.scan(update_buffers, (results[0]), results[3])
        (train_state_deploy), _ = jax.lax.scan(update_buffers, (results[1]), results[4])

        can_sample = dqn_unit_comb.buffer.can_sample(
            train_state_comb.buffer_state
        ) & dqn_unit_deploy.buffer.can_sample(train_state_deploy.buffer_state)

    @jax.jit
    def train_fn(train_state_comb, train_state_deploy, train_state_bs):
        # Train on TABSUnitComb
        train_states, train_info_comb = train_comb(
            train_state_comb, train_state_deploy, train_state_bs, repeated_scenarios
        )

        # Train on TABSUnitDeploy
        train_states, train_info_deploy = train_deploy(
            train_states[0], train_states[1], train_states[2], repeated_scenarios
        )

        # Train on TABSBattleSimulator
        train_states, train_info_bs = jax.lax.scan(train_bs, train_states, None, config.log_step)

        return train_states, train_info_comb, train_info_deploy, train_info_bs

    # Alternating traininig for the end-to-end agent
    for step in tqdm(range(config.train_step // config.log_step)):
        train_states, train_info_comb, train_info_deploy, train_info_bs = train_fn(
            train_state_comb, train_state_deploy, train_state_bs
        )
        train_state_comb, train_state_deploy, train_state_bs = train_states

        # Log results
        train_info_bs["episode_returns"] = (
            train_info_bs["returned_episode_returns"][:, :, 0].mean(axis=1).flatten()
        )
        train_info_bs["episode_lengths"] = (
            train_info_bs["returned_episode_lengths"][:, :, 0].mean(axis=1).flatten()
        )
        train_info_bs["episode_wins"] = (
            train_info_bs["returned_episode_wins"][:, :, 0].mean(axis=1).flatten()
        )
        train_info_bs["episode_returns"] += train_info_bs["episode_wins"]

        for i in range(config.log_step):
            log_comb = {}
            data_comb = jax.tree.map(lambda x: x[i], train_info_comb)
            for k, v in data_comb.items():
                if k == "unit_list":
                    # Get unit names for better logging
                    unit_names = get_all_unit_names()

                    # Log individual unit counts as separate metrics
                    for unit_idx, unit_name in enumerate(unit_names):
                        if unit_idx < v.shape[0]:  # Check if unit index exists
                            log_comb[f"train_comb/unit_count/{unit_name}"] = v[unit_idx]

                    # Log unit composition summary (no complex charts to avoid empty tables)
                    for unit_idx, unit_name in enumerate(unit_names):
                        if unit_idx < v.shape[0]:
                            log_comb[f"train_comb/composition/{unit_name}"] = float(v[unit_idx])

                    # Also log as histogram for distribution analysis
                    wandb.log({"comb_train_info/unit_list_histogram": wandb.Histogram(v)})

                    # Log summary statistics
                    total_units = float(jnp.sum(v))
                    if total_units > 0:
                        # Unit diversity (number of different unit types used)
                        unit_diversity = float(jnp.sum(v > 0))
                        log_comb["comb_train_info/unit_diversity"] = unit_diversity
                        log_comb["comb_train_info/total_units"] = total_units

                        # Most used unit type
                        most_used_idx = int(jnp.argmax(v))
                        if most_used_idx < len(unit_names):
                            log_comb["comb_train_info/dominant_unit_idx"] = most_used_idx
                            # Log unit ratios for better analysis
                            for unit_idx, unit_name in enumerate(unit_names):
                                if unit_idx < v.shape[0] and total_units > 0:
                                    ratio = float(v[unit_idx] / total_units)
                                    log_comb[f"comb_train_info/unit_ratio/{unit_name}"] = ratio
                else:
                    log_comb[f"train_comb/{k}"] = v

            log_deploy = {
                f"train_deploy/{k}": v
                for k, v in jax.tree.map(lambda x: x[i], train_info_deploy).items()
            }

            log_bs = {
                f"train_bs/{k}": v for k, v in jax.tree.map(lambda x: x[i], train_info_bs).items()
            }

            wandb.log(log_comb)
            wandb.log(log_deploy)
            wandb.log(log_bs)

        # Save models
        dqn_unit_comb.save_state(
            train_state_comb, os.path.join(config.save_path, f"comb/dqn/{step}")
        )
        dqn_unit_deploy.save_state(
            train_state_deploy, os.path.join(config.save_path, f"deploy/dqn/{step}")
        )
        mappo_bs.save_state(train_state_bs, os.path.join(config.save_path, f"bs/mappo/{step}"))
