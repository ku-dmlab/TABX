"""
Based on JaxMARL Implementation of IQL
"""

import os
from dataclasses import dataclass
from functools import partial
from typing import Literal, Tuple

import flashbax as fbx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import tyro

import wandb
from src.baseline.layers import QScannedRNN, RNNQNetwork
from src.baseline.utils import CustomTrainState, Timestep, get_battle_metric, save_params
from src.tabs import TABS, build_batched_env_params_and_config
from src.tabs.wrappers.wrappers import (
    TABSAutoResetWrapper,
    TABSEnemyHeuristicWrapper,
    TABSLogWrapper,
)


@dataclass
class Config:
    TOTAL_TIMESTEPS: int = 1e7
    NUM_ENVS: int = 16
    NUM_STEPS: int = 512
    BUFFER_SIZE: int = 5000
    BUFFER_BATCH_SIZE: int = 32
    HIDDEN_SIZE: int = 512
    EPS_START: float = 1.0
    EPS_FINISH: float = 0.05
    EPS_DECAY: float = 0.1  # percentage of updates
    MAX_GRAD_NORM: int = 10
    TARGET_UPDATE_INTERVAL: int = 10
    TAU: float = 1.0
    NUM_EPOCHS: int = 8
    LR: float = 0.00005
    LEARNING_STARTS: int = 10000  # timesteps
    LR_LINEAR_DECAY: bool = False
    GAMMA: float = 0.99
    REW_SCALE: float = 10.0  # scale the reward to the original scale of SMAC
    TEST_DURING_TRAINING: bool = True
    TEST_INTERVAL: float | None = (
        None  # as a fraction of updates, i.e. log every 5% of training process
    )
    TEST_NUM_STEPS: int = 512
    TEST_NUM_ENVS: int = 128  # number of episodes to average over, can affect performance
    # Env
    SCENARIO: str = "elbow"
    PHYSICS: str = "default"
    HEURISTIC: str = "easy"
    WORLD_STATE_TYPE: Literal["concat", "global"] = "concat"
    # Misc.
    SEED: int | Tuple[int, ...] = 0
    PROJECT_NAME: str = "iql_rnn"  # wandb project name
    SAVE_PATH: str = "./ckpt"
    SAVE_VIDEO: bool = False
    VALUE_EVAL_NUM_ENVS: int | None = 128


def get_greedy_actions(q_vals, valid_actions):
    unavail_actions = 1 - valid_actions
    q_vals = q_vals - (unavail_actions * 1e10)
    return jnp.argmax(q_vals, axis=-1)


def make_train(config, env, eval_env, env_params, test_env_params):
    config["NUM_UPDATES"] = config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]

    eps_scheduler = optax.linear_schedule(
        init_value=config["EPS_START"],
        end_value=config["EPS_FINISH"],
        transition_steps=config["EPS_DECAY"] * config["NUM_UPDATES"],
    )

    # epsilon-greedy exploration
    def eps_greedy_exploration(rng, q_vals, eps, valid_actions):
        rng_a, rng_e = jax.random.split(
            rng
        )  # a key for sampling random actions and one for picking

        greedy_actions = get_greedy_actions(q_vals, valid_actions)

        # pick random actions from the valid actions
        def get_random_actions(rng, val_action):
            return jax.random.choice(
                rng,
                jnp.arange(val_action.shape[-1]),
                p=val_action * 1.0 / jnp.sum(val_action, axis=-1),
            )

        _rngs = jax.random.split(rng_a, valid_actions.shape[0])
        random_actions = jax.vmap(get_random_actions)(_rngs, valid_actions)

        chosed_actions = jnp.where(
            jax.random.uniform(rng_e, greedy_actions.shape)
            < eps,  # pick the actions that should be random
            random_actions,
            greedy_actions,
        )
        return chosed_actions

    def batchify(x: dict):
        return jnp.stack([x[agent] for agent in env.agents], axis=0)

    def unbatchify(x: jnp.ndarray):
        return {agent: x[i] for i, agent in enumerate(env.agents)}

    def train(rng):
        # INIT ENV
        init_rng = rng
        rng, _rng = jax.random.split(rng)

        # INIT NETWORK AND OPTIMIZER
        network = RNNQNetwork(
            action_dim=env.action_space(env.agents[0]).n,
            hidden_dim=config["HIDDEN_SIZE"],
        )

        def create_agent(rng):
            init_x = (
                jnp.zeros(
                    (1, 1, env.observation_space(env.agents[0]).shape[0])
                ),  # (time_step, batch_size, obs_size)
                jnp.zeros((1, 1)),  # (time_step, batch size)
            )
            init_hs = QScannedRNN.initialize_carry(
                config["HIDDEN_SIZE"], 1
            )  # (batch_size, hidden_dim)
            network_params = network.init(rng, init_hs, *init_x)

            lr_scheduler = optax.linear_schedule(
                init_value=config["LR"],
                end_value=1e-10,
                transition_steps=(config["NUM_EPOCHS"]) * config["NUM_UPDATES"],
            )

            lr = lr_scheduler if config.get("LR_LINEAR_DECAY", False) else config["LR"]

            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.radam(learning_rate=lr),
            )

            train_state = CustomTrainState.create(
                apply_fn=network.apply,
                params=network_params,
                target_network_params=network_params,
                tx=tx,
            )
            return train_state

        rng, _rng = jax.random.split(rng)
        train_state = create_agent(rng)

        # INIT BUFFER
        # to initalize the buffer is necessary to sample a trajectory to know its strucutre
        def _env_sample_step(env_state, unused):
            rng, key_a, key_s = jax.random.split(jax.random.PRNGKey(0), 3)  # use a dummy rng here
            key_a = jax.random.split(key_a, env.num_agents)
            actions = {
                agent: jax.vmap(env.action_spaces[agent].sample)(
                    jax.random.split(key_a[i], config["NUM_ENVS"])
                )
                for i, agent in enumerate(env.agents)
            }
            avail_actions = jax.vmap(env.get_avail_actions)(env_state)
            obs, env_state, rewards, dones, infos = jax.vmap(env.step, in_axes=(0, 0, 0, 0))(
                jax.random.split(key_s, config["NUM_ENVS"]), env_state, actions, env_params
            )
            timestep = Timestep(
                obs=obs,
                actions=actions,
                rewards=rewards,
                dones=dones,
                avail_actions=avail_actions,
            )
            return env_state, timestep

        _, _env_state = jax.vmap(env.reset, in_axes=(0, 0))(
            jax.random.split(rng, config["NUM_ENVS"]), env_params
        )
        _, sample_traj = jax.lax.scan(_env_sample_step, _env_state, None, config["NUM_STEPS"])
        sample_traj_unbatched = jax.tree.map(
            lambda x: x[:, 0], sample_traj
        )  # remove the NUM_ENV dim
        buffer = fbx.make_trajectory_buffer(
            max_length_time_axis=config["BUFFER_SIZE"] // config["NUM_ENVS"],
            min_length_time_axis=config["BUFFER_BATCH_SIZE"],
            sample_batch_size=config["BUFFER_BATCH_SIZE"],
            add_batch_size=config["NUM_ENVS"],
            sample_sequence_length=1,
            period=1,
        )
        buffer_state = buffer.init(sample_traj_unbatched)

        # TRAINING LOOP
        def _update_step(runner_state, unused):
            train_state, buffer_state, test_state, rng = runner_state

            # SAMPLE PHASE
            def _step_env(carry, _):
                hs, last_obs, last_dones, env_state, rng = carry
                rng, rng_a, rng_s = jax.random.split(rng, 3)

                # (num_agents, 1 (dummy time), num_envs, obs_size)
                _obs = batchify(last_obs)[:, np.newaxis]
                _dones = batchify(last_dones)[:, np.newaxis]

                new_hs, q_vals = jax.vmap(
                    network.apply, in_axes=(None, 0, 0, 0)
                )(  # vmap across the agent dim
                    train_state.params,
                    hs,
                    _obs,
                    _dones,
                )
                q_vals = q_vals.squeeze(
                    axis=1
                )  # (num_agents, num_envs, num_actions) remove the time dim

                # explore
                avail_actions = jax.vmap(env.get_avail_actions)(env_state)

                eps = eps_scheduler(train_state.n_updates)
                _rngs = jax.random.split(rng_a, env.num_agents)
                actions = jax.vmap(eps_greedy_exploration, in_axes=(0, 0, None, 0))(
                    _rngs, q_vals, eps, batchify(avail_actions)
                )
                actions = unbatchify(actions)

                new_obs, new_env_state, rewards, dones, infos = jax.vmap(
                    env.step, in_axes=(0, 0, 0, 0)
                )(jax.random.split(rng_s, config["NUM_ENVS"]), env_state, actions, env_params)
                timestep = Timestep(
                    obs=last_obs,
                    actions=actions,
                    rewards=jax.tree.map(lambda x: config.get("REW_SCALE", 1) * x, rewards),
                    dones=last_dones,
                    avail_actions=avail_actions,
                )
                return (new_hs, new_obs, dones, new_env_state, rng), (timestep, infos)

            # step the env (should be a complete rollout)
            rng, _rng = jax.random.split(rng)
            init_obs, env_state = jax.vmap(env.reset, in_axes=(0, 0))(
                jax.random.split(_rng, config["NUM_ENVS"]), env_params
            )
            init_dones = {
                agent: jnp.zeros((config["NUM_ENVS"]), dtype=bool)
                for agent in env.agents + ["__all__"]
            }
            init_hs = QScannedRNN.initialize_carry(
                config["HIDDEN_SIZE"], len(env.agents), config["NUM_ENVS"]
            )
            expl_state = (init_hs, init_obs, init_dones, env_state)
            rng, _rng = jax.random.split(rng)
            (_, _, _, last_state, _), (timesteps, infos) = jax.lax.scan(
                _step_env,
                (*expl_state, _rng),
                None,
                config["NUM_STEPS"],
            )

            train_state = train_state.replace(
                timesteps=train_state.timesteps + config["NUM_STEPS"] * config["NUM_ENVS"]
            )  # update timesteps count

            # BUFFER UPDATE
            buffer_traj_batch = jax.tree.map(
                lambda x: jnp.swapaxes(x, 0, 1)[
                    :, np.newaxis
                ],  # put the batch dim first and add a dummy sequence dim
                timesteps,
            )  # (num_envs, 1, time_steps, ...)
            buffer_state = buffer.add(buffer_state, buffer_traj_batch)

            # NETWORKS UPDATE
            def _learn_phase(carry, _):
                train_state, rng = carry
                rng, _rng = jax.random.split(rng)
                minibatch = buffer.sample(buffer_state, _rng).experience
                minibatch = jax.tree.map(
                    lambda x: jnp.swapaxes(
                        x[:, 0], 0, 1
                    ),  # remove the dummy sequence dim (1) and swap batch and temporal dims
                    minibatch,
                )  # (max_time_steps, batch_size, ...)

                # preprocess network input
                init_hs = QScannedRNN.initialize_carry(
                    config["HIDDEN_SIZE"],
                    len(env.agents),
                    config["BUFFER_BATCH_SIZE"],
                )
                # num_agents, timesteps, batch_size, ...
                _obs = batchify(minibatch.obs)
                _dones = batchify(minibatch.dones)
                _actions = batchify(minibatch.actions)
                _rewards = batchify(minibatch.rewards)
                _avail_actions = batchify(minibatch.avail_actions)

                _, q_next_target = jax.vmap(network.apply, in_axes=(None, 0, 0, 0))(
                    train_state.target_network_params,
                    init_hs,
                    _obs,
                    _dones,
                )  # (num_agents, timesteps, batch_size, num_actions)

                def _loss_fn(params):
                    _, q_vals = jax.vmap(network.apply, in_axes=(None, 0, 0, 0))(
                        params,
                        init_hs,
                        _obs,
                        _dones,
                    )  # (num_agents, timesteps, batch_size, num_actions)

                    # get logits of the chosen actions
                    chosen_action_q_vals = jnp.take_along_axis(
                        q_vals,
                        _actions[..., np.newaxis],
                        axis=-1,
                    ).squeeze(-1)  # (num_agents, timesteps, batch_size,)

                    unavailable_actions = 1 - _avail_actions
                    valid_q_vals = q_vals - (unavailable_actions * 1e10)

                    # get the q values of the next state
                    q_next = jnp.take_along_axis(
                        q_next_target,
                        jnp.argmax(valid_q_vals, axis=-1)[..., np.newaxis],
                        axis=-1,
                    ).squeeze(-1)  # (num_agents, timesteps, batch_size,)

                    target = (
                        _rewards[:, :-1] + (1 - _dones[:, :-1]) * config["GAMMA"] * q_next[:, 1:]
                    )

                    chosen_action_q_vals = chosen_action_q_vals[:, :-1]
                    loss = jnp.mean((chosen_action_q_vals - jax.lax.stop_gradient(target)) ** 2)

                    return loss, chosen_action_q_vals.mean()

                (loss, qvals), grads = jax.value_and_grad(_loss_fn, has_aux=True)(
                    train_state.params
                )
                train_state = train_state.apply_gradients(grads=grads)
                train_state = train_state.replace(
                    grad_steps=train_state.grad_steps + 1,
                )
                return (train_state, rng), (loss, qvals)

            rng, _rng = jax.random.split(rng)
            is_learn_time = (buffer.can_sample(buffer_state)) & (  # enough experience in buffer
                train_state.timesteps > config["LEARNING_STARTS"]
            )
            (train_state, rng), (loss, qvals) = jax.lax.cond(
                is_learn_time,
                lambda train_state, rng: jax.lax.scan(
                    _learn_phase, (train_state, rng), None, config["NUM_EPOCHS"]
                ),
                lambda train_state, rng: (
                    (train_state, rng),
                    (
                        jnp.zeros(config["NUM_EPOCHS"]),
                        jnp.zeros(config["NUM_EPOCHS"]),
                    ),
                ),  # do nothing
                train_state,
                _rng,
            )

            # update target network
            train_state = jax.lax.cond(
                train_state.n_updates % config["TARGET_UPDATE_INTERVAL"] == 0,
                lambda train_state: train_state.replace(
                    target_network_params=optax.incremental_update(
                        train_state.params,
                        train_state.target_network_params,
                        config["TAU"],
                    )
                ),
                lambda train_state: train_state,
                operand=train_state,
            )

            # UPDATE METRICS
            train_state = train_state.replace(n_updates=train_state.n_updates + 1)
            metrics = {
                "env_step": train_state.timesteps,
                "update_steps": train_state.n_updates,
                "grad_steps": train_state.grad_steps,
                "env_steps": train_state.n_updates * config["NUM_ENVS"] * config["NUM_STEPS"],
                "loss": loss.mean(),
                "qvals": qvals.mean(),
            }
            metrics.update(get_battle_metric(env, last_state))

            # update the test metrics
            if config.get("TEST_DURING_TRAINING", True):
                rng, _rng = jax.random.split(rng)
                if config["TEST_INTERVAL"] is not None:
                    test_state = jax.lax.cond(
                        train_state.n_updates % int(config["NUM_UPDATES"] * config["TEST_INTERVAL"])
                        == 0,
                        lambda _: get_greedy_metrics(_rng, train_state),
                        lambda _: test_state,
                        operand=None,
                    )
                else:
                    test_state = get_greedy_metrics(_rng, train_state)
                metrics.update({"test_" + k: v for k, v in test_state.items()})

            def callback(metric, init_rng):
                seed = jax.random.key_data(init_rng)[-1].item()
                metric_plus_seed = {
                    f"{k}_seed{seed}" if isinstance(config["SEED"], tuple) else k: v
                    for k, v in metric.items()
                }
                wandb.log(metric_plus_seed)

            jax.experimental.io_callback(callback, None, metrics, init_rng)

            runner_state = (train_state, buffer_state, test_state, rng)

            return runner_state, None

        def get_greedy_metrics(rng, train_state):
            """Help function to test greedy policy during training"""
            if not config.get("TEST_DURING_TRAINING", True):
                return None

            params = train_state.params

            def _greedy_env_step(step_state, unused):
                params, env_state, last_obs, last_dones, hstate, rng = step_state
                rng, key_s = jax.random.split(rng)
                _obs = batchify(last_obs)[:, np.newaxis]
                _dones = batchify(last_dones)[:, np.newaxis]
                next_hstate, q_vals = jax.vmap(network.apply, in_axes=(None, 0, 0, 0))(
                    params,
                    hstate,
                    _obs,
                    _dones,
                )
                q_vals = q_vals.squeeze(axis=1)
                valid_actions = jax.vmap(env.get_avail_actions)(env_state)
                actions = get_greedy_actions(q_vals, batchify(valid_actions))
                actions = unbatchify(actions)
                obs, next_env_state, rewards, dones, infos = jax.vmap(
                    env.step, in_axes=(0, 0, 0, 0)
                )(
                    jax.random.split(key_s, config["TEST_NUM_ENVS"]),
                    env_state,
                    actions,
                    test_env_params,
                )
                timestep = Timestep(
                    obs=last_obs,
                    actions=actions,
                    rewards=jax.tree.map(lambda x: config.get("REW_SCALE", 1) * x, rewards),
                    dones=last_dones,
                    avail_actions=valid_actions,
                )
                step_state = (params, next_env_state, obs, dones, next_hstate, rng)
                return step_state, (timestep, env_state, q_vals, hstate)

            rng, _rng = jax.random.split(rng)
            init_obs, env_state = jax.vmap(env.reset, in_axes=(0, 0))(
                jax.random.split(_rng, config["TEST_NUM_ENVS"]), test_env_params
            )
            init_dones = {
                agent: jnp.zeros((config["TEST_NUM_ENVS"]), dtype=bool)
                for agent in env.agents + ["__all__"]
            }
            rng, _rng = jax.random.split(rng)
            hstate = QScannedRNN.initialize_carry(
                config["HIDDEN_SIZE"], len(env.agents), config["TEST_NUM_ENVS"]
            )  # (n_agents*n_envs, hs_size)
            step_state = (
                params,
                env_state,
                init_obs,
                init_dones,
                hstate,
                _rng,
            )
            step_state, (timestep, stacked_env_state, stacked_q_vals, stacked_hstate) = (
                jax.lax.scan(_greedy_env_step, step_state, None, config["TEST_NUM_STEPS"])
            )

            def timestep_sample(array, idx, axis=1):
                return jax.vmap(lambda idx, num: jnp.take(array[idx], num, axis=axis), in_axes=0)(
                    idx, jnp.arange(config["TEST_NUM_ENVS"])
                )

            rng, _rng = jax.random.split(rng, 2)
            timestep_idx = jax.random.randint(
                _rng, (config["TEST_NUM_ENVS"],), 0, config["TEST_NUM_STEPS"]
            )
            value_eval_env_params = jax.tree.map(
                partial(timestep_sample, idx=timestep_idx, axis=0), stacked_env_state
            )

            eval_hstate = timestep_sample(stacked_hstate, timestep_idx, axis=1).swapaxes(0, 1)
            estimated_q = timestep_sample(stacked_q_vals.max(axis=-1), timestep_idx, axis=1)
            eval_obsv = jax.vmap(env.get_obs)(value_eval_env_params["state"])
            eval_obsv = env.filter_obs(eval_obsv)

            def _eval_step(step_state, unused):
                params, env_state, last_obs, last_dones, hstate, rng, all_done, returns = step_state
                rng, key_s = jax.random.split(rng)
                _obs = batchify(last_obs)[:, np.newaxis]
                _dones = batchify(last_dones)[:, np.newaxis]
                hstate, q_vals = jax.vmap(network.apply, in_axes=(None, 0, 0, 0))(
                    params,
                    hstate,
                    _obs,
                    _dones,
                )
                q_vals = q_vals.squeeze(axis=1)
                valid_actions = jax.vmap(eval_env.get_avail_actions)(env_state)
                actions = get_greedy_actions(q_vals, batchify(valid_actions))
                actions = unbatchify(actions)
                obs, env_state, rewards, dones, infos = jax.vmap(eval_env.step, in_axes=(0, 0, 0))(
                    jax.random.split(key_s, config["TEST_NUM_ENVS"]),
                    env_state,
                    actions,
                )
                step_state = (
                    params,
                    env_state,
                    obs,
                    dones,
                    hstate,
                    rng,
                    dones["__all__"],
                    jnp.where(all_done, returns, returns * config["GAMMA"] + rewards["__all__"]),
                )
                return step_state, None

            def mc_value_estimate(rng):
                eval_runner_state = (
                    train_state.params,
                    value_eval_env_params,
                    eval_obsv,
                    init_dones,
                    eval_hstate,
                    rng,
                    jnp.zeros((config["TEST_NUM_ENVS"]), dtype=bool),
                    jnp.zeros((config["TEST_NUM_ENVS"]), dtype=float),
                )
                eval_runner_state, _ = jax.lax.scan(
                    _eval_step, eval_runner_state, None, env.max_episode_steps
                )
                return eval_runner_state[-1]

            mc_returns = jax.vmap(mc_value_estimate)(
                jax.random.split(rng, config["VALUE_EVAL_NUM_ENVS"])
            )
            mc_returns = mc_returns.mean(axis=0)[..., None]
            dones = jax.vmap(batchify)(timestep.dones)
            dones = timestep_sample(dones, timestep_idx, axis=1)
            all_dones = timestep_sample(timestep.dones["__all__"], timestep_idx, axis=0)
            error_target = ~dones | all_dones[:, None]
            estimated_value = jnp.sum(error_target * estimated_q) / jnp.sum(error_target)
            value_error = (
                error_target * jnp.abs(estimated_q - mc_returns.mean(axis=0)[:, None])
            ).sum() / error_target.sum()

            metrics = get_battle_metric(env, step_state[1])
            metrics["value_error"] = value_error
            metrics["mc_returns"] = mc_returns.mean()
            metrics["estimated_q"] = estimated_value.mean()
            return metrics

        rng, _rng = jax.random.split(rng)
        test_state = get_greedy_metrics(_rng, train_state)

        # train
        rng, _rng = jax.random.split(rng)
        runner_state = (train_state, buffer_state, test_state, _rng)

        runner_state, metrics = jax.lax.scan(
            _update_step, runner_state, None, config["NUM_UPDATES"]
        )

        return {"runner_state": runner_state, "metrics": metrics}

    return train


def main(config):
    wandb.init(project=config.PROJECT_NAME, mode="online", config=config)

    train_env_params, tabs_config = build_batched_env_params_and_config(
        scenario_names=config.SCENARIO,
        physics_param_names=config.PHYSICS,
        heuristic_param_names=config.HEURISTIC,
        n_repeat=config.NUM_ENVS,
    )
    test_env_params, tabs_config = build_batched_env_params_and_config(
        scenario_names=config.SCENARIO,
        physics_param_names=config.PHYSICS,
        heuristic_param_names=config.HEURISTIC,
        n_repeat=config.TEST_NUM_ENVS,
    )
    env = TABS(cfg=tabs_config, world_state_type=config.WORLD_STATE_TYPE)
    env = TABSLogWrapper(env)
    env = TABSEnemyHeuristicWrapper(env)
    env = TABSAutoResetWrapper(env)

    eval_env = TABS(cfg=tabs_config, world_state_type=config.WORLD_STATE_TYPE)
    eval_env = TABSLogWrapper(eval_env, reset_when_done=False)
    eval_env = TABSEnemyHeuristicWrapper(eval_env)

    train_fn = jax.jit(
        make_train(config.__dict__, env, eval_env, train_env_params, test_env_params)
    )
    with jax.disable_jit(False):
        if isinstance(config.SEED, int):
            result = train_fn(jax.random.key(config.SEED))
        else:
            result = jax.vmap(train_fn)(jax.vmap(jax.random.key)(jnp.array(config.SEED)))

    # Save trained model
    save_path = os.path.join(config.SAVE_PATH, config.PROJECT_NAME)
    os.makedirs(save_path, exist_ok=True)
    runner_state = result["runner_state"]
    save_params(
        runner_state[0].params,
        os.path.join(
            save_path,
            f"{config.SCENARIO}_seed{config.SEED}_qf.safetensors",
        ),
    )

    if isinstance(config.SEED, tuple):
        runner_state = jax.tree.map(lambda x: x[0], runner_state)

    if config.SAVE_VIDEO:
        # Visualize
        from src.tabs.visualize import Visualizer

        vis_num_envs = 1
        env_params, tabs_config = build_batched_env_params_and_config(
            scenario_names=config.SCENARIO,
            physics_param_names=config.PHYSICS,
            heuristic_param_names=config.HEURISTIC,
            n_repeat=vis_num_envs,
            squeeze_when_single_scenario=False,
        )
        env = TABS(cfg=tabs_config)
        env = TABSEnemyHeuristicWrapper(env)
        num_steps = env.max_episode_steps

        def batchify(x: dict):
            return jnp.stack([x[agent] for agent in env.agents], axis=0)

        def unbatchify(x: jnp.ndarray):
            return {agent: x[i] for i, agent in enumerate(env.agents)}

        network = RNNQNetwork(
            action_dim=env.action_space(env.agents[0]).n,
            hidden_dim=config.HIDDEN_SIZE,
        )

        rng = jax.random.PRNGKey(config.SEED[0] if isinstance(config.SEED, tuple) else config.SEED)

        rng, _rng = jax.random.split(rng)
        init_obs, env_state = jax.vmap(env.reset, in_axes=(0, 0))(
            jax.random.split(_rng, vis_num_envs), env_params
        )
        init_dones = {
            agent: jnp.zeros((vis_num_envs), dtype=bool) for agent in env.agents + ["__all__"]
        }
        hstate = QScannedRNN.initialize_carry(config.HIDDEN_SIZE, len(env.agents), vis_num_envs)

        def rollout_body(carry, _):
            params, env_state, last_obs, last_dones, hstate, rng = carry
            rng, rng_step = jax.random.split(rng)
            hstate, q_vals = jax.vmap(network.apply, in_axes=(None, 0, 0, 0))(
                params,
                hstate,
                batchify(last_obs)[:, np.newaxis],
                batchify(last_dones)[:, np.newaxis],
            )
            q_vals = q_vals.squeeze(axis=1)
            valid_actions = jax.vmap(env.get_avail_actions)(env_state)
            actions = get_greedy_actions(q_vals, batchify(valid_actions))
            actions = unbatchify(actions)
            obs, env_state, rewards, dones, infos = jax.vmap(env.step, in_axes=(0, 0, 0))(
                jax.random.split(rng_step, vis_num_envs), env_state, actions
            )
            return (params, env_state, obs, dones, hstate, rng), env_state

        rng, _rng = jax.random.split(rng)
        _, stacked = jax.lax.scan(
            rollout_body,
            (runner_state[0].params, env_state, init_obs, init_dones, hstate, _rng),
            None,
            num_steps,
        )
        state_seq = [jax.tree.map(lambda x: x[i].squeeze(), stacked) for i in range(num_steps)]

        visualizer = Visualizer(env, state_seq)
        gif_path = os.path.join(
            save_path, f"{config.SCENARIO}_{config.HEURISTIC}_{config.SEED}.gif"
        )
        visualizer.animate(save_fname=gif_path, view=False)
        wandb.log({"visualization": wandb.Video(gif_path, fps=4, format="gif")})

    return result


if __name__ == "__main__":
    main(tyro.cli(Config))
