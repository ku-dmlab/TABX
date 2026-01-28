"""
Based on JaxMARL Implementation of MAPPO and Craftax Implementation of RND
"""

import hashlib
import json
import os
from dataclasses import dataclass
from typing import Literal, Tuple

import chex
import jax
import jax.numpy as jnp
import numpy as np
import optax
import tyro
from flax import struct
from flax.training.train_state import TrainState

import wandb
from src.baseline.layers import ActorRNN, RNDCriticRNN, RNDNetwork, ScannedRNN
from src.baseline.utils import (
    batchify,
    dataclass_to_dict,
    get_battle_metric,
    save_params,
    unbatchify,
)
from src.tabx import TABX, build_batched_env_params_and_config
from src.tabx.wrappers.wrappers import (
    TABXAutoResetWrapper,
    TABXEnemyHeuristicWrapper,
    TABXLogWrapper,
)


@dataclass
class Config:
    LR: float = 0.004
    NUM_ENVS: int = 128
    NUM_STEPS: int = 128
    GRU_HIDDEN_DIM: int = 128
    FC_DIM_SIZE: int = 128
    TOTAL_TIMESTEPS: int = 1e7
    UPDATE_EPOCHS: int = 4
    NUM_MINIBATCHES: int = 4
    GAMMA: float = 0.99
    GAE_LAMBDA: float = 0.95
    CLIP_EPS: float = 0.05
    SCALE_CLIP_EPS: bool = False
    ENT_COEF: float = 0.01
    VF_COEF: float = 0.5
    MAX_GRAD_NORM: float = 0.25
    ACTIVATION: str = "relu"
    ANNEAL_LR: bool = True
    LN_EPS: float = 1e-6
    # RND
    RND_HIDDEN_DIM: int = 128
    RND_OUTPUT_DIM: int = 256
    RND_NUM_LAYERS: int = 3
    RND_LR: float = 3e-4
    RND_REWARD_COEF: float = 1.0
    RND_LOSS_COEF: float = 0.01
    RND_GAE_COEF: float = 0.01
    RND_IS_EPISODIC: bool = False
    EXPLORATION_UPDATE_EPOCHS: int = 1
    # Env
    SCENARIO: str = "elbow"
    PHYSICS: str = "default"
    HEURISTIC: str = "easy"
    WORLD_STATE_TYPE: Literal["concat", "global"] = "global"
    # Misc.
    SEED: int | Tuple[int, ...] = 0
    ALGORITHM: str = "mappo_rnd"  # for distinguishing wandb runs
    PROJECT_NAME: str = "mappo_rnn"  # wandb project name
    SAVE_PATH: str = "./ckpt"
    SAVE_VIDEO: bool = False
    VALUE_EVAL_NUM_ENVS: int | None = 128
    POSITION_PERMUTATION: bool = False


@struct.dataclass
class RNDTransition:
    global_done: chex.Array
    done: chex.Array
    action: chex.Array
    log_prob: chex.Array
    obs: chex.Array
    world_state: chex.Array
    info: chex.Array
    avail_actions: chex.Array
    value_e: chex.Array
    reward_e: chex.Array
    value_i: chex.Array
    reward_i: chex.Array
    next_world_state: chex.Array


@struct.dataclass
class EvaluateState:
    dones: jnp.ndarray
    global_dones: jnp.ndarray
    world_state: jnp.ndarray
    value: jnp.ndarray
    cr_hstates: jnp.ndarray
    ac_hstates: jnp.ndarray


def make_train(config):
    env_params, tabx_config = build_batched_env_params_and_config(
        scenario_names=config["SCENARIO"],
        physics_param_names=config["PHYSICS"],
        heuristic_param_names=config["HEURISTIC"],
        n_repeat=config["NUM_ENVS"],
    )
    env = TABX(
        cfg=tabx_config,
        world_state_type=config["WORLD_STATE_TYPE"],
        position_permutation=config["POSITION_PERMUTATION"],
    )
    env = TABXLogWrapper(env)
    env = TABXEnemyHeuristicWrapper(env)
    env = TABXAutoResetWrapper(env)

    eval_env = TABX(
        cfg=tabx_config,
        world_state_type=config["WORLD_STATE_TYPE"],
        position_permutation=config["POSITION_PERMUTATION"],
    )
    eval_env = TABXLogWrapper(eval_env, reset_when_done=False)
    eval_env = TABXEnemyHeuristicWrapper(eval_env)

    config["NUM_ACTORS"] = env.num_agents * config["NUM_ENVS"]
    config["NUM_UPDATES"] = config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    config["MINIBATCH_SIZE"] = (
        config["NUM_ACTORS"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"]
    )
    config["CLIP_EPS"] = (
        config["CLIP_EPS"] / env.num_agents if config["SCALE_CLIP_EPS"] else config["CLIP_EPS"]
    )

    def linear_schedule(count):
        frac = (
            1.0
            - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"]))
            / config["NUM_UPDATES"]
        )
        return config["LR"] * frac

    def train(rng):
        init_rng = rng
        # INIT NETWORK
        actor_network = ActorRNN(env.action_space(env.agents[0]).n, config=config)
        critic_network = RNDCriticRNN(config=config)
        rng, _rng_actor, _rng_critic = jax.random.split(rng, 3)
        ac_init_x = (
            jnp.zeros((1, config["NUM_ENVS"], env.observation_space(env.agents[0]).shape[0])),
            jnp.zeros((1, config["NUM_ENVS"])),
            jnp.zeros((1, config["NUM_ENVS"], env.action_space(env.agents[0]).n)),
        )
        ac_init_hstate = ScannedRNN.initialize_carry(config["NUM_ENVS"], config["GRU_HIDDEN_DIM"])
        actor_network_params = actor_network.init(_rng_actor, ac_init_hstate, ac_init_x)
        cr_init_x = (
            jnp.zeros((1, config["NUM_ENVS"], env.world_state_size())),
            jnp.zeros((1, config["NUM_ENVS"])),
        )
        cr_init_hstate = ScannedRNN.initialize_carry(config["NUM_ENVS"], config["GRU_HIDDEN_DIM"])
        critic_network_params = critic_network.init(_rng_critic, cr_init_hstate, cr_init_x)

        if config["ANNEAL_LR"]:
            actor_tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(learning_rate=linear_schedule, eps=1e-5),
            )
            critic_tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(learning_rate=linear_schedule, eps=1e-5),
            )
        else:
            actor_tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(config["LR"], eps=1e-5),
            )
            critic_tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(config["LR"], eps=1e-5),
            )
        actor_train_state = TrainState.create(
            apply_fn=actor_network.apply,
            params=actor_network_params,
            tx=actor_tx,
        )
        critic_train_state = TrainState.create(
            apply_fn=critic_network.apply,
            params=critic_network_params,
            tx=critic_tx,
        )

        # Random Network
        rnd_random_network = RNDNetwork(
            hidden_dim=config["RND_HIDDEN_DIM"],
            output_dim=config["RND_OUTPUT_DIM"],
            num_layers=config["RND_NUM_LAYERS"],
        )
        rng, _rng = jax.random.split(rng)
        rnd_random_network_params = rnd_random_network.init(
            _rng, jnp.zeros((1, env.world_state_size()))
        )

        # Distillation Network
        rnd_distillation_network = RNDNetwork(
            hidden_dim=config["RND_HIDDEN_DIM"],
            output_dim=config["RND_OUTPUT_DIM"],
            num_layers=config["RND_NUM_LAYERS"],
        )
        rng, _rng = jax.random.split(rng)
        rnd_distillation_network_params = rnd_distillation_network.init(
            _rng, jnp.zeros((1, env.world_state_size()))
        )
        rnd_tx = optax.chain(
            optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
            optax.adam(config["RND_LR"], eps=1e-5),
        )

        # Exploration state
        rnd_state = TrainState.create(
            apply_fn=rnd_distillation_network.apply,
            params=rnd_distillation_network_params,
            tx=rnd_tx,
        )

        # INIT ENV
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config["NUM_ENVS"])

        obsv, env_state = jax.vmap(env.reset, in_axes=(0, 0))(reset_rng, env_params)
        ac_init_hstate = ScannedRNN.initialize_carry(config["NUM_ACTORS"], config["GRU_HIDDEN_DIM"])
        cr_init_hstate = ScannedRNN.initialize_carry(config["NUM_ACTORS"], config["GRU_HIDDEN_DIM"])

        # TRAIN LOOP
        def _update_step(update_runner_state, unused):
            # COLLECT TRAJECTORIES
            runner_state, update_steps = update_runner_state

            def _env_step(runner_state, unused):
                train_states, env_state, last_obs, last_done, rnd_state, hstates, rng = runner_state

                # SELECT ACTION
                rng, _rng = jax.random.split(rng)
                avail_actions = jax.vmap(env.get_avail_actions)(env_state)
                avail_actions = jax.lax.stop_gradient(
                    batchify(avail_actions, env.agents, config["NUM_ACTORS"])
                )
                obs_batch = batchify(last_obs, env.agents, config["NUM_ACTORS"])
                ac_in = (obs_batch[np.newaxis, :], last_done[np.newaxis, :], avail_actions)
                # print('env step ac in', ac_in)
                ac_hstate, pi = actor_network.apply(train_states[0].params, hstates[0], ac_in)
                action = pi.sample(seed=_rng)
                log_prob = pi.log_prob(action)
                env_act = unbatchify(action, env.agents, config["NUM_ENVS"], env.num_agents)
                env_act = {k: v.squeeze() for k, v in env_act.items()}

                # VALUE
                # world_state is (num_envs, world_state_size)
                # repeat for each agent to get (num_actors, world_state_size)
                world_state = last_obs["world_state"]
                world_state = jnp.tile(world_state, (env.num_agents, 1))

                cr_in = (world_state[None, :], last_done[np.newaxis, :])
                cr_hstate, value_e, value_i = critic_network.apply(
                    train_states[1].params, hstates[1], cr_in
                )

                # STEP ENV
                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, config["NUM_ENVS"])
                obsv, next_env_state, reward, done, info = jax.vmap(env.step, in_axes=(0, 0, 0, 0))(
                    rng_step, env_state, env_act, env_params
                )

                reward_e = batchify(reward, env.agents, config["NUM_ACTORS"]).squeeze()
                global_done = jnp.tile(done["__all__"], env.num_agents)

                next_world_state = jnp.tile(obsv["world_state"], (env.num_agents, 1))
                random_pred = rnd_random_network.apply(rnd_random_network_params, next_world_state)
                distill_pred = rnd_state.apply_fn(rnd_state.params, next_world_state)
                error = (random_pred - distill_pred) * (1 - global_done[:, None])

                reward_i = config["RND_REWARD_COEF"] * jnp.square(error).mean(axis=-1)

                done_batch = batchify(done, env.agents, config["NUM_ACTORS"]).squeeze()
                transition = RNDTransition(
                    global_done=global_done,
                    done=last_done,
                    action=action.squeeze(),
                    log_prob=log_prob.squeeze(),
                    obs=obs_batch,
                    world_state=world_state,
                    info=info,
                    avail_actions=avail_actions,
                    value_e=value_e.squeeze(),
                    reward_e=reward_e,
                    value_i=value_i.squeeze(),
                    reward_i=reward_i,
                    next_world_state=next_world_state,
                )
                runner_state = (
                    train_states,
                    next_env_state,
                    obsv,
                    done_batch,
                    rnd_state,
                    (ac_hstate, cr_hstate),
                    rng,
                )
                return runner_state, transition

            initial_hstates = runner_state[-2]
            runner_state, traj_batch = jax.lax.scan(
                _env_step, runner_state, None, config["NUM_STEPS"]
            )

            # CALCULATE ADVANTAGE
            train_states, env_state, last_obs, last_done, rnd_state, hstates, rng = runner_state

            last_world_state = last_obs["world_state"]
            last_world_state = jnp.repeat(last_world_state, env.num_agents, axis=0)

            cr_in = (
                last_world_state[None, :],
                last_done[np.newaxis, :],
            )
            _, last_val_e, last_val_i = critic_network.apply(
                train_states[1].params, hstates[1], cr_in
            )
            last_val_e = last_val_e.squeeze()
            last_val_i = last_val_i.squeeze()

            def _calculate_gae(traj_batch, last_val, is_extrinsic):
                def _get_advantages(gae_and_next_value, transition):
                    gae, next_value, is_extrinsic = gae_and_next_value
                    done, value, reward = (
                        transition.global_done,
                        jax.lax.select(is_extrinsic, transition.value_e, transition.value_i),
                        jax.lax.select(is_extrinsic, transition.reward_e, transition.reward_i),
                    )

                    done = jnp.logical_and(
                        done, jnp.logical_or(config["RND_IS_EPISODIC"], is_extrinsic)
                    )

                    delta = reward + config["GAMMA"] * next_value * (1 - done) - value
                    gae = delta + config["GAMMA"] * config["GAE_LAMBDA"] * (1 - done) * gae
                    return (gae, value, is_extrinsic), gae

                _, advantages = jax.lax.scan(
                    _get_advantages,
                    (jnp.zeros_like(last_val), last_val, is_extrinsic),
                    traj_batch,
                    reverse=True,
                    unroll=16,
                )
                return advantages, advantages + jax.lax.select(
                    is_extrinsic, traj_batch.value_e, traj_batch.value_i
                )

            advantages_e, targets_e = _calculate_gae(traj_batch, last_val_e, True)
            advantages_i, targets_i = _calculate_gae(traj_batch, last_val_i, False)

            # UPDATE NETWORK
            def _update_epoch(update_state, unused):
                def _update_minbatch(train_states, batch_info):
                    actor_train_state, critic_train_state = train_states
                    (
                        ac_init_hstate,
                        cr_init_hstate,
                        traj_batch,
                        advantages_e,
                        targets_e,
                        advantages_i,
                        targets_i,
                    ) = batch_info

                    def _actor_loss_fn(actor_params, init_hstate, traj_batch, gae_e, gae_i):
                        # RERUN NETWORK
                        _, pi = actor_network.apply(
                            actor_params,
                            init_hstate.squeeze(),
                            (traj_batch.obs, traj_batch.done, traj_batch.avail_actions),
                        )
                        log_prob = pi.log_prob(traj_batch.action)

                        # CALCULATE ACTOR LOSS
                        logratio = log_prob - traj_batch.log_prob
                        ratio = jnp.exp(logratio)
                        gae = gae_e + config["RND_GAE_COEF"] * gae_i
                        gae = (gae - gae.mean()) / (gae.std() + 1e-8)
                        loss_actor1 = ratio * gae
                        loss_actor2 = (
                            jnp.clip(
                                ratio,
                                1.0 - config["CLIP_EPS"],
                                1.0 + config["CLIP_EPS"],
                            )
                            * gae
                        )
                        loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
                        loss_actor = loss_actor.mean()
                        entropy = pi.entropy().mean()

                        # debug
                        approx_kl = ((ratio - 1) - logratio).mean()
                        clip_frac = jnp.mean(jnp.abs(ratio - 1) > config["CLIP_EPS"])

                        actor_loss = loss_actor - config["ENT_COEF"] * entropy

                        return actor_loss, (loss_actor, entropy, ratio, approx_kl, clip_frac)

                    def _critic_loss_fn(
                        critic_params, init_hstate, traj_batch, targets_e, targets_i
                    ):
                        # Rerun network
                        _, value_e, value_i = critic_network.apply(
                            critic_params,
                            init_hstate.squeeze(),
                            (traj_batch.world_state, traj_batch.done),
                        )

                        # Calculate extrinsic value loss
                        value_pred_clipped_e = traj_batch.value_e + (
                            value_e - traj_batch.value_e
                        ).clip(-config["CLIP_EPS"], config["CLIP_EPS"])
                        value_losses_e = jnp.square(value_e - targets_e)
                        value_losses_clipped_e = jnp.square(value_pred_clipped_e - targets_e)
                        value_loss_e = (
                            0.5 * jnp.maximum(value_losses_e, value_losses_clipped_e).mean()
                        )

                        # Calculate instrinsic value loss
                        value_pred_clipped_i = traj_batch.value_i + (
                            value_i - traj_batch.value_i
                        ).clip(-config["CLIP_EPS"], config["CLIP_EPS"])
                        value_losses_i = jnp.square(value_i - targets_i)
                        value_losses_clipped_i = jnp.square(value_pred_clipped_i - targets_i)
                        value_loss_i = (
                            0.5 * jnp.maximum(value_losses_i, value_losses_clipped_i).mean()
                        )

                        value_loss = value_loss_e + value_loss_i
                        critic_loss = config["VF_COEF"] * value_loss

                        return critic_loss, (value_loss_e, value_loss_i)

                    actor_grad_fn = jax.value_and_grad(_actor_loss_fn, has_aux=True)
                    actor_loss, actor_grads = actor_grad_fn(
                        actor_train_state.params,
                        ac_init_hstate,
                        traj_batch,
                        advantages_e,
                        advantages_i,
                    )
                    critic_grad_fn = jax.value_and_grad(_critic_loss_fn, has_aux=True)
                    critic_loss, critic_grads = critic_grad_fn(
                        critic_train_state.params,
                        cr_init_hstate,
                        traj_batch,
                        targets_e,
                        targets_i,
                    )

                    actor_train_state = actor_train_state.apply_gradients(grads=actor_grads)
                    critic_train_state = critic_train_state.apply_gradients(grads=critic_grads)

                    total_loss = actor_loss[0] + critic_loss[0]
                    loss_info = {
                        "total_loss": total_loss,
                        "actor_loss": actor_loss[0],
                        "value_loss": critic_loss[0],
                        "entropy": actor_loss[1][1],
                        "ratio": actor_loss[1][2],
                        "approx_kl": actor_loss[1][3],
                        "clip_frac": actor_loss[1][4],
                    }

                    return (actor_train_state, critic_train_state), loss_info

                (
                    train_states,
                    init_hstates,
                    traj_batch,
                    advantages_e,
                    targets_e,
                    advantages_i,
                    targets_i,
                    rng,
                ) = update_state
                rng, _rng = jax.random.split(rng)

                init_hstates = jax.tree.map(
                    lambda x: jnp.reshape(x, (1, config["NUM_ACTORS"], -1)), init_hstates
                )

                batch = (
                    init_hstates[0],
                    init_hstates[1],
                    traj_batch,
                    advantages_e.squeeze(),
                    targets_e.squeeze(),
                    advantages_i.squeeze(),
                    targets_i.squeeze(),
                )
                permutation = jax.random.permutation(_rng, config["NUM_ACTORS"])

                shuffled_batch = jax.tree.map(lambda x: jnp.take(x, permutation, axis=1), batch)

                minibatches = jax.tree.map(
                    lambda x: jnp.swapaxes(
                        jnp.reshape(
                            x,
                            [x.shape[0], config["NUM_MINIBATCHES"], -1] + list(x.shape[2:]),
                        ),
                        1,
                        0,
                    ),
                    shuffled_batch,
                )

                # train_states = (actor_train_state, critic_train_state)
                train_states, loss_info = jax.lax.scan(_update_minbatch, train_states, minibatches)
                update_state = (
                    train_states,
                    jax.tree.map(lambda x: x.squeeze(), init_hstates),
                    traj_batch,
                    advantages_e,
                    targets_e,
                    advantages_i,
                    targets_i,
                    rng,
                )
                return update_state, loss_info

            update_state = (
                train_states,
                initial_hstates,
                traj_batch,
                advantages_e,
                targets_e,
                advantages_i,
                targets_i,
                rng,
            )
            update_state, loss_info = jax.lax.scan(
                _update_epoch, update_state, None, config["UPDATE_EPOCHS"]
            )
            loss_info["ratio_0"] = loss_info["ratio"].at[0, 0].get()
            loss_info = jax.tree.map(lambda x: x.mean(), loss_info)

            train_states = update_state[0]
            metric = loss_info | get_battle_metric(env, env_state)

            rng = update_state[-1]

            # Update exploration state
            def _update_exp_epoch(rnd_update_state, unused):
                def _update_exp_minibatch(rnd_state, traj_batch):
                    def _rnd_loss_fn(rnd_distillation_params, traj_batch):
                        random_network_out = rnd_random_network.apply(
                            rnd_random_network_params, traj_batch.next_world_state
                        )

                        distillation_network_out = rnd_state.apply_fn(
                            rnd_distillation_params, traj_batch.next_world_state
                        )

                        error = (random_network_out - distillation_network_out) * (
                            1 - traj_batch.global_done[..., None]
                        )

                        return config["RND_LOSS_COEF"] * jnp.square(error).mean()

                    rnd_grad_fn = jax.value_and_grad(_rnd_loss_fn, has_aux=False)
                    rnd_loss, rnd_grad = rnd_grad_fn(rnd_state.params, traj_batch)
                    rnd_state = rnd_state.apply_gradients(grads=rnd_grad)

                    losses = (rnd_loss,)
                    return rnd_state, losses

                (rnd_state, traj_batch, rng) = rnd_update_state
                rng, _rng = jax.random.split(rng)

                permutation = jax.random.permutation(_rng, config["NUM_ACTORS"])
                shuffled_batch = jax.tree.map(
                    lambda x: jnp.take(x, permutation, axis=1), traj_batch
                )

                minibatches = jax.tree.map(
                    lambda x: jnp.swapaxes(
                        jnp.reshape(
                            x,
                            [x.shape[0], config["NUM_MINIBATCHES"], -1] + list(x.shape[2:]),
                        ),
                        1,
                        0,
                    ),
                    shuffled_batch,
                )
                rnd_state, losses = jax.lax.scan(_update_exp_minibatch, rnd_state, minibatches)
                update_state = (rnd_state, traj_batch, rng)
                return update_state, losses

            rnd_update_state = (rnd_state, traj_batch, rng)
            rnd_update_state, rnd_loss = jax.lax.scan(
                _update_exp_epoch, rnd_update_state, None, config["EXPLORATION_UPDATE_EPOCHS"]
            )

            metric["rnd_loss"] = rnd_loss[0].mean()
            metric["reward_i"] = traj_batch.reward_i.mean()

            rnd_state = rnd_update_state[0]
            rng = rnd_update_state[-1]

            def callback(metric, init_rng, update_steps):
                seed = jax.random.key_data(init_rng)[-1].item()
                metric_plus_seed = {
                    f"{k}_seed{seed}" if isinstance(config["SEED"], tuple) else k: v
                    for k, v in metric.items()
                }
                metric_plus_seed[
                    f"update_steps_seed{seed}"
                    if isinstance(config["SEED"], tuple)
                    else "update_steps"
                ] = update_steps
                metric_plus_seed[
                    f"env_step_seed{seed}" if isinstance(config["SEED"], tuple) else "env_step"
                ] = update_steps * config["NUM_ENVS"] * config["NUM_STEPS"]
                wandb.log(metric_plus_seed)

            update_steps = update_steps + 1
            jax.experimental.io_callback(callback, None, metric, init_rng, update_steps)
            runner_state = (train_states, env_state, last_obs, last_done, rnd_state, hstates, rng)
            return (runner_state, update_steps), metric

        rng, _rng = jax.random.split(rng)
        runner_state = (
            (actor_train_state, critic_train_state),
            env_state,
            obsv,
            jnp.zeros((config["NUM_ACTORS"]), dtype=bool),
            rnd_state,
            (ac_init_hstate, cr_init_hstate),
            _rng,
        )
        runner_state, metric = jax.lax.scan(
            _update_step, (runner_state, 0), None, config["NUM_UPDATES"]
        )
        return {"runner_state": runner_state, "metric": metric}

    return train


if __name__ == "__main__":
    config = tyro.cli(Config)
    config_dict = dataclass_to_dict(config)
    config_json = json.dumps(config_dict, sort_keys=True)
    config_hash = hashlib.md5(config_json.encode()).hexdigest()[:8]
    save_path = os.path.join(config.SAVE_PATH, config.PROJECT_NAME, config_hash)
    os.makedirs(save_path, exist_ok=True)

    # Save config to logs directory
    with open(os.path.join(save_path, "config.json"), "w") as f:
        json.dump(config_dict, f, indent=2)

    wandb.init(
        project=config.PROJECT_NAME, mode="online", config=config_dict | {"HASH": config_hash}
    )
    train_fn = make_train(config.__dict__)
    with jax.disable_jit(False):
        if isinstance(config.SEED, int):
            result = train_fn(jax.random.key(config.SEED))
        else:
            result = jax.vmap(train_fn)(jax.vmap(jax.random.key)(jnp.array(config.SEED)))

    # Save trained model
    runner_state = result["runner_state"][0]
    save_params(
        runner_state[0][0].params,
        os.path.join(save_path, f"{config.SCENARIO}_seed{config.SEED}_actor.safetensors"),
    )
    if isinstance(config.SEED, tuple):
        runner_state = jax.tree.map(lambda x: x[0], runner_state)

    if config.SAVE_VIDEO:
        # Visualize
        from src.tabx.visualize import Visualizer

        env_params, tabx_config = build_batched_env_params_and_config(
            scenario_names=config.SCENARIO,
            physics_param_names=config.PHYSICS,
            heuristic_param_names=config.HEURISTIC,
            n_repeat=1,
        )
        env = TABX(cfg=tabx_config)
        env = TABXEnemyHeuristicWrapper(env)
        num_steps = env.max_episode_steps

        ac_init_hstate = ScannedRNN.initialize_carry(1, config.GRU_HIDDEN_DIM)
        actor_network = ActorRNN(env.action_space(env.agents[0]).n, config=config.__dict__)

        rng = jax.random.PRNGKey(config.SEED[0] if isinstance(config.SEED, tuple) else config.SEED)
        rng, _rng = jax.random.split(rng)
        obs, env_state = env.reset(_rng, env_params)

        def rollout_body(carry, _):
            (obs, env_state, done, ac_hstate, rng) = carry
            # Random policy
            rng, step_rng, action_rng = jax.random.split(rng, 3)
            avail_actions = env.get_avail_actions(env_state)
            ac_in = (
                jnp.expand_dims(batchify(obs, env.agents, env.num_agents), 1),
                done,
                jnp.expand_dims(batchify(avail_actions, env.agents, env.num_agents), 1),
            )

            ac_hstate, pi = actor_network.apply(runner_state[0][0].params, ac_hstate, ac_in)
            action = pi.sample(seed=action_rng)
            env_act = unbatchify(action, env.agents, 1, env.num_agents)
            env_act = {k: v.squeeze() for k, v in env_act.items()}
            obs, next_state, reward, done, info = env.step(step_rng, env_state, env_act)

            return (
                obs,
                next_state,
                batchify(done, env.agents, env.num_agents),
                ac_hstate,
                rng,
            ), next_state

        _, stacked = jax.lax.scan(
            rollout_body,
            (obs, env_state, jnp.zeros((env.num_agents, 1), dtype=jnp.bool), ac_init_hstate, rng),
            None,
            num_steps,
        )
        state_seq = [jax.tree.map(lambda x: x[i], stacked) for i in range(num_steps)]

        visualizer = Visualizer(env, state_seq)
        gif_path = os.path.join(
            save_path, f"{config.SCENARIO}_{config.HEURISTIC}_{config.SEED}.gif"
        )
        visualizer.animate(save_fname=gif_path, view=False)
        wandb.log({"visualization": wandb.Video(gif_path, fps=4, format="gif")})
