"""
Based on JaxMARL Implementation of MAPPO
"""

from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import optax
import tyro
from flax.training.train_state import TrainState

import wandb
from src.baseline.layers import ActorCriticRNN, ScannedRNN
from src.baseline.utils import batchify, get_battle_metric, unbatchify
from src.tabs import TABS
from src.tabs.config import PhysicsParams, TABSHeuristicConfig
from src.tabs.scenarios import generate_scenario_config
from src.tabs.utils import Transition
from src.tabs.wrappers.wrappers import (
    TABSAutoResetWrapper,
    TABSEnemyHeuristicWrapper,
    TABSLogWrapper,
)


class Transition(NamedTuple):
    global_done: jnp.ndarray
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: jnp.ndarray
    avail_actions: jnp.ndarray


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
    # Env
    SCENARIO: str = "2F1K2A1H"
    # Misc.
    SEED: int = 0
    PROJECT_NAME: str = "mappo_rnn"  # wandb project name


def make_train(config):
    vscenario, zone_scenario, tabs_config = generate_scenario_config(
        scenario_name=config["SCENARIO"]
    )
    env = TABS(cfg=tabs_config)
    env = TABSLogWrapper(env)
    env = TABSEnemyHeuristicWrapper(env)
    env = TABSAutoResetWrapper(env)
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
        # INIT NETWORK
        network = ActorCriticRNN(env.action_space(env.agents[0]).n, config=config)
        rng, _rng = jax.random.split(rng)
        init_x = (
            jnp.zeros((1, config["NUM_ENVS"], env.observation_space(env.agents[0]).shape[0])),
            jnp.zeros((1, config["NUM_ENVS"])),
            jnp.zeros((1, config["NUM_ENVS"], env.action_space(env.agents[0]).n)),
        )
        init_hstate = ScannedRNN.initialize_carry(config["NUM_ENVS"], config["GRU_HIDDEN_DIM"])
        network_params = network.init(_rng, init_hstate, init_x)
        if config["ANNEAL_LR"]:
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(learning_rate=linear_schedule, eps=1e-5),
            )
        else:
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(config["LR"], eps=1e-5),
            )
        train_state = TrainState.create(
            apply_fn=network.apply,
            params=network_params,
            tx=tx,
        )

        # INIT ENV
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config["NUM_ENVS"])

        env_params = {
            "scenario": vscenario,
            "zone_scenario": zone_scenario,
            "physics_params": PhysicsParams(),
            "heuristic_params": TABSHeuristicConfig(),
        }
        env_params = jax.tree.map(
            lambda x: jnp.repeat(x[None], config["NUM_ENVS"], axis=0), env_params
        )
        obsv, env_state = jax.vmap(env.reset, in_axes=(0, 0))(reset_rng, env_params)
        init_hstate = ScannedRNN.initialize_carry(config["NUM_ACTORS"], config["GRU_HIDDEN_DIM"])

        # TRAIN LOOP
        def _update_step(update_runner_state, unused):
            # COLLECT TRAJECTORIES
            runner_state, update_steps = update_runner_state

            def _env_step(runner_state, unused):
                train_state, env_state, last_obs, last_done, hstate, rng = runner_state

                # SELECT ACTION
                rng, _rng = jax.random.split(rng)
                avail_actions = jax.vmap(env.get_avail_actions)(env_state)
                avail_actions = jax.lax.stop_gradient(
                    batchify(avail_actions, env.agents, config["NUM_ACTORS"])
                )
                obs_batch = batchify(last_obs, env.agents, config["NUM_ACTORS"])
                ac_in = (
                    obs_batch[np.newaxis, :],
                    last_done[np.newaxis, :],
                    avail_actions,
                )
                hstate, pi, value = network.apply(train_state.params, hstate, ac_in)
                action = pi.sample(seed=_rng)
                log_prob = pi.log_prob(action)
                env_act = unbatchify(action, env.agents, config["NUM_ENVS"], env.num_agents)
                env_act = {k: v.squeeze() for k, v in env_act.items()}

                # STEP ENV
                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, config["NUM_ENVS"])
                obsv, env_state, reward, done, info = jax.vmap(env.step, in_axes=(0, 0, 0, 0))(
                    rng_step, env_state, env_act, env_params
                )
                # info = jax.tree.map(lambda x: x.reshape((config["NUM_ACTORS"])), info)
                done_batch = batchify(done, env.agents, config["NUM_ACTORS"]).squeeze()
                transition = Transition(
                    jnp.tile(done["__all__"], env.num_agents),
                    last_done,
                    action.squeeze(),
                    value.squeeze(),
                    batchify(reward, env.agents, config["NUM_ACTORS"]).squeeze(),
                    log_prob.squeeze(),
                    obs_batch,
                    info,
                    avail_actions,
                )
                runner_state = (train_state, env_state, obsv, done_batch, hstate, rng)
                return runner_state, transition

            initial_hstate = runner_state[-2]
            runner_state, traj_batch = jax.lax.scan(
                _env_step, runner_state, None, config["NUM_STEPS"]
            )

            # CALCULATE ADVANTAGE
            train_state, env_state, last_obs, last_done, hstate, rng = runner_state
            last_obs_batch = batchify(last_obs, env.agents, config["NUM_ACTORS"])
            avail_actions = jnp.ones((config["NUM_ACTORS"], env.action_space(env.agents[0]).n))
            ac_in = (
                last_obs_batch[np.newaxis, :],
                last_done[np.newaxis, :],
                avail_actions,
            )
            _, _, last_val = network.apply(train_state.params, hstate, ac_in)
            last_val = last_val.squeeze()

            def _calculate_gae(traj_batch, last_val):
                def _get_advantages(gae_and_next_value, transition):
                    gae, next_value = gae_and_next_value
                    done, value, reward = (
                        transition.global_done,
                        transition.value,
                        transition.reward,
                    )
                    delta = reward + config["GAMMA"] * next_value * (1 - done) - value
                    gae = delta + config["GAMMA"] * config["GAE_LAMBDA"] * (1 - done) * gae
                    return (gae, value), gae

                _, advantages = jax.lax.scan(
                    _get_advantages,
                    (jnp.zeros_like(last_val), last_val),
                    traj_batch,
                    reverse=True,
                    unroll=16,
                )
                return advantages, advantages + traj_batch.value

            advantages, targets = _calculate_gae(traj_batch, last_val)

            # UPDATE NETWORK
            def _update_epoch(update_state, unused):
                def _update_minbatch(train_state, batch_info):
                    init_hstate, traj_batch, advantages, targets = batch_info

                    def _loss_fn(params, init_hstate, traj_batch, gae, targets):
                        # RERUN NETWORK
                        _, pi, value = network.apply(
                            params,
                            init_hstate.squeeze(),
                            (traj_batch.obs, traj_batch.done, traj_batch.avail_actions),
                        )
                        log_prob = pi.log_prob(traj_batch.action)

                        # CALCULATE VALUE LOSS
                        value_pred_clipped = traj_batch.value + (value - traj_batch.value).clip(
                            -config["CLIP_EPS"], config["CLIP_EPS"]
                        )
                        value_losses = jnp.square(value - targets)
                        value_losses_clipped = jnp.square(value_pred_clipped - targets)
                        value_loss = 0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()

                        # CALCULATE ACTOR LOSS
                        logratio = log_prob - traj_batch.log_prob
                        ratio = jnp.exp(logratio)
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

                        total_loss = (
                            loss_actor
                            + config["VF_COEF"] * value_loss
                            - config["ENT_COEF"] * entropy
                        )
                        return total_loss, (
                            value_loss,
                            loss_actor,
                            entropy,
                            ratio,
                            approx_kl,
                            clip_frac,
                        )

                    grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                    total_loss, grads = grad_fn(
                        train_state.params, init_hstate, traj_batch, advantages, targets
                    )
                    train_state = train_state.apply_gradients(grads=grads)
                    return train_state, total_loss

                (
                    train_state,
                    init_hstate,
                    traj_batch,
                    advantages,
                    targets,
                    rng,
                ) = update_state
                rng, _rng = jax.random.split(rng)

                # adding an additional "fake" dimensionality to perform minibatching correctly
                init_hstate = jnp.reshape(init_hstate, (1, config["NUM_ACTORS"], -1))
                batch = (
                    init_hstate,
                    traj_batch,
                    advantages.squeeze(),
                    targets.squeeze(),
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

                train_state, total_loss = jax.lax.scan(_update_minbatch, train_state, minibatches)
                update_state = (
                    train_state,
                    init_hstate.squeeze(),
                    traj_batch,
                    advantages,
                    targets,
                    rng,
                )
                return update_state, total_loss

            update_state = (
                train_state,
                initial_hstate,
                traj_batch,
                advantages,
                targets,
                rng,
            )
            update_state, loss_info = jax.lax.scan(
                _update_epoch, update_state, None, config["UPDATE_EPOCHS"]
            )
            train_state = update_state[0]
            metric = get_battle_metric(env, env_state)
            ratio_0 = loss_info[1][3].at[0, 0].get().mean()
            loss_info = jax.tree.map(lambda x: x.mean(), loss_info)
            metric["loss"] = {
                "total_loss": loss_info[0],
                "value_loss": loss_info[1][0],
                "actor_loss": loss_info[1][1],
                "entropy": loss_info[1][2],
                "ratio": loss_info[1][3],
                "ratio_0": ratio_0,
                "approx_kl": loss_info[1][4],
                "clip_frac": loss_info[1][5],
            }

            rng = update_state[-1]

            def callback(metric):
                wandb.log(metric)

            metric["update_steps"] = update_steps
            jax.experimental.io_callback(callback, None, metric)
            update_steps = update_steps + 1
            runner_state = (train_state, env_state, last_obs, last_done, hstate, rng)
            return (runner_state, update_steps), metric

        rng, _rng = jax.random.split(rng)
        runner_state = (
            train_state,
            env_state,
            obsv,
            jnp.zeros((config["NUM_ACTORS"]), dtype=bool),
            init_hstate,
            _rng,
        )
        runner_state, metric = jax.lax.scan(
            _update_step, (runner_state, 0), None, config["NUM_UPDATES"]
        )
        return {"runner_state": runner_state}

    return train


if __name__ == "__main__":
    config = tyro.cli(Config)
    wandb.init(project=config.PROJECT_NAME, mode="online", config=config)
    train_fn = make_train(config.__dict__)
    with jax.disable_jit(False):
        result = train_fn(jax.random.key(config.SEED))
