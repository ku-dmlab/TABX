"""
Based on JaxMARL Implementation of MAPPO
"""

import hashlib
import json
import os
from dataclasses import dataclass
from typing import Literal, NamedTuple, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import optax
import tyro
from flax.training.train_state import TrainState

import wandb
from src.baseline.layers import ActorCriticRNN, ScannedRNN
from src.baseline.utils import (
    batchify,
    dataclass_to_dict,
    get_battle_metric,
    save_params,
    unbatchify,
)
from src.tabs import TABS, build_batched_env_params_and_config
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
    LN_EPS: float = 1e-6
    # Env
    SCENARIO: str = "elbow"
    PHYSICS: str = "default"
    HEURISTIC: str = "easy"
    WORLD_STATE_TYPE: Literal["concat", "global"] = "global"
    # Misc.
    SEED: int | Tuple[int, ...] = 0
    ALGORITHM: str = "ippo"  # for distinguishing wandb runs
    PROJECT_NAME: str = "ippo_rnn"  # wandb project name
    SAVE_PATH: str = "./ckpt"
    SAVE_VIDEO: bool = False


def make_train(config):
    env_params, tabs_config = build_batched_env_params_and_config(
        scenario_names=config["SCENARIO"],
        physics_param_names=config["PHYSICS"],
        heuristic_param_names=config["HEURISTIC"],
        n_repeat=config["NUM_ENVS"],
    )
    env = TABS(cfg=tabs_config, world_state_type=config["WORLD_STATE_TYPE"])
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
        init_rng = rng
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
        train_state = TrainState.create(apply_fn=network.apply, params=network_params, tx=tx)

        # INIT ENV
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config["NUM_ENVS"])

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
                ac_in = (obs_batch[np.newaxis, :], last_done[np.newaxis, :], avail_actions)
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
                batch = (init_hstate, traj_batch, advantages.squeeze(), targets.squeeze())
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

            update_state = (train_state, initial_hstate, traj_batch, advantages, targets, rng)
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

            metric["update_steps"] = update_steps
            update_steps = update_steps + 1
            jax.experimental.io_callback(callback, None, metric, init_rng, update_steps)
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
        runner_state[0].params,
        os.path.join(save_path, f"{config.SCENARIO}_seed{config.SEED}_actor.safetensors"),
    )
    if isinstance(config.SEED, tuple):
        runner_state = jax.tree.map(lambda x: x[0], runner_state)

    if config.SAVE_VIDEO:
        # Visualize
        from src.tabs.visualize import Visualizer

        env_params, tabs_config = build_batched_env_params_and_config(
            scenario_names=config.SCENARIO
        )
        env = TABS(cfg=tabs_config)
        env = TABSEnemyHeuristicWrapper(env)
        num_steps = env.max_episode_steps

        init_hstate = ScannedRNN.initialize_carry(1, 128)
        network = ActorCriticRNN(env.action_space(env.agents[0]).n, config=config.__dict__)

        rng = jax.random.PRNGKey(config.SEED[0] if isinstance(config.SEED, tuple) else config.SEED)
        rng, _rng = jax.random.split(rng)

        obs, env_state = env.reset(_rng, env_params)

        def rollout_body(carry, _):
            (obs, env_state, done, hstate, rng) = carry
            # Random policy
            rng, step_rng, action_rng = jax.random.split(rng, 3)
            avail_actions = env.get_avail_actions(env_state)
            ac_in = (
                jnp.expand_dims(batchify(obs, env.agents, env.num_agents), 1),
                done,
                jnp.expand_dims(batchify(avail_actions, env.agents, env.num_agents), 1),
            )

            hstate, pi, _ = network.apply(runner_state[0].params, hstate, ac_in)
            action = pi.sample(seed=action_rng)
            env_act = unbatchify(action, env.agents, 1, env.num_agents)
            env_act = {k: v.squeeze() for k, v in env_act.items()}
            obs, next_state, reward, done, info = env.step(step_rng, env_state, env_act)

            return (
                obs,
                next_state,
                batchify(done, env.agents, env.num_agents),
                hstate,
                rng,
            ), next_state

        _, stacked = jax.lax.scan(
            rollout_body,
            (obs, env_state, jnp.zeros((env.num_agents, 1), dtype=jnp.bool), init_hstate, rng),
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
