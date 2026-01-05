"""
Based on JaxMARL Implementation of MAPPO and JaxUED Implementation of PLR
"""

import os
from dataclasses import dataclass
from enum import IntEnum
from typing import Literal

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import chex
import jax
import jax.numpy as jnp
import numpy as np
import optax
import tyro
import wandb
from flax import core, struct
from flax.training.train_state import TrainState

from src.baseline.layers import ActorRNN, CriticRNN, ScannedRNN
from src.baseline.ued.level_generator import (
    FREE_PARAM_TYPES,
    level_generator,
    mutate_level_generator,
)
from src.baseline.ued.level_sampler import LevelSampler
from src.baseline.ued.scores import compute_max_returns, max_mc, positive_value_loss
from src.baseline.utils import batchify, get_battle_metric, unbatchify
from src.tabs import TABS
from src.tabs.config import PhysicsParams, TABSHeuristicConfig
from src.tabs.scenarios import build_batched_scenarios
from src.tabs.utils import Transition
from src.tabs.wrappers.wrappers import (
    TABSAutoResetWrapper,
    TABSEnemyHeuristicWrapper,
    TABSLogWrapper,
)


@dataclass
class Config:
    LR: float = 0.004
    NUM_ENVS: int = 128
    NUM_STEPS: int = 256
    GRU_HIDDEN_DIM: int = 128
    FC_DIM_SIZE: int = 128
    TOTAL_TIMESTEPS: int = 5e7  # NOTE
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
    SCENARIO: str = "2F1K2A1H_2L"
    FREE_PARAM_TYPE: Literal["zone", "unit_spec", "heuristic_config"] = "zone"
    # PLR
    SCORE_FUNC: str = "MaxMC"  # MaxMC, pvl
    EXPLORATORY_GRAD_UPDATES: bool = False  # False if Robust PLR or ACCEL
    LEVEL_BUFFER_CAPACITY: int = 4000
    REPLAY_PROB: float = 0.8
    STALENESS_COEF: float = 0.3
    MINIMUM_FILL_RATIO: float = 0.5
    PRIORITIZATION: str = "rank"  # rank, topk
    TEMPERATURE: float = 0.3
    TOPK_K: int = 4
    BUFFER_DUPLICATE_CHECK: bool = True
    # Accel
    USE_ACCEL: bool = False
    NUM_EDITS: int = 5
    # Eval.
    EVAL_STEPS: int = 256
    NUM_EVAL: int = 10
    # Misc.
    SEED: int = 0
    PROJECT_NAME: str = "plr_mappo_rnn"  # wandb project name


class UpdateState(IntEnum):
    DR = 0
    REPLAY = 1


@struct.dataclass
class SampleState:
    sampler: core.FrozenDict[str, chex.ArrayTree] = struct.field(pytree_node=True)
    update_state: UpdateState = struct.field(pytree_node=True)
    # === Below is used for logging ===
    num_dr_updates: int
    num_replay_updates: int
    num_mutation_updates: int
    dr_last_level_batch: chex.ArrayTree = struct.field(pytree_node=True)
    replay_last_level_batch: chex.ArrayTree = struct.field(pytree_node=True)
    mutation_last_level_batch: chex.ArrayTree = struct.field(pytree_node=True)


def make_train(config):
    vscenario, zone_scenario, tabs_config = build_batched_scenarios(
        scenario_names=config["SCENARIO"], n_repeat=config["NUM_ENVS"]
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

    def compute_score(dones, values, max_returns, advantages):
        if config["SCORE_FUNC"] == "MaxMC":
            return max_mc(dones, values, max_returns)
        elif config["SCORE_FUNC"] == "pvl":
            return positive_value_loss(dones, advantages)
        else:
            raise ValueError(f"Unknown score function: {config['SCORE_FUNC']}")

    def train(rng):
        # INIT NETWORK
        actor_network = ActorRNN(env.action_space(env.agents[0]).n, config=config)
        critic_network = CriticRNN(config=config)
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

        # Level sampler
        level_sampler = LevelSampler(
            capacity=config["LEVEL_BUFFER_CAPACITY"],
            replay_prob=config["REPLAY_PROB"],
            staleness_coef=config["STALENESS_COEF"],
            minimum_fill_ratio=config["MINIMUM_FILL_RATIO"],
            prioritization=config["PRIORITIZATION"],
            prioritization_params={"temperature": config["TEMPERATURE"], "k": config["TOPK_K"]},
            duplicate_check=config["BUFFER_DUPLICATE_CHECK"],
        )

        rng, _rng = jax.random.split(rng)
        init_env_params = jax.tree.map(
            lambda x: jnp.repeat(x[None], config["NUM_ENVS"], axis=0),
            {
                "physics_params": PhysicsParams(),
                "heuristic_params": TABSHeuristicConfig(),
            },
        ) | {
            "scenario": vscenario,
            "zone_scenario": zone_scenario,
        }
        sample_random_level = level_generator(FREE_PARAM_TYPES[config["FREE_PARAM_TYPE"]])
        mutate_level = mutate_level_generator(FREE_PARAM_TYPES[config["FREE_PARAM_TYPE"]])
        pholder_level = sample_random_level(init_env_params, _rng)
        pholder_level_batch = jax.tree.map(
            lambda x: jnp.repeat(x[None], config["NUM_ENVS"], axis=0), pholder_level
        )
        sample_state = SampleState(
            sampler=level_sampler.initialize(pholder_level, {"max_return": -jnp.inf}),
            update_state=0,
            num_dr_updates=0,
            num_replay_updates=0,
            num_mutation_updates=0,
            dr_last_level_batch=pholder_level_batch,
            replay_last_level_batch=pholder_level_batch,
            mutation_last_level_batch=pholder_level_batch,
        )

        # INIT ENV
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config["NUM_ENVS"])

        obsv, env_state = jax.vmap(env.reset, in_axes=(0, 0))(reset_rng, pholder_level_batch)
        ac_init_hstate = ScannedRNN.initialize_carry(config["NUM_ACTORS"], 128)
        cr_init_hstate = ScannedRNN.initialize_carry(config["NUM_ACTORS"], 128)

        # For evaluation
        rng, _rng, _rng_reset = jax.random.split(rng, 3)
        sample_rngs = jax.random.split(_rng, config["NUM_EVAL"])
        eval_levels = jax.vmap(sample_random_level, in_axes=(None, 0))(init_env_params, sample_rngs)
        reset_rngs = jax.random.split(_rng_reset, config["NUM_EVAL"])
        eval_obsv, eval_env_state = jax.vmap(env.reset, in_axes=(0, 0))(reset_rngs, eval_levels)

        # # TRAIN LOOP
        def _update_step(update_runner_state, unused):
            # COLLECT TRAJECTORIES
            runner_state, update_steps = update_runner_state

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

            # Rollout
            def _rollout(runner_state, unused):
                train_states, env_state, last_obs, last_done, hstates, env_params, rng = (
                    runner_state
                )

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
                ac_hstate, pi = actor_network.apply(train_states[0].params, hstates[0], ac_in)
                action = pi.sample(seed=_rng)
                log_prob = pi.log_prob(action)
                env_act = unbatchify(action, env.agents, config["NUM_ENVS"], env.num_agents)
                env_act = {k: v.squeeze() for k, v in env_act.items()}

                # VALUE
                # world_state is (num_envs, world_state_size)
                # repeat for each agent to get (num_actors, world_state_size)
                world_state = last_obs["world_state"]  # (NUM_ENVS, 280)
                world_state = jnp.repeat(world_state, env.num_agents, axis=0)  # (NUM_ACTORS, 280)

                cr_in = (
                    world_state[None, :],
                    last_done[np.newaxis, :],
                )
                cr_hstate, value = critic_network.apply(train_states[1].params, hstates[1], cr_in)

                # STEP ENV
                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, config["NUM_ENVS"])
                obsv, env_state, reward, done, info = jax.vmap(env.step, in_axes=(0, 0, 0, 0))(
                    rng_step, env_state, env_act, env_params
                )
                done_batch = batchify(done, env.agents, config["NUM_ACTORS"]).squeeze()
                transition = Transition(
                    jnp.tile(done["__all__"], env.num_agents),
                    last_done,
                    action.squeeze(),
                    value.squeeze(),
                    batchify(reward, env.agents, config["NUM_ACTORS"]).squeeze(),
                    log_prob.squeeze(),
                    obs_batch,
                    world_state,
                    info,
                    avail_actions,
                )
                runner_state = (
                    train_states,
                    env_state,
                    obsv,
                    done_batch,
                    (ac_hstate, cr_hstate),
                    env_params,
                    rng,
                )
                return runner_state, transition

            def _update(runner_state):
                # Update network
                def _update_epoch(
                    train_states,
                    initial_hstates,
                    traj_batch,
                    advantages,
                    targets,
                    rng,
                    update_grad: bool = True,
                ):
                    def _update_epoch_fn(update_state, unused):
                        (
                            train_states,
                            init_hstates,
                            traj_batch,
                            advantages,
                            targets,
                            rng,
                        ) = update_state

                        def _update_minbatch(train_states, batch_info):
                            actor_train_state, critic_train_state = train_states
                            ac_init_hstate, cr_init_hstate, traj_batch, advantages, targets = (
                                batch_info
                            )

                            def _actor_loss_fn(actor_params, init_hstate, traj_batch, gae):
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

                                return actor_loss, (
                                    loss_actor,
                                    entropy,
                                    ratio,
                                    approx_kl,
                                    clip_frac,
                                )

                            def _critic_loss_fn(critic_params, init_hstate, traj_batch, targets):
                                # RERUN NETWORK
                                _, value = critic_network.apply(
                                    critic_params,
                                    init_hstate.squeeze(),
                                    (traj_batch.world_state, traj_batch.done),
                                )

                                # CALCULATE VALUE LOSS
                                value_pred_clipped = traj_batch.value + (
                                    value - traj_batch.value
                                ).clip(-config["CLIP_EPS"], config["CLIP_EPS"])
                                value_losses = jnp.square(value - targets)
                                value_losses_clipped = jnp.square(value_pred_clipped - targets)
                                value_loss = (
                                    0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()
                                )
                                critic_loss = config["VF_COEF"] * value_loss
                                return critic_loss, (value_loss)

                            actor_grad_fn = jax.value_and_grad(_actor_loss_fn, has_aux=True)
                            actor_loss, actor_grads = actor_grad_fn(
                                actor_train_state.params, ac_init_hstate, traj_batch, advantages
                            )
                            critic_grad_fn = jax.value_and_grad(_critic_loss_fn, has_aux=True)
                            critic_loss, critic_grads = critic_grad_fn(
                                critic_train_state.params, cr_init_hstate, traj_batch, targets
                            )

                            if update_grad:
                                actor_train_state = actor_train_state.apply_gradients(
                                    grads=actor_grads
                                )
                                critic_train_state = critic_train_state.apply_gradients(
                                    grads=critic_grads
                                )

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

                        rng, _rng = jax.random.split(rng)

                        init_hstates = jax.tree.map(
                            lambda x: jnp.reshape(x, (1, config["NUM_ACTORS"], -1)), init_hstates
                        )

                        batch = (
                            init_hstates[0],
                            init_hstates[1],
                            traj_batch,
                            advantages.squeeze(),
                            targets.squeeze(),
                        )
                        permutation = jax.random.permutation(_rng, config["NUM_ACTORS"])

                        shuffled_batch = jax.tree.map(
                            lambda x: jnp.take(x, permutation, axis=1), batch
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

                        train_states, loss_info = jax.lax.scan(
                            _update_minbatch, train_states, minibatches
                        )
                        update_state = (
                            train_states,
                            jax.tree.map(lambda x: x.squeeze(), init_hstates),
                            traj_batch,
                            advantages,
                            targets,
                            rng,
                        )
                        return update_state, loss_info

                    update_state = (
                        train_states,
                        initial_hstates,
                        traj_batch,
                        advantages,
                        targets,
                        rng,
                    )
                    update_state, loss_info = jax.lax.scan(
                        _update_epoch_fn, update_state, None, config["UPDATE_EPOCHS"]
                    )

                    return update_state, loss_info

                def _on_new_levels(runner_state):
                    """
                    Samples new (randomly-generated) levels and evaluates the policy on these.
                    It also then adds the levels to the level buffer if they have high-enough scores.
                    The agent is updated on these trajectories iff `config["exploratory_grad_updates"]` is True.
                    """
                    train_states, sample_state, env_state, rng = runner_state

                    sampler = sample_state.sampler

                    # Reset
                    rng, rng_levels, rng_reset = jax.random.split(rng, 3)
                    new_levels = jax.vmap(sample_random_level, in_axes=(None, 0))(
                        init_env_params, jax.random.split(rng_levels, config["NUM_ENVS"])
                    )
                    reset_rng = jax.random.split(rng_reset, config["NUM_ENVS"])
                    init_obs, env_state = jax.vmap(env.reset, in_axes=(0, 0))(reset_rng, new_levels)

                    runner_state = (
                        train_states,
                        env_state,
                        init_obs,
                        jnp.zeros((config["NUM_ACTORS"]), dtype=bool),
                        (ac_init_hstate, cr_init_hstate),
                        new_levels,
                        rng,
                    )

                    # Rollout
                    runner_state, traj_batch = jax.lax.scan(
                        _rollout, runner_state, None, config["NUM_STEPS"]
                    )

                    # Calculate Advantage
                    train_states, env_state, last_obs, last_done, hstates, new_levels, rng = (
                        runner_state
                    )

                    last_world_state = last_obs["world_state"]  # (NUM_ENVS, 280)
                    last_world_state = jnp.repeat(
                        last_world_state, env.num_agents, axis=0
                    )  # (NUM_ACTORS, 280)
                    cr_in = (
                        last_world_state[None, :],
                        last_done[np.newaxis, :],
                    )
                    _, last_val = critic_network.apply(train_states[1].params, hstates[1], cr_in)
                    last_val = last_val.squeeze()

                    advantages, targets = _calculate_gae(traj_batch, last_val)

                    # Calculate scores
                    _done = traj_batch.global_done.reshape(
                        config["NUM_STEPS"], -1, config["NUM_ENVS"]
                    ).mean(axis=1)
                    _reward = traj_batch.reward.reshape(
                        config["NUM_STEPS"], -1, config["NUM_ENVS"]
                    ).mean(axis=1)
                    max_returns = compute_max_returns(_done, _reward)
                    scores = compute_score(
                        _done,
                        traj_batch.value.reshape(config["NUM_STEPS"], -1, config["NUM_ENVS"]).mean(
                            axis=1
                        ),
                        max_returns,
                        advantages.reshape(config["NUM_STEPS"], -1, config["NUM_ENVS"]).mean(
                            axis=1
                        ),
                    )
                    sampler, _ = level_sampler.insert_batch(
                        sampler, new_levels, scores, {"max_return": max_returns}
                    )

                    sample_state = sample_state.replace(
                        sampler=sampler,
                        update_state=UpdateState.DR,
                        num_dr_updates=sample_state.num_dr_updates + 1,
                        dr_last_level_batch=new_levels,
                    )

                    # Update
                    update_state, loss_info = _update_epoch(
                        train_states,
                        (ac_init_hstate, cr_init_hstate),
                        traj_batch,
                        advantages,
                        targets,
                        rng,
                        update_grad=config["EXPLORATORY_GRAD_UPDATES"],
                    )

                    train_states, hstates, traj_batch, advantages, targets, rng = update_state

                    output_state = (train_states, sample_state, env_state, rng)

                    return output_state, loss_info

                def _on_replay_levels(runner_state):
                    """
                    This samples levels from the level buffer.
                    """
                    train_states, sample_state, env_state, rng = runner_state

                    sampler = sample_state.sampler

                    # Collect trajectories on replay levels
                    rng, rng_levels, rng_reset = jax.random.split(rng, 3)
                    sampler, (level_inds, levels) = level_sampler.sample_replay_levels(
                        sampler, rng_levels, config["NUM_ENVS"]
                    )
                    reset_rng = jax.random.split(rng_reset, config["NUM_ENVS"])
                    init_obs, env_state = jax.vmap(env.reset, in_axes=(0, 0))(reset_rng, levels)

                    runner_state = (
                        train_states,
                        env_state,
                        init_obs,
                        jnp.zeros((config["NUM_ACTORS"]), dtype=bool),
                        (ac_init_hstate, cr_init_hstate),
                        levels,
                        rng,
                    )

                    # Rollout
                    runner_state, traj_batch = jax.lax.scan(
                        _rollout, runner_state, None, config["NUM_STEPS"]
                    )

                    # Calculate Advantage
                    train_states, env_state, last_obs, last_done, hstates, levels, rng = (
                        runner_state
                    )

                    last_world_state = last_obs["world_state"]  # (NUM_ENVS, 280)
                    last_world_state = jnp.repeat(
                        last_world_state, env.num_agents, axis=0
                    )  # (NUM_ACTORS, 280)
                    cr_in = (
                        last_world_state[None, :],
                        last_done[np.newaxis, :],
                    )
                    _, last_val = critic_network.apply(train_states[1].params, hstates[1], cr_in)
                    last_val = last_val.squeeze()

                    advantages, targets = _calculate_gae(traj_batch, last_val)

                    # Calculate scores
                    _done = traj_batch.global_done.reshape(
                        config["NUM_STEPS"], -1, config["NUM_ENVS"]
                    ).mean(axis=1)
                    _reward = traj_batch.reward.reshape(
                        config["NUM_STEPS"], -1, config["NUM_ENVS"]
                    ).mean(axis=1)
                    max_returns = jnp.maximum(
                        level_sampler.get_levels_extra(sampler, level_inds)["max_return"],
                        compute_max_returns(_done, _reward),
                    )
                    scores = compute_score(
                        _done,
                        traj_batch.value.reshape(config["NUM_STEPS"], -1, config["NUM_ENVS"]).mean(
                            axis=1
                        ),
                        max_returns,
                        advantages.reshape(config["NUM_STEPS"], -1, config["NUM_ENVS"]).mean(
                            axis=1
                        ),
                    )
                    sampler = level_sampler.update_batch(
                        sampler, level_inds, scores, {"max_return": max_returns}
                    )

                    sample_state = sample_state.replace(
                        sampler=sampler,
                        update_state=UpdateState.REPLAY,
                        num_replay_updates=sample_state.num_replay_updates + 1,
                        replay_last_level_batch=levels,
                    )

                    # Update
                    update_state, loss_info = _update_epoch(
                        train_states,
                        (ac_init_hstate, cr_init_hstate),
                        traj_batch,
                        advantages,
                        targets,
                        rng,
                        update_grad=True,
                    )

                    train_states, hstates, traj_batch, advantages, targets, rng = update_state

                    output_state = (train_states, sample_state, env_state, rng)

                    return output_state, loss_info

                def _on_mutate_levels(runner_state):
                    """
                    This mutates the previous batch of replay levels and potentially adds them to the level buffer.
                    This also updates the policy iff `config["exploratory_grad_updates"]` is True.
                    """
                    train_states, sample_state, env_state, rng = runner_state

                    sampler = sample_state.sampler

                    # Mutate
                    rng, rng_mutate, rng_reset = jax.random.split(rng, 3)
                    parent_levels = sample_state.replay_last_level_batch
                    child_levels = jax.vmap(mutate_level, (0, 0))(
                        parent_levels, jax.random.split(rng_mutate, config["NUM_ENVS"])
                    )
                    reset_rng = jax.random.split(rng_reset, config["NUM_ENVS"])
                    init_obs, env_state = jax.vmap(env.reset, in_axes=(0, 0))(
                        reset_rng, child_levels
                    )

                    runner_state = (
                        train_states,
                        env_state,
                        init_obs,
                        jnp.zeros((config["NUM_ACTORS"]), dtype=bool),
                        (ac_init_hstate, cr_init_hstate),
                        child_levels,
                        rng,
                    )

                    # Rollout
                    runner_state, traj_batch = jax.lax.scan(
                        _rollout, runner_state, None, config["NUM_STEPS"]
                    )

                    # Calculate Advantage
                    train_states, env_state, last_obs, last_done, hstates, child_levels, rng = (
                        runner_state
                    )

                    last_world_state = last_obs["world_state"]  # (NUM_ENVS, 280)
                    last_world_state = jnp.repeat(
                        last_world_state, env.num_agents, axis=0
                    )  # (NUM_ACTORS, 280)
                    cr_in = (
                        last_world_state[None, :],
                        last_done[np.newaxis, :],
                    )
                    _, last_val = critic_network.apply(train_states[1].params, hstates[1], cr_in)
                    last_val = last_val.squeeze()

                    advantages, targets = _calculate_gae(traj_batch, last_val)

                    # Calculate scores
                    _done = traj_batch.global_done.reshape(
                        config["NUM_STEPS"], -1, config["NUM_ENVS"]
                    ).mean(axis=1)
                    _reward = traj_batch.reward.reshape(
                        config["NUM_STEPS"], -1, config["NUM_ENVS"]
                    ).mean(axis=1)
                    max_returns = compute_max_returns(_done, _reward)
                    scores = compute_score(
                        _done,
                        traj_batch.value.reshape(config["NUM_STEPS"], -1, config["NUM_ENVS"]).mean(
                            axis=1
                        ),
                        max_returns,
                        advantages.reshape(config["NUM_STEPS"], -1, config["NUM_ENVS"]).mean(
                            axis=1
                        ),
                    )
                    sampler, _ = level_sampler.insert_batch(
                        sampler, child_levels, scores, {"max_return": max_returns}
                    )

                    sample_state = sample_state.replace(
                        sampler=sampler,
                        update_state=UpdateState.DR,
                        num_mutation_updates=sample_state.num_mutation_updates + 1,
                        mutation_last_level_batch=child_levels,
                    )

                    # Update
                    update_state, loss_info = _update_epoch(
                        train_states,
                        (ac_init_hstate, cr_init_hstate),
                        traj_batch,
                        advantages,
                        targets,
                        rng,
                        update_grad=config["EXPLORATORY_GRAD_UPDATES"],
                    )

                    train_states, hstates, traj_batch, advantages, targets, rng = update_state

                    output_state = (train_states, sample_state, env_state, rng)

                    return output_state, loss_info

                sample_state = runner_state[1]
                rng = runner_state[-1]

                rng, _rng = jax.random.split(rng)
                # The train step makes a decision on which branch to take, either on_new, on_replay or on_mutate.
                # on_mutate is only called if the replay branch has been taken before (as it uses `train_state.update_state`)
                if config["USE_ACCEL"]:
                    # Adversarially Compounding Complexity by Editing Levels (ACCEL)
                    s = sample_state.update_state
                    branch = (1 - s) * level_sampler.sample_replay_decision(
                        sample_state.sampler, _rng
                    ) + 2 * s
                else:
                    # Prioritized Level Replay (PLR)
                    branch = level_sampler.sample_replay_decision(
                        sample_state.sampler, _rng
                    ).astype(int)

                output_state, loss_info = jax.lax.switch(
                    branch, [_on_new_levels, _on_replay_levels, _on_mutate_levels], runner_state
                )

                return output_state, loss_info

            output_state, loss_info = _update(runner_state)

            (train_states, sample_state, env_state, rng) = output_state

            loss_info["ratio_0"] = loss_info["ratio"].at[0, 0].get()
            loss_info = jax.tree.map(lambda x: x.mean(), loss_info)

            metric = loss_info

            metric |= get_battle_metric(env, env_state)

            # UED logs
            metric["num_dr_updates"] = sample_state.num_dr_updates
            metric["num_replay_updates"] = sample_state.num_replay_updates
            metric["num_mutation_updates"] = sample_state.num_mutation_updates
            metric["update_count"] = (
                sample_state.num_dr_updates
                + sample_state.num_replay_updates
                + sample_state.num_mutation_updates
            )

            # TODO: Log rendering levels
            highest_scoring_level = level_sampler.get_levels(
                sample_state.sampler, sample_state.sampler["scores"].argmax()
            )
            highest_weighted_level = level_sampler.get_levels(
                sample_state.sampler, level_sampler.level_weights(sample_state.sampler).argmax()
            )

            def evaluate(actor_train_state, rng):
                BATCH_ACTORS = config["NUM_EVAL"] * env.num_agents

                def _rollout(carry, unused):
                    actor_train_state, env_state, last_obs, last_done, ac_hstate, levels, rng = (
                        carry
                    )

                    # SELECT ACTION
                    rng, _rng = jax.random.split(rng)
                    avail_actions = jax.vmap(env.get_avail_actions)(env_state)
                    avail_actions = jax.lax.stop_gradient(
                        batchify(avail_actions, env.agents, BATCH_ACTORS)
                    )
                    obs_batch = batchify(last_obs, env.agents, BATCH_ACTORS)
                    ac_in = (
                        obs_batch[np.newaxis, :],
                        last_done[np.newaxis, :],
                        avail_actions,
                    )
                    ac_hstate, pi = actor_network.apply(actor_train_state.params, ac_hstate, ac_in)
                    action = pi.sample(seed=_rng)
                    env_act = unbatchify(action, env.agents, config["NUM_EVAL"], env.num_agents)
                    env_act = {k: v.squeeze() for k, v in env_act.items()}

                    # STEP ENV
                    rng, _rng = jax.random.split(rng)
                    rng_step = jax.random.split(_rng, config["NUM_EVAL"])
                    obsv, env_state, reward, done, info = jax.vmap(env.step, in_axes=(0, 0, 0, 0))(
                        rng_step, env_state, env_act, levels
                    )
                    done_batch = batchify(done, env.agents, BATCH_ACTORS).squeeze()

                    carry = (actor_train_state, env_state, obsv, done_batch, ac_hstate, levels, rng)

                    return carry, None

                carry = (
                    actor_train_state,
                    eval_env_state,
                    eval_obsv,
                    jnp.zeros((BATCH_ACTORS), dtype=bool),
                    ScannedRNN.initialize_carry(BATCH_ACTORS, 128),
                    eval_levels,
                    rng,
                )
                carry, _ = jax.lax.scan(_rollout, carry, None, config["EVAL_STEPS"])
                metric = get_battle_metric(env, carry[1])

                eval_metric = {}
                for key in metric.keys():
                    eval_metric[f"eval/{key}"] = metric[key]

                return eval_metric

            rng, _rng = jax.random.split(rng)
            eval_metric = evaluate(train_states[0], _rng)

            metric |= eval_metric

            def callback(metric):
                wandb.log(metric)

            update_steps = update_steps + 1
            jax.experimental.io_callback(callback, None, metric)
            runner_state = (train_states, sample_state, env_state, rng)
            return (runner_state, update_steps), metric

        rng, _rng = jax.random.split(rng)
        runner_state = ((actor_train_state, critic_train_state), sample_state, env_state, _rng)
        runner_state, metric = jax.lax.scan(
            _update_step, (runner_state, 0), None, config["NUM_UPDATES"]
        )
        return {"runner_state": runner_state, "metric": metric}

    return train


if __name__ == "__main__":
    config = tyro.cli(Config)
    wandb.init(project=config.PROJECT_NAME, mode="online", config=config)
    train_fn = make_train(config.__dict__)
    with jax.disable_jit(False):
        result = train_fn(jax.random.key(config.SEED))
