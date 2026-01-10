"""
Based on JaxMARL Implementation of MAPPO
"""

import os
from dataclasses import dataclass
from typing import Literal

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import jax
import jax.numpy as jnp
import numpy as np
import optax
import tyro
from flax.training.train_state import TrainState

import wandb
from src.baseline.layers import ActorRNN, CriticRNN, ScannedRNN
from src.baseline.ued.level_generator import level_generator
from src.baseline.ued.utils import get_evaluation_heuristic_params, get_evaluation_scenarios
from src.baseline.ued.wrappers import LevelAutoResetWrapper
from src.baseline.utils import batchify, get_battle_metric, save_params, unbatchify
from src.tabs import TABS, build_batched_env_params_and_config
from src.tabs.scenarios.constants import CHALLENGES
from src.tabs.utils import Transition
from src.tabs.wrappers import TABSAutoResetWrapper, TABSEnemyHeuristicWrapper, TABSLogWrapper


@dataclass
class Config:
    LR: float = 0.004
    NUM_ENVS: int = 128
    NUM_STEPS: int = 256
    GRU_HIDDEN_DIM: int = 128
    FC_DIM_SIZE: int = 128
    TOTAL_TIMESTEPS: int = 2e7  # NOTE
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
    SCENARIO: str = "1F1M3A1Hvs2F1S1K1A1H_2L"
    PHYSICS: str = "default"
    HEURISTIC: str = "easy"
    FREE_PARAM_TYPE: tuple[Literal["zone", "unit_spec", "heuristic_config"], ...] = ("zone",)
    # Eval.
    EVAL_STEPS: int = 256
    NUM_EVAL: int = 10  # The number of episodes to evaluate
    # Misc.
    SEED: int = 0
    PROJECT_NAME: str = "dr_mappo_rnn"  # wandb project name
    SAVE_PATH: str = "./ckpt"
    SAVE_VIDEO: bool = False


def make_train(config):
    init_env_params, tabs_config = build_batched_env_params_and_config(
        scenario_names=config["SCENARIO"],
        physics_param_names=config["PHYSICS"],
        heuristic_param_names=config["HEURISTIC"],
    )
    env = TABS(cfg=tabs_config)
    env = TABSLogWrapper(env)
    env = TABSEnemyHeuristicWrapper(env)
    sample_random_level = level_generator(config["FREE_PARAM_TYPE"])
    env = LevelAutoResetWrapper(env, sample_random_level)

    config["NUM_ACTORS"] = env.num_agents * config["NUM_ENVS"]
    config["NUM_UPDATES"] = config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    config["MINIBATCH_SIZE"] = (
        config["NUM_ACTORS"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"]
    )
    config["CLIP_EPS"] = (
        config["CLIP_EPS"] / env.num_agents if config["SCALE_CLIP_EPS"] else config["CLIP_EPS"]
    )

    config["NUM_EVAL"] = max(config["NUM_EVAL"], env.max_episode_steps)

    def linear_schedule(count):
        frac = (
            1.0
            - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"]))
            / config["NUM_UPDATES"]
        )
        return config["LR"] * frac

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

        # INIT ENV
        rng, _rng = jax.random.split(rng)
        sample_rngs = jax.random.split(_rng, config["NUM_ENVS"])
        env_params = jax.vmap(sample_random_level, in_axes=(None, 0))(init_env_params, sample_rngs)

        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
        obsv, env_state = jax.vmap(env.reset, in_axes=(0, 0))(reset_rng, env_params)
        ac_init_hstate = ScannedRNN.initialize_carry(config["NUM_ACTORS"], config["GRU_HIDDEN_DIM"])
        cr_init_hstate = ScannedRNN.initialize_carry(config["NUM_ACTORS"], config["GRU_HIDDEN_DIM"])

        # For evaluation
        rng, _rng, _rng_reset = jax.random.split(rng, 3)
        eval_scenarios = get_evaluation_scenarios(
            config["SCENARIO"], list(config["FREE_PARAM_TYPE"])
        )
        n_eval_scenarios = len(eval_scenarios)
        heuristic_params = get_evaluation_heuristic_params(
            config["HEURISTIC"], list(config["FREE_PARAM_TYPE"])
        )
        eval_levels, tabs_config = build_batched_env_params_and_config(
            scenario_names=eval_scenarios,
            physics_param_names=[config["PHYSICS"]] * n_eval_scenarios,
            heuristic_param_names=heuristic_params * n_eval_scenarios,
            n_repeat=config["NUM_EVAL"],
        )
        eval_env = TABS(cfg=tabs_config)
        eval_env = TABSLogWrapper(eval_env)
        eval_env = TABSEnemyHeuristicWrapper(eval_env)
        eval_env = TABSAutoResetWrapper(eval_env)
        reset_rngs = jax.random.split(_rng_reset, config["NUM_EVAL"] * n_eval_scenarios)
        eval_obsv, eval_env_state = jax.vmap(eval_env.reset, in_axes=(0, 0))(
            reset_rngs, eval_levels
        )

        # TRAIN LOOP
        def _update_step(update_runner_state, unused):
            # COLLECT TRAJECTORIES
            runner_state, update_steps = update_runner_state

            def _env_step(runner_state, unused):
                train_states, env_state, last_obs, last_done, hstates, levels, rng = runner_state

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
                # print('env step ac in', ac_in)
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
                    rng_step, env_state, env_act, levels
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
                    info["levels"],
                    rng,
                )
                return runner_state, transition

            initial_hstates = runner_state[-3]
            runner_state, traj_batch = jax.lax.scan(
                _env_step, runner_state, None, config["NUM_STEPS"]
            )

            # CALCULATE ADVANTAGE
            train_states, env_state, last_obs, last_done, hstates, levels, rng = runner_state

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
                def _update_minbatch(train_states, batch_info):
                    actor_train_state, critic_train_state = train_states
                    ac_init_hstate, cr_init_hstate, traj_batch, advantages, targets = batch_info

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

                        return actor_loss, (loss_actor, entropy, ratio, approx_kl, clip_frac)

                    def _critic_loss_fn(critic_params, init_hstate, traj_batch, targets):
                        # RERUN NETWORK
                        _, value = critic_network.apply(
                            critic_params,
                            init_hstate.squeeze(),
                            (traj_batch.world_state, traj_batch.done),
                        )

                        # CALCULATE VALUE LOSS
                        value_pred_clipped = traj_batch.value + (value - traj_batch.value).clip(
                            -config["CLIP_EPS"], config["CLIP_EPS"]
                        )
                        value_losses = jnp.square(value - targets)
                        value_losses_clipped = jnp.square(value_pred_clipped - targets)
                        value_loss = 0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()
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
                    advantages,
                    targets,
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

                # train_states = (actor_train_state, critic_train_state)
                train_states, loss_info = jax.lax.scan(_update_minbatch, train_states, minibatches)
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
                _update_epoch, update_state, None, config["UPDATE_EPOCHS"]
            )
            loss_info["ratio_0"] = loss_info["ratio"].at[0, 0].get()
            loss_info = jax.tree.map(lambda x: x.mean(), loss_info)

            train_states = update_state[0]
            metric = loss_info
            rng = update_state[-1]

            metric |= get_battle_metric(env, env_state)

            def evaluate(actor_train_state, rng):
                BATCH_SIZE = n_eval_scenarios * config["NUM_EVAL"]
                BATCH_ACTORS = BATCH_SIZE * env.num_agents

                def _rollout(carry, unused):
                    actor_train_state, env_state, last_obs, last_done, ac_hstate, levels, rng = (
                        carry
                    )

                    # SELECT ACTION
                    rng, _rng = jax.random.split(rng)
                    avail_actions = jax.vmap(eval_env.get_avail_actions)(env_state)
                    avail_actions = jax.lax.stop_gradient(
                        batchify(avail_actions, eval_env.agents, BATCH_ACTORS)
                    )
                    obs_batch = batchify(last_obs, eval_env.agents, BATCH_ACTORS)
                    ac_in = (
                        obs_batch[np.newaxis, :],
                        last_done[np.newaxis, :],
                        avail_actions,
                    )
                    ac_hstate, pi = actor_network.apply(actor_train_state.params, ac_hstate, ac_in)
                    action = pi.sample(seed=_rng)
                    env_act = unbatchify(action, eval_env.agents, BATCH_SIZE, eval_env.num_agents)
                    env_act = {k: v.squeeze() for k, v in env_act.items()}

                    # STEP ENV
                    rng, _rng = jax.random.split(rng)
                    rng_step = jax.random.split(_rng, BATCH_SIZE)
                    obsv, env_state, reward, done, info = jax.vmap(
                        eval_env.step, in_axes=(0, 0, 0, 0)
                    )(rng_step, env_state, env_act, levels)
                    done_batch = batchify(done, eval_env.agents, BATCH_ACTORS).squeeze()

                    carry = (actor_train_state, env_state, obsv, done_batch, ac_hstate, levels, rng)

                    return carry, None

                carry = (
                    actor_train_state,
                    eval_env_state,
                    eval_obsv,
                    jnp.zeros((BATCH_ACTORS), dtype=bool),
                    ScannedRNN.initialize_carry(BATCH_ACTORS, config["GRU_HIDDEN_DIM"]),
                    eval_levels,
                    rng,
                )
                carry, _ = jax.lax.scan(_rollout, carry, None, config["EVAL_STEPS"])
                _env_state = jax.tree.map(
                    lambda x: x.reshape((n_eval_scenarios, config["NUM_EVAL"]) + x.shape[1:]),
                    carry[1],
                )
                metric = jax.vmap(get_battle_metric, in_axes=(None, 0))(eval_env, _env_state)

                eval_metric = {}
                for key in metric.keys():
                    for s in range(n_eval_scenarios):
                        eval_metric[f"eval/{eval_scenarios[s]}/{key}"] = metric[key][s]

                return eval_metric

            rng, _rng = jax.random.split(rng)
            eval_metric = evaluate(train_states[0], _rng)

            metric |= eval_metric

            def callback(metric):
                wandb.log(metric)

            update_steps = update_steps + 1
            jax.experimental.io_callback(callback, None, metric)
            runner_state = (train_states, env_state, last_obs, last_done, hstates, levels, rng)
            return (runner_state, update_steps), metric

        rng, _rng = jax.random.split(rng)
        runner_state = (
            (actor_train_state, critic_train_state),
            env_state,
            obsv,
            jnp.zeros((config["NUM_ACTORS"]), dtype=bool),
            (ac_init_hstate, cr_init_hstate),
            env_params,
            _rng,
        )
        runner_state, metric = jax.lax.scan(
            _update_step, (runner_state, 0), None, config["NUM_UPDATES"]
        )
        return {"runner_state": runner_state, "metric": metric}

    return train


if __name__ == "__main__":
    config = tyro.cli(Config)
    if config.SCENARIO in CHALLENGES:
        raise ValueError(f"{config.SCENARIO} is not supported in the UED setting.")

    wandb.init(project=config.PROJECT_NAME, mode="online", config=config)
    train_fn = make_train(config.__dict__)
    with jax.disable_jit(False):
        result = train_fn(jax.random.key(config.SEED))

    # Save trained model
    save_path = os.path.join(config.SAVE_PATH, config.PROJECT_NAME)
    os.makedirs(save_path, exist_ok=True)
    runner_state = result["runner_state"][0]
    save_params(
        runner_state[0][0].params,
        os.path.join(
            save_path,
            f"{config.SCENARIO}_seed{config.SEED}_actor.safetensors",
        ),
    )
