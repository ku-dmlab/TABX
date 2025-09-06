from typing import Tuple, Dict, Any
import deepcopy

import chex
import jax
import optax
import jax.numpy as jnp
import flashbax as fbx
from flashbax.buffers.trajectory_buffer import TrajectoryBuffer, BufferState
from flax import struct, nnx
import tensorflow_probability.substrates.jax as tfp

from src.baseline.base_algo import BaseAlgo
from src.baseline.module.modules import QNetwork
from src.baseline.utils import NetworkState, TrainState, get_model
from src.maenv.tabs.scenarios import Scenario
from src.maenv.tabs import TABSUnitComb, TABSUnitDeploy


tfd = tfp.distributions
tfb = tfp.bijectors


@chex.dataclass(frozen=True)
class TimeStep:
    obs: chex.Array
    action: chex.Array
    reward: chex.Array
    done: chex.Array
    next_obs: chex.Array


@struct.dataclass
class DQNTrainState(TrainState):
    qnet_state: NetworkState
    qnet_target_param: nnx.Param
    buffer_state: BufferState


class DQN(BaseAlgo):
    def __init__(self, config, env: TABSUnitComb | TABSUnitDeploy):
        super(DQN, self).__init__(config, env)
        self.eps = self.config.eps if hasattr(self.config, "eps") else 0.1
        self.buffer_size = self.config.buffer_size if hasattr(self.confg, "buffer_size") else 100
        self.buffer_batch_size = (
            self.config.buffer_batch_size if hasattr(self.config, "buffer_batch_size") else 32
        )

        buffer: TrajectoryBuffer = fbx.make_flat_buffer(
            max_length=self.buffer_size,
            min_length=self.buffer_batch_size,
            sample_batch_size=self.buffer_batch_size,
            add_sequences=False,
            add_batch_size=self.config.n_env,
        )
        self.buffer = buffer.replace(
            init=jax.jit(buffer.init),
            add=jax.jit(buffer.add, donate_argnums=0),
            sample=jax.jit(buffer.sample),
            can_sample=jax.jit(buffer.can_sample),
        )

        self.v_reset = jax.vmap(env.reset, in_axes=(0, 0))
        self.v_step = jax.vmap(env.step, in_axes=(0, 0, 0))

        self.observation_dim = sum(self.env.observation_space.shape)
        self.action_dim = self.env.action_space.n

    def init_train_state(self, key: jax.random.PRNGKey, scenario: Scenario) -> DQNTrainState:
        rngs = nnx.Rngs(self.config.seed)

        # Instantiate value function
        qnet = QNetwork(
            self.observation_dim, self.action_dim, layer_dim=self.config.dqn_layer_dim, rngs=rngs
        )

        # Optimizer
        optimizer = nnx.Optimizer(
            qnet,
            optax.chain(
                optax.clip_by_global_norm(self.config.max_grad_norm),
                optax.adam(learning_rate=self.config.lr),
            ),
        )

        (gd, qnet_state) = nnx.split((qnet, optimizer))
        _, qnet_target_param, _ = nnx.split(qnet, nnx.Param, ...)

        # Init replay buffer
        key_reset, key_step, key_action, key = jax.random.split(key, 4)
        _, _env_state = self.env.reset(key_reset, scenario)
        _action = self.env.action_space.sample(key_action)
        _obs, _, _, _done, _ = self.env.step(key_step, _env_state, _action)
        _timestep = TimeStep(obs=_obs, action=_action, reward=0, done=_done, next_obs=_obs)
        buffer_state = self.buffer.init(_timestep)

        return DQNTrainState(
            qnet_state=NetworkState(gd, qnet_state),
            qnet_target_param=deepcopy.copy(qnet_target_param),
            key=key,
            buffer_state=buffer_state,
        )

    def sample_action(
        self,
        train_state: DQNTrainState,
        obs: chex.Array,
        unavail_action: chex.Array,
        key: jax.random.PRNGKey,
    ) -> Dict[str, Any]:
        key_a, key_eps = jax.random.split(key)

        qnet, _ = get_model(train_state.critic_state)
        q_vals = jnp.where(unavail_action, -jnp.inf, qnet(obs))
        greedy_actions = jnp.argmax(q_vals, axis=-1)
        chosed_actions = jnp.where(
            jax.random.uniform(key_eps, greedy_actions.shape)
            < self.eps,  # Pick the actions that should be random
            jax.random.randint(
                key_a, shape=greedy_actions.shape, minval=0, maxval=q_vals.shape[-1]
            ),  # Sample random actions
            greedy_actions,
        )

        return {"actions": chosed_actions}

    def rollout(
        self, train_state: DQNTrainState, scenario: Scenario
    ) -> Tuple[DQNTrainState, Dict[str, Any]]:
        # Reset the env
        reset_key, key = jax.random.split(train_state.key)
        init_obs, env_state = self.v_reset(jax.random.split(reset_key, self.config.n_env), scenario)

        def rollout_body(carry, _):
            obs, env_state, train_state, key = carry
            # Step the env
            action_key, step_key, key = jax.random.split(key, 3)
            sample_result = self.sample_action(
                train_state, obs, env_state.unavail_action, action_key
            )
            next_obs, env_state, _, done, _ = self.v_step(
                jax.random.split(step_key, self.config.n_env), env_state, sample_result["actions"]
            )

            sample_result.update(
                {
                    "dones": done,
                    "obs": obs,
                    "unavail_actions": env_state.unavail_action,
                    "env_state": env_state,
                    "actions": sample_result["actions"],
                    "next_obs": next_obs,
                }
            )

            return (next_obs, env_state, key), sample_result

        initial_carry = (init_obs, env_state, key)
        (_, last_state, key), rollout_result = jax.lax.scan(
            rollout_body, initial_carry, None, self.config.rollout_step
        )

        rollout_result["last_state"] = last_state

        return train_state.replace(key=key), rollout_result

    def train_step(self, train_state: DQNTrainState, batch):
        gd, other_variables = train_state.qnet_state.split(nnx.Param, ...)
        qnet_target, _ = nnx.merge(gd, train_state.qnet_target_param, other_variables)
        q_next_target = qnet_target(batch.second.obs)
        q_next_target = jnp.max(q_next_target, axis=-1)
        target = batch.first.reward + (1 - batch.first.done) * self.config.gamma * q_next_target

        def loss_fn(qnet: QNetwork):
            q_vals = qnet(batch.first.obs)
            chosen_action_qvals = jnp.take_along_axis(
                q_vals, jnp.expand_dims(batch.first.action, axis=-1), axis=-1
            ).squeeze(axis=-1)

            loss = jnp.mean((chosen_action_qvals - target) ** 2)

            return loss, {"loss": loss, "chosen_action_qval": chosen_action_qvals.mean()}

        qnet, optimizer = get_model(train_state.qnet_state)

        (loss, info), grads = nnx.value_and_grad(loss_fn, has_aux=True)(qnet)
        optimizer.update(grads)

        train_state = train_state.replace(
            qnet_state=train_state.qnet_state.replace(state=nnx.state((qnet, optimizer)))
        )

        return train_state, info

    def train(self, train_state: DQNTrainState) -> Tuple[DQNTrainState, Dict[str, Any]]:
        key_buf, key = jax.random.split(train_state.key)
        train_state = train_state.replace(key=key)

        # Sample batch from the replay buffer
        batch = self.buffer.sample(train_state.buffer_state, key_buf).experience

        # Update network
        train_state, train_result = self.train_step(train_state, batch)

        info = {
            "loss": train_result["loss"].mean(),
            "chosen_action_qval": train_result["chosen_action_qval"].mean(),
        }

        return train_state, info

    def update_target(self, train_state: DQNTrainState, step: int) -> DQNTrainState:
        train_state = jax.lax.cond(
            step % self.config.target_update_interval == 0,
            lambda train_state: train_state.replace(
                qnet_target_param=optax.incremental_update(
                    train_state.qnet_state.filter(nnx.Param),
                    train_state.qnet_target_param,
                    self.config.tau,
                )
            ),
            lambda train_state: train_state,
            operand=train_state,
        )

        return train_state
