from typing import Dict, Any, Optional, Callable

import chex
import jax
import jax.numpy as jnp
import numpy as np
import optax
import tensorflow_probability.substrates.jax as tfp
from flax import nnx, struct

from src.baseline.algorithm.base_algo import BaseAlgo
from src.baseline.configs.config import PQNConfig
from src.baseline.modules import PQN_Critic
from src.baseline.utils import (
    NetworkState,
    TrainState,
    get_model,
)
from src.tabs import TABSUnitComb, TABSUnitDeploy
from src.tabs.scenarios import Scenario

tfd = tfp.distributions
tfb = tfp.bijectors


@struct.dataclass
class TrainState:
    critic_state: NetworkState
    key: jax.random.PRNGKey
    timesteps: int = jnp.array(0)
    num_updates: int = jnp.array(0)


class PQN(BaseAlgo):
    def __init__(self, config: PQNConfig, env: TABSUnitComb | TABSUnitDeploy):
        super(PQN, self).__init__(config, env)
        self.v_reset = jax.vmap(env.reset, in_axes=(0, 0))
        self.v_step = jax.vmap(env.step, in_axes=(0, 0, 0))

        self.observation_dim = sum(self.env.observation_space.shape)
        self.action_dim = self.env.action_space.n

    def get_epsilon(self, train_state: TrainState):
        I = self.config.eps_start
        E = self.config.eps_finish
        T = (
            train_state.num_updates
            * self.config.n_env
            * self.config.rollout_step
            * self.config.eps_decay
        )
        t = train_state.timesteps

        return jnp.where(t < T, I + (E - I) * (t) / T, E)

    def init_train_state(
        self, key: jax.random.PRNGKey, num_updates: Optional[int] = None
    ) -> TrainState:
        if self.config.learning_scheduler and num_updates is None:
            raise ValueError("num_updates must be provided if learning_scheduler is True")
        rngs = nnx.Rngs(self.config.seed)

        # Instantiate policy and critic functions
        critic = PQN_Critic(
            action_dim=self.action_dim,
            state_dim=self.observation_dim,
            layer_dim=self.config.layer_dim,
            rngs=rngs,
        )
        if self.config.learning_scheduler:
            learning_rate = optax.linear_schedule(
                init_value=self.config.lr,
                end_value=0.0,
                transition_steps=num_updates
                * ((self.config.n_env // self.config.batch_size) * self.config.num_epochs),
            )
        else:
            learning_rate = self.config.lr

        # Optimizer
        critic_optimizer = nnx.Optimizer(
            critic,
            optax.chain(
                optax.clip_by_global_norm(self.config.max_grad_norm),
                optax.radam(learning_rate=learning_rate),
            ),
        )
        (critic_gd, critic_state) = nnx.split((critic, critic_optimizer))

        return TrainState(
            critic_state=NetworkState(critic_gd, critic_state),
            key=key,
            timesteps=jnp.array(0),
            num_updates=jnp.array(num_updates),
        )

    def sample_action(
        self,
        train_state: TrainState,
        obs: chex.Array,
        unavail_action: chex.Array,
        key: jax.random.PRNGKey,
    ) -> Dict[str, Any]:
        critic, _ = get_model(train_state.critic_state)
        q_values = critic(obs)
        q_values = jnp.where(unavail_action, -1e9, q_values)
        actions = jnp.argmax(q_values, axis=-1)
        random_logits = jnp.where(unavail_action, -jnp.inf, jnp.zeros_like(q_values))
        eps = self.get_epsilon(train_state)

        dist = tfd.Categorical(logits=random_logits)
        random_actions = dist.sample(seed=key)

        is_random = jax.random.uniform(key, random_actions.shape) < eps
        actions = jnp.where(is_random, random_actions, actions)
        return {"actions": actions, "q_values": q_values}

    def greedy_action(
        self, train_state: TrainState, obs: chex.Array, unavail_action: chex.Array
    ) -> Dict[str, Any]:
        critic, _ = get_model(train_state.critic_state)
        critic.eval()
        q_values = critic(obs)
        q_values = jnp.where(unavail_action, -1e9, q_values)
        actions = jnp.argmax(q_values, axis=-1)
        return {"actions": actions, "q_values": q_values}

    def rollout(self, train_state: TrainState, scenario: Scenario, greedy: bool = False):
        critic, _ = get_model(train_state.critic_state)
        critic.eval()
        reset_key, key = jax.random.split(train_state.key)
        init_obs, env_state = self.v_reset(jax.random.split(reset_key, self.config.n_env), scenario)

        def rollout_body(carry, _):
            obs, env_state, key = carry
            action_key, step_key, key = jax.random.split(key, 3)

            if greedy:
                sample_result = self.greedy_action(train_state, obs, env_state.unavail_action)
            else:
                sample_result = self.sample_action(
                    train_state, obs, env_state.unavail_action, action_key
                )
            next_obs, next_env_state, _, done, _ = self.v_step(
                jax.random.split(step_key, self.config.n_env), env_state, sample_result["actions"]
            )

            transition = {
                "observations": obs,
                "actions": sample_result["actions"],
                "reward": jnp.zeros_like(done, dtype=jnp.float32),
                "next_obs": next_obs,
                "dones": done,
                "q_val": sample_result["q_values"],
            }

            return (next_obs, next_env_state, key), transition

        initial_carry = (init_obs, env_state, key)
        (last_obs, last_state, key), transitions = jax.lax.scan(
            rollout_body, initial_carry, None, self.config.rollout_step
        )

        rollout_result = {
            "last_state": last_state,
            "last_obs": last_obs,
        }
        rollout_result.update(transitions)

        # Update timesteps
        return train_state.replace(
            key=key, timesteps=train_state.timesteps + self.config.rollout_step * self.config.n_env
        ), rollout_result

    def train(self, train_state, rollout_result):
        critic, _ = get_model(train_state.critic_state)
        critic.eval()
        transitions = {
            "observations": rollout_result["observations"],
            "actions": rollout_result["actions"],
            "rewards": rollout_result["rewards"] * self.config.reward_scale,
            "next_obs": rollout_result["next_obs"],
            "dones": rollout_result["dones"],
            "q_val": rollout_result["q_val"],
        }
        last_q = jnp.max(
            critic(rollout_result["last_obs"]) - rollout_result["last_state"].unavail_action * 1e9,
            axis=-1,
            keepdims=True,
        )

        def _get_target(lambda_returns_and_next_q, transition):
            lambda_returns, next_q = lambda_returns_and_next_q
            target_bootstrap = (
                transition["rewards"] + self.config.gamma * (1 - transition["dones"]) * next_q
            )
            delta = lambda_returns - next_q
            lambda_returns = target_bootstrap + self.config.gamma * self.config.lamda * delta
            lambda_returns = (1 - transition["dones"]) * lambda_returns + transition[
                "dones"
            ] * transition["rewards"]
            next_q = jnp.max(transition["q_val"], axis=-1, keepdims=True)
            return (lambda_returns, next_q), lambda_returns

        last_q = last_q * (1 - transitions["dones"][-1])
        lambda_returns = transitions["rewards"][-1] + self.config.gamma * last_q
        _, targets = jax.lax.scan(
            _get_target,
            (lambda_returns, last_q),
            jax.tree_util.tree_map(lambda x: x[:-1], transitions),
            reverse=True,
        )
        lambda_targets = jnp.concatenate((targets, lambda_returns[np.newaxis]))

        def batch_scan_body(train_state, _):
            batch_key, key = jax.random.split(train_state.key)

            batch_idx = jax.random.permutation(batch_key, self.config.n_env).reshape(
                -1, self.config.batch_size
            )
            batch = jax.vmap(
                lambda batch_idx: jax.tree.map(
                    lambda x: x[:, batch_idx], (transitions, lambda_targets)
                )
            )(batch_idx)

            def train_body(train_state, batch):
                transitions, targets = batch

                critic, critic_optimizer = get_model(train_state.critic_state)
                critic.train()

                def critic_loss_fn(critic: PQN_Critic):
                    q_vals = critic(transitions["observations"])
                    chosen_action_qvals = jnp.take_along_axis(
                        q_vals, jnp.expand_dims(transitions["actions"], axis=-1), axis=-1
                    )
                    target_transitions = transitions["dones"].cumsum(axis=0) <= 1
                    loss = (
                        0.5
                        * (jnp.square(targets - chosen_action_qvals) * target_transitions).sum()
                        / target_transitions.sum()
                    )
                    return loss

                critic_loss, critic_grads = nnx.value_and_grad(critic_loss_fn)(critic)
                critic_optimizer.update(critic_grads)

                train_state = train_state.replace(
                    critic_state=train_state.critic_state.replace(
                        state=nnx.state((critic, critic_optimizer))
                    ),
                    key=key,
                )
                return train_state, critic_loss

            train_state, critic_loss = jax.lax.scan(train_body, train_state, batch)
            return train_state, critic_loss.mean()

        train_state, critic_loss = jax.lax.scan(
            batch_scan_body, train_state, None, self.config.num_epochs
        )
        q_vals = transitions["q_val"]
        return train_state, {
            "critic_loss": critic_loss.mean(),
            "q_val_max": q_vals.max(),
            "eps": self.get_epsilon(train_state),
        }
