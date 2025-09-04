from typing import Tuple, Dict, Any

import chex
import jax
import optax
import jax.numpy as jnp
from flax import nnx
import tensorflow_probability.substrates.jax as tfp

from src.baseline.base_algo import BaseAlgo
from src.baseline.module.modules import Policy, Critic
from src.baseline.utils import NetworkState, TrainState, get_model, get_gae
from src.maenv.tabs.scenarios import Scenario
from src.maenv.tabs.tabs_unit_deploy.tabs_unit_deploy import TABSUnitDeploy
from src.maenv.tabs.tabs_unit_comb.tabs_unit_comb import TABSUnitComb


tfd = tfp.distributions
tfb = tfp.bijectors


class PPO(BaseAlgo):
    def __init__(self, config, env: TABSUnitComb | TABSUnitDeploy):
        super(PPO, self).__init__(config, env)
        self.v_reset = jax.vmap(env.reset, in_axes=(0, 0))
        self.v_step = jax.vmap(env.step, in_axes=(0, 0, 0))

        self.observation_dim = sum(self.env.observation_space.shape)
        self.action_dim = self.env.action_space.n

    def init_train_state(self, key: jax.random.PRNGKey) -> TrainState:
        rngs = nnx.Rngs(self.config.seed)

        # Instantiate policy and critic functions
        pi = Policy(
            action_dim=self.action_dim,
            state_dim=self.observation_dim,
            layer_dim=self.config.ppo_layer_dim,
            rngs=rngs,
        )
        critic = Critic(self.observation_dim, layer_dim=self.config.ppo_layer_dim, rngs=rngs)

        # Optimizer
        pi_optimizer = nnx.Optimizer(
            pi,
            optax.chain(
                optax.clip_by_global_norm(self.config.max_grad_norm),
                optax.adam(learning_rate=self.config.lr),
            ),
        )
        critic_optimizer = nnx.Optimizer(
            critic,
            optax.chain(
                optax.clip_by_global_norm(self.config.max_grad_norm),
                optax.adam(learning_rate=self.config.lr),
            ),
        )

        (pi_gd, policy_state) = nnx.split((pi, pi_optimizer))
        (critic_gd, critic_state) = nnx.split((critic, critic_optimizer))

        return TrainState(
            policy_state=NetworkState(pi_gd, policy_state),
            critic_state=NetworkState(critic_gd, critic_state),
            key=key,
        )

    def sample_action(
        self,
        train_state: TrainState,
        obs: chex.Array,
        unavail_action: chex.Array,
        key: jax.random.PRNGKey,
    ) -> Dict[str, Any]:
        policy, _ = get_model(train_state.policy_state)
        logits = policy(obs)
        logits = jnp.where(unavail_action, -jnp.inf, logits)
        dist = tfd.Categorical(logits=logits)
        actions = dist.sample(seed=key)
        log_probs = dist.log_prob(actions)

        result = {
            "actions": actions,
            "log_probs": log_probs,
        }

        return result

    def rollout(
        self, train_state: TrainState, scenario: Scenario
    ) -> Tuple[TrainState, Dict[str, Any]]:
        policy, _ = get_model(train_state.policy_state)
        critic, _ = get_model(train_state.critic_state)

        reset_key, key = jax.random.split(train_state.key)
        init_obs, env_state = self.v_reset(jax.random.split(reset_key, self.config.n_env), scenario)

        def rollout_body(carry, _):
            obs, env_state, key = carry
            action_key, step_key, key = jax.random.split(key, 3)
            sample_result = self.sample_action(policy, obs, env_state.unavail_action, action_key)
            next_obs, env_state, _, done, _ = self.v_step(
                jax.random.split(step_key, self.config.n_env), env_state, sample_result["actions"]
            )
            values = critic(obs)

            sample_result.update(
                {
                    "values": values,
                    "dones": done,
                    "observations": obs,
                    "unavail_actions": env_state.unavail_action,
                    "env_state": env_state,
                }
            )

            return (next_obs, env_state, key), sample_result

        initial_carry = (init_obs, env_state, key)
        (last_obs, last_state, key), rollout_result = jax.lax.scan(
            rollout_body, initial_carry, None, self.config.rollout_step
        )

        last_value = critic(last_obs)
        rollout_result["last_value"] = last_value
        rollout_result["last_state"] = last_state

        return train_state.replace(key=key), rollout_result

    def train_step(
        self, train_state: TrainState, batch: Dict[str, Any]
    ) -> Tuple[TrainState, Dict[str, Any]]:
        def critic_loss_fn(critic: Critic):
            state_values = critic(batch["observations"])
            batch_values = batch["values"]
            clip_value = batch_values + jnp.clip(
                state_values - batch_values, -self.config.clip_value, self.config.clip_value
            )
            critic_losses = jnp.square(batch["returns"] - state_values)
            clip_losses = jnp.square(batch["returns"] - clip_value)
            critic_loss = 0.5 * jnp.maximum(critic_losses, clip_losses).mean()

            return critic_loss

        def policy_loss_fn(policy: Policy):
            logits = policy(batch["observations"])
            logits = jnp.where(batch["unavail_actions"], -jnp.inf, logits)
            dist = tfd.Categorical(logits=logits)
            log_pi = dist.log_prob(batch["actions"])

            log_diff = log_pi - batch["log_probs"]
            isnan_log_diff = jnp.isnan(log_diff)
            log_diff = jnp.where(isnan_log_diff, 0.0, log_diff)

            ratio = jnp.exp(log_diff).reshape(batch["advantages"].shape)
            loss = (
                jnp.minimum(
                    ratio * batch["advantages"],
                    jnp.clip(ratio, 1 - self.config.clip_ratio, 1 + self.config.clip_ratio)
                    * batch["advantages"],
                )
                * (1 - isnan_log_diff.reshape(batch["advantages"].shape))
            ).sum() / (~isnan_log_diff).sum()

            is_nan_log_pi = jnp.isnan(log_pi)
            entropy = (
                jnp.where(is_nan_log_pi, 0.0, -log_pi * jnp.exp(log_pi)).sum()
                / (~is_nan_log_pi).sum()
            )

            return -loss + self.config.entropy_coef * entropy, {
                "policy_loss": loss,
                "entropy": entropy,
                "ratio": ratio,
                "ratio_max": ratio.max(),
                "ratio_min": ratio.min(),
                "ratio_mean": ratio.mean(),
                "isnan_log_diff": isnan_log_diff.mean(),
                "advantage_max": batch["advantages"].max(),
                "advantage_min": batch["advantages"].min(),
                "advantage_mean": batch["advantages"].mean(),
            }

        critic, critic_optimizer = get_model(train_state.critic_state)
        policy, policy_optimizer = get_model(train_state.policy_state)

        critic_loss, critic_grads = nnx.value_and_grad(critic_loss_fn)(critic)
        (policy_loss, info), policy_grads = nnx.value_and_grad(policy_loss_fn, has_aux=True)(policy)

        critic_optimizer.update(critic_grads)
        policy_optimizer.update(policy_grads)

        train_state = train_state.replace(
            policy_state=train_state.policy_state.replace(
                state=nnx.state((policy, policy_optimizer))
            ),
            critic_state=train_state.critic_state.replace(
                state=nnx.state((critic, critic_optimizer))
            ),
        )

        info.update({"critic_loss": critic_loss})

        return train_state, info

    def train(
        self, train_state: TrainState, batch: Dict[str, Any]
    ) -> Tuple[TrainState, Dict[str, Any]]:
        # Compute GAE and returns
        advantages, returns = jax.vmap(get_gae, in_axes=(1, 1, 1, 0, None, None), out_axes=1)(
            batch["rewards"],
            batch["dones"],
            batch["values"],
            batch["last_value"],
            self.config.gamma,
            self.config.lamda,
        )

        # Normalization
        advantages = (advantages - advantages.mean(axis=1, keepdims=True)) / (
            advantages.std(axis=1, keepdims=True) + 1e-8
        )

        batch.update({"advantages": advantages, "returns": returns})

        # TODO : consider batch size
        def train_body(carry, _):
            (train_state,) = carry

            train_state, train_result = self.train_step(train_state, batch)

            return (train_state,), train_result

        (train_state,), train_result = jax.lax.scan(
            train_body, (train_state,), None, self.config.ppo_epochs
        )

        info = {
            "critic_loss": train_result["critic_loss"].mean(),
            "policy_loss": train_result["policy_loss"].mean(),
            "entropy": train_result["entropy"].mean(),
            "ratio": train_result["ratio"].mean(),
            "ratio_max": train_result["ratio_max"].max(),
            "ratio_min": train_result["ratio_min"].min(),
            "ratio_mean": train_result["ratio_mean"].mean(),
            "isnan_log_diff": train_result["isnan_log_diff"].mean(),
            "advantage_max": train_result["advantage_max"].max(),
            "advantage_min": train_result["advantage_min"].min(),
            "advantage_mean": train_result["advantage_mean"].mean(),
            "rewards": batch["rewards"].sum() / self.config.n_env,
        }

        return train_state, info
