from typing import Tuple, Dict, Any, Optional

import chex
import jax
import optax
import jax.numpy as jnp
from flax import nnx
import tensorflow_probability.substrates.jax as tfp

from src.baseline.algorithm.base_algo import BaseAlgo
from src.baseline.modules import RNNActorCritic
from src.baseline.utils import NetworkState, TrainState, get_model, rnn_result, get_gae
from src.tabs.scenarios import Scenario
from src.tabs import TABSBattleSimulator


tfd = tfp.distributions
tfb = tfp.bijectors


class IPPO(BaseAlgo):
    def __init__(self, config, env: TABSBattleSimulator):
        super(IPPO, self).__init__(config, env)
        # self.v_reset = jax.vmap(lambda key: env.reset(key, None))
        self.v_reset = jax.vmap(env.reset, in_axes=(0, 0))
        self.v_step = jax.vmap(env.step, in_axes=(0, 0, 0))

        self.observation_dim = sum(self.env.observation_spaces["unit_0"].shape)
        self.action_dim = self.env.action_space.discrete.n
        self.state_dim = self.observation_dim * self.env.max_n_ally

    def init_train_state(
        self, key: jax.random.PRNGKey, num_updates: Optional[int] = None
    ) -> TrainState:
        if self.config.learning_scheduler and num_updates is None:
            raise ValueError("num_updates must be provided if learning_scheduler is True")
        rngs = nnx.Rngs(self.config.seed)

        # Instantiate actor-critic model
        model = RNNActorCritic(
            obs_dim=self.observation_dim,
            action_dim=self.action_dim,
            layer_dim=self.config.layer_dim,
            rngs=rngs,
        )

        if self.config.learning_scheduler:
            learning_rate = optax.linear_schedule(
                init_value=self.config.lr,
                end_value=0.0,
                transition_steps=num_updates
                * self.config.epochs
                * (self.config.n_env // self.config.batch_size),
            )
        else:
            learning_rate = self.config.lr

        # Optimizer
        optimizer = nnx.Optimizer(
            model,
            optax.chain(
                optax.clip_by_global_norm(self.config.max_grad_norm),
                optax.adam(learning_rate=learning_rate, eps=1e-5),
            ),
        )

        (gd, network_state) = nnx.split((model, optimizer))

        return TrainState(
            policy_state=NetworkState(gd, network_state),
            critic_state=None,
            key=key,
        )

    def sample_action(
        self,
        train_state: TrainState,
        obs: chex.Array,
        hidden_state: chex.Array,
        key: jax.random.PRNGKey,
    ) -> Dict[str, Any]:
        model, _ = get_model(train_state.policy_state)
        next_hidden_state, logits, mean, log_std, value = model(hidden_state, obs)

        # Get hybrid action
        discrete_key, continuous_key = jax.random.split(key)
        discrete_distribution = tfd.Categorical(logits=logits)
        continuous_distribution = tfd.Normal(mean, jnp.exp(log_std))
        discrete_actions = discrete_distribution.sample(seed=discrete_key)[:, None]
        continuous_actions = continuous_distribution.sample(seed=continuous_key)

        actions = jnp.concatenate([continuous_actions, discrete_actions], axis=-1)

        discrete_log_probs = discrete_distribution.log_prob(
            discrete_actions[:, 0].astype(jnp.int32)
        )[:, None]
        continuous_log_probs = continuous_distribution.log_prob(continuous_actions)
        log_probs = discrete_log_probs + continuous_log_probs
        result = {
            "actions": actions,
            "log_probs": log_probs,
            "next_hidden_state": next_hidden_state,
            "discrete_log_probs": discrete_log_probs,
            "continuous_log_probs": continuous_log_probs,
            "discrete_actions": discrete_actions,
            "continuous_actions": continuous_actions,
            "value": value,
        }

        return result

    def rollout(
        self, train_state: TrainState, scenario: Scenario
    ) -> Tuple[TrainState, Dict[str, Any]]:
        model, _ = get_model(train_state.policy_state)

        reset_key, key = jax.random.split(train_state.key)
        init_obs, env_state = self.v_reset(jax.random.split(reset_key, self.config.n_env), scenario)

        # Hidden states for RNN
        hidden_state = model.initialize_carry((self.env.max_n_ally,) + init_obs["unit_0"].shape)

        def rollout_body(carry, _):
            obs, env_state, hidden_state, key = carry
            stacked_obs = jnp.stack([obs[i] for i in self.env.ally_keys], axis=1)

            action_key, step_key, key = jax.random.split(key, 3)
            sample_result = jax.vmap(self.sample_action, in_axes=(None, 1, 0, 0))(
                train_state,
                stacked_obs,
                hidden_state,
                jax.random.split(action_key, self.env.max_n_ally),
            )

            actions = {}
            for i, agent in enumerate(self.env.ally_keys):
                actions[agent] = sample_result["actions"][i]

            next_obs, env_state, rewards, dones, infos = self.v_step(
                jax.random.split(step_key, self.config.n_env), env_state, actions
            )
            hidden_state = sample_result["next_hidden_state"] * ~dones["__all__"][None, :]

            sample_result.update(
                {
                    "state_value": sample_result["value"],
                    "dones": dones,
                    "observations": obs,
                    "rewards": rewards,
                    "infos": infos,
                    "env_state": env_state,
                }
            )

            return (
                next_obs,
                env_state,
                hidden_state,
                key,
            ), sample_result

        initial_carry = (init_obs, env_state, hidden_state, key)
        (last_obs, last_env_state, _, key), rollout_result = jax.lax.scan(
            rollout_body, initial_carry, None, self.config.rollout_step
        )

        last_obs = jnp.stack([last_obs[i] for i in self.env.ally_keys], axis=1)
        last_obs = jnp.transpose(last_obs, axes=(1, 0, 2))
        _, _, _, _, last_value = model(hidden_state, last_obs)
        common_reward = rollout_result["rewards"][:, :, 0]

        rollout_result["common_reward"] = common_reward
        rollout_result["dones"] = rollout_result["dones"]["__all__"]
        rollout_result["last_value"] = last_value
        rollout_result["last_state"] = last_env_state

        rollout_result["returned_episode_returns"] = last_env_state.returned_episode_returns
        rollout_result["returned_episode_lengths"] = last_env_state.returned_episode_lengths
        rollout_result["returned_episode_wins"] = last_env_state.returned_episode_wins

        return train_state.replace(key=key), rollout_result

    def train_step(
        self, train_state: TrainState, batch: Dict[str, Any]
    ) -> Tuple[TrainState, Dict[str, Any]]:
        def loss_fn(model: RNNActorCritic):
            stacked_obs = jnp.stack(
                [batch["observations"][key] for key in self.env.ally_keys], axis=0
            )
            m_vmap = jax.vmap(
                lambda feature, done: rnn_result(model, (self.config.layer_dim,), feature, done),
                in_axes=(1, 1),
                out_axes=1,
            )
            logits, mean, log_std, values = jax.vmap(
                lambda obs: m_vmap(obs, batch["dones"]), out_axes=1
            )(stacked_obs)

            # Calculate actor loss
            continuous_dist, discrete_dist = model.get_distribution(logits, mean, log_std)
            discrete_action = batch["discrete_actions"]
            discrete_log_pi = discrete_dist.log_prob(discrete_action[:, :, :, 0])[:, :, :, None]

            continuous_action = batch["continuous_actions"]
            continuous_log_pi = continuous_dist.log_prob(continuous_action)

            log_pi = discrete_log_pi + continuous_log_pi

            log_ratio = log_pi - batch["log_probs"]
            ratio = jnp.exp(log_ratio)
            actor_loss = jnp.minimum(
                ratio * batch["advantages"],
                jnp.clip(ratio, 1 - self.config.clip_ratio, 1 + self.config.clip_ratio)
                * batch["advantages"],
            ).mean()

            discrete_entropy = discrete_dist.entropy().mean()
            continuous_entropy = -continuous_log_pi.mean()
            entropy = discrete_entropy + continuous_entropy

            # Calculate critic loss
            batch_values = batch["state_value"]
            clip_value = batch_values + jnp.clip(
                values - batch_values, -self.config.clip_value, self.config.clip_value
            )
            critic_losses = jnp.square(batch["returns"] - values)
            clip_losses = jnp.square(batch["returns"] - clip_value)
            critic_loss = 0.5 * jnp.maximum(critic_losses, clip_losses).mean()

            total_loss = (
                -actor_loss
                - self.config.entropy_coef * entropy
                + self.config.critic_coef * critic_loss
            )

            approx_kl = ((ratio - 1) - log_ratio).mean()

            return total_loss, {
                "policy_loss": actor_loss,
                "critic_loss": critic_loss,
                "entropy": entropy,
                "ratio": ratio,
                "ratio_max": ratio.max(),
                "ratio_min": ratio.min(),
                "ratio_mean": ratio.mean(),
                "advantage_max": batch["advantages"].max(),
                "advantage_min": batch["advantages"].min(),
                "advantage_mean": batch["advantages"].mean(),
                "approx_kl": approx_kl,
                "discrete_entropy": discrete_entropy,
                "continuous_entropy": continuous_entropy,
            }

        model, optimizer = get_model(train_state.policy_state)
        (loss, info), grads = nnx.value_and_grad(loss_fn, has_aux=True)(model)

        optimizer.update(grads)

        train_state = train_state.replace(
            policy_state=train_state.policy_state.replace(state=nnx.state((model, optimizer)))
        )

        return train_state, info

    def train(
        self, train_state: TrainState, batch: Dict[str, Any]
    ) -> Tuple[TrainState, Dict[str, Any]]:
        # Compute GAE and returns
        advantages, returns = jax.vmap(
            lambda state_value, last_value: jax.vmap(
                get_gae, in_axes=(1, 1, 1, 0, None, None), out_axes=1
            )(
                batch["common_reward"],
                batch["dones"],
                state_value,
                last_value,
                self.config.gamma,
                self.config.lamda,
            ),
            in_axes=(1, 0),
            out_axes=1,
        )(batch["state_value"], batch["last_value"])

        # Normalization (is there need to normalize each batch?)
        # advantages = (advantages - advantages.mean(axis=1, keepdims=True)) / (
        #     advantages.std(axis=1, keepdims=True) + 1e-8
        # )

        batch.update({"advantages": advantages, "returns": returns})
        batch1 = {
            "advantages": batch["advantages"],
            "continuous_actions": batch["continuous_actions"],
            "discrete_actions": batch["discrete_actions"],
            "log_probs": batch["log_probs"],
            "returns": batch["returns"],
        }

        batch2 = {
            "dones": batch["dones"],
            "observations": batch["observations"],
            "state_value": batch["state_value"],
        }

        def train_body(carry, _):
            (train_state,) = carry
            batch_key, key = jax.random.split(train_state.key)
            batch_idx = jax.random.permutation(batch_key, self.config.n_env).reshape(
                -1, self.config.batch_size
            )

            def batch_scan_body(carry, batch_idx):
                (train_state,) = carry
                batch1_ = jax.vmap(
                    lambda idx: jax.tree.map(lambda x: x[:, :, idx], batch1), out_axes=2
                )(batch_idx)
                batch2_ = jax.vmap(
                    lambda idx: jax.tree.map(lambda x: x[:, idx, :], batch2), out_axes=1
                )(batch_idx)

                batch1_.update(batch2_)
                batch1_["advantages"] = (batch1_["advantages"] - batch1_["advantages"].mean()) / (
                    batch1_["advantages"].std() + 1e-8
                )
                train_state, train_result = self.train_step(train_state, batch1_)
                return (train_state.replace(key=key),), train_result

            (train_state,), train_result = jax.lax.scan(batch_scan_body, (train_state,), batch_idx)

            return (train_state.replace(key=key),), train_result

        (train_state,), train_result = jax.lax.scan(
            train_body, (train_state,), None, self.config.epochs
        )

        info = {
            "critic_loss": train_result["critic_loss"].mean(),
            "policy_loss": train_result["policy_loss"].mean(),
            "entropy": train_result["entropy"].mean(),
            "ratio": train_result["ratio"].mean(),
            "ratio_max": train_result["ratio_max"].max(),
            "ratio_min": train_result["ratio_min"].min(),
            "ratio_mean": train_result["ratio_mean"].mean(),
            "advantage_max": train_result["advantage_max"].max(),
            "advantage_min": train_result["advantage_min"].min(),
            "advantage_mean": train_result["advantage_mean"].mean(),
            "rewards": batch["common_reward"].sum() / self.config.n_env,
            "approx_kl_max": train_result["approx_kl"].max(),
            "approx_kl_min": train_result["approx_kl"].min(),
            "approx_kl_mean": train_result["approx_kl"].mean(),
            "returned_episode_returns": batch["returned_episode_returns"],
            "returned_episode_lengths": batch["returned_episode_lengths"],
            "returned_episode_wins": batch["returned_episode_wins"],
            "discrete_entropy": train_result["discrete_entropy"].mean(),
            "continuous_entropy": train_result["continuous_entropy"].mean(),
        }

        return train_state, info
