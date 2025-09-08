from typing import Tuple, Dict, Any

import chex
import jax
import optax
import jax.numpy as jnp
from flax import nnx
import tensorflow_probability.substrates.jax as tfp

from src.baseline.algorithm.base_algo import BaseAlgo
from src.baseline.modules import RNNHybridPolicy, RNNCritic
from src.baseline.utils import NetworkState, TrainState, get_model, rnn_result, get_gae
from src.tabs.scenarios import Scenario
from src.tabs import TABSBattleSimulator


tfd = tfp.distributions
tfb = tfp.bijectors


class MAPPO(BaseAlgo):
    def __init__(self, config, env: TABSBattleSimulator):
        super(MAPPO, self).__init__(config, env)
        # self.v_reset = jax.vmap(lambda key: env.reset(key, None))
        self.v_reset = jax.vmap(env.reset, in_axes=(0, 0))
        self.v_step = jax.vmap(env.step, in_axes=(0, 0, 0))

        self.observation_dim = sum(self.env.observation_spaces["unit_0"].shape)
        self.action_dim = self.env.action_space.discrete.n
        self.state_dim = self.observation_dim * self.env.max_n_ally

    def init_train_state(self, key: jax.random.PRNGKey) -> TrainState:
        rngs = nnx.Rngs(self.config.seed)

        # Instantiate policy and critic functions
        pi = RNNHybridPolicy(
            action_dim=self.action_dim,
            obs_dim=self.observation_dim,
            layer_dim=self.config.mappo_layer_dim,
            rngs=rngs,
        )
        critic = RNNCritic(self.state_dim, layer_dim=self.config.mappo_layer_dim, rngs=rngs)

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
        hidden_state: chex.Array,
        key: jax.random.PRNGKey,
    ) -> Dict[str, Any]:
        policy, _ = get_model(train_state.policy_state)
        next_hidden_state, logits, mean, log_std = policy(hidden_state, obs)

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
        }

        return result

    def rollout(
        self, train_state: TrainState, scenario: Scenario
    ) -> Tuple[TrainState, Dict[str, Any]]:
        policy, _ = get_model(train_state.policy_state)
        critic, _ = get_model(train_state.critic_state)

        reset_key, key = jax.random.split(train_state.key)
        init_obs, env_state = self.v_reset(jax.random.split(reset_key, self.config.n_env), scenario)

        # Hidden states for RNN
        policy_hidden_state = policy.initialize_carry(
            (self.env.max_n_ally,) + init_obs["unit_0"].shape
        )
        critic_hidden_state = critic.initialize_carry(
            (self.config.n_env, self.env.max_n_ally + init_obs["unit_0"].shape[1])
        )

        def rollout_body(carry, _):
            obs, env_state, policy_hidden_state, critic_hidden_state, key = carry
            # TODO: need processed world state instead of concatenation?
            stacked_obs = jnp.stack([obs[i] for i in self.env.ally_keys], axis=1)
            world_state = stacked_obs.reshape(self.config.n_env, -1)

            action_key, step_key, key = jax.random.split(key, 3)
            sample_result = jax.vmap(self.sample_action, in_axes=(None, 1, 0, 0))(
                train_state,
                stacked_obs,
                policy_hidden_state,
                jax.random.split(action_key, self.env.max_n_ally),
            )

            actions = {}
            for i, agent in enumerate(self.env.ally_keys):
                actions[agent] = sample_result["actions"][i]

            next_obs, env_state, rewards, dones, infos = self.v_step(
                jax.random.split(step_key, self.config.n_env), env_state, actions
            )
            policy_hidden_state = sample_result["next_hidden_state"] * ~dones["__all__"][None, :]

            critic_hidden_state, state_value = critic(critic_hidden_state, world_state)
            critic_hidden_state = critic_hidden_state * ~dones["__all__"]

            sample_result.update(
                {
                    "state_value": state_value,
                    "dones": dones,
                    "observations": obs,
                    "rewards": rewards,
                    "infos": infos,
                    "env_state": env_state,
                    "states": world_state,
                }
            )

            return (
                next_obs,
                env_state,
                policy_hidden_state,
                critic_hidden_state,
                key,
            ), sample_result

        initial_carry = (init_obs, env_state, policy_hidden_state, critic_hidden_state, key)
        (last_obs, last_env_state, _, last_critic_hidden_state, key), rollout_result = jax.lax.scan(
            rollout_body, initial_carry, None, self.config.rollout_step_bs
        )

        last_obs = jnp.stack([last_obs[i] for i in self.env.ally_keys], axis=1)
        world_state = last_obs.reshape(self.config.n_env, -1)
        _, last_value = critic(last_critic_hidden_state, world_state)
        common_reward = rollout_result["rewards"][:, :, 0]

        rollout_result["common_reward"] = common_reward
        rollout_result["dones"] = rollout_result["dones"]["__all__"]
        rollout_result["last_value"] = last_value

        rollout_result["returned_episode_returns"] = last_env_state.returned_episode_returns
        rollout_result["returned_episode_lengths"] = last_env_state.returned_episode_lengths
        rollout_result["returned_episode_wins"] = last_env_state.returned_episode_wins

        return train_state.replace(key=key), rollout_result

    def train_step(
        self, train_state: TrainState, batch: Dict[str, Any]
    ) -> Tuple[TrainState, Dict[str, Any]]:
        def critic_loss_fn(critic: RNNCritic):
            state_values = jax.vmap(
                lambda feature, done: rnn_result(
                    critic, (self.config.mappo_layer_dim,), feature, done
                ),
                in_axes=(1, 1),
                out_axes=1,
            )(batch["states"], batch["dones"])[0]
            batch_values = batch["state_value"]
            clip_value = batch_values + jnp.clip(
                state_values - batch_values, -self.config.clip_value, self.config.clip_value
            )
            critic_losses = jnp.square(batch["returns"] - state_values)
            clip_losses = jnp.square(batch["returns"] - clip_value)
            critic_loss = 0.5 * jnp.maximum(critic_losses, clip_losses).mean()

            return critic_loss

        def policy_loss_fn(policy: RNNHybridPolicy):
            stacked_obs = jnp.stack(
                [batch["observations"][key] for key in self.env.ally_keys], axis=0
            )
            policy_vmap = jax.vmap(
                lambda feature, done: rnn_result(
                    policy, (self.config.mappo_layer_dim,), feature, done
                ),
                in_axes=(1, 1),
                out_axes=1,
            )
            logits, mean, log_std = jax.vmap(
                lambda obs: policy_vmap(obs, batch["dones"]), out_axes=1
            )(stacked_obs)

            discrete_distribution = tfd.Categorical(logits=logits)
            discrete_action = batch["discrete_actions"]
            discrete_log_pi = discrete_distribution.log_prob(discrete_action[:, :, :, 0])[
                :, :, :, None
            ]

            continuous_distribution = tfd.Normal(mean, jnp.exp(log_std))
            continuous_action = batch["continuous_actions"]
            continuous_log_pi = continuous_distribution.log_prob(continuous_action)

            log_pi = discrete_log_pi + continuous_log_pi

            ratio = jnp.exp(log_pi - batch["log_probs"])
            loss = jnp.minimum(
                ratio * batch["advantages"][:, None],
                jnp.clip(ratio, 1 - self.config.clip_ratio, 1 + self.config.clip_ratio)
                * batch["advantages"][:, None],
            ).mean()

            entropy = (
                discrete_distribution.entropy().mean() + continuous_distribution.entropy().mean()
            )

            return -loss - self.config.entropy_coef * entropy, {
                "policy_loss": loss,
                "entropy": entropy,
                "ratio": ratio,
                "ratio_max": ratio.max(),
                "ratio_min": ratio.min(),
                "ratio_mean": ratio.mean(),
                "advantage_max": batch["advantages"].max(),
                "advantage_min": batch["advantages"].min(),
                "advantage_mean": batch["advantages"].mean(),
            }

        critic, critic_optimizer = get_model(train_state.critic_state)
        policy, policy_optimizer = get_model(train_state.policy_state)

        critic_loss, critic_grads = nnx.value_and_grad(critic_loss_fn)(critic)
        (loss, info), policy_grads = nnx.value_and_grad(policy_loss_fn, has_aux=True)(policy)

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
        advantages, returns = jax.vmap(
            get_gae,
            in_axes=(1, 1, 1, 0, None, None),
            out_axes=1,
        )(
            batch["common_reward"],
            batch["dones"],
            batch["state_value"],
            batch["last_value"],
            self.config.gamma,
            self.config.lamda,
        )

        # Normalization (is there need to normalize each batch?)
        advantages = (advantages - advantages.mean(axis=1, keepdims=True)) / (
            advantages.std(axis=1, keepdims=True) + 1e-8
        )

        batch.update({"advantages": advantages, "returns": returns})

        def train_body(carry, _):
            (train_state,) = carry

            train_state, train_result = self.train_step(train_state, batch)

            return (train_state,), train_result

        (train_state,), train_result = jax.lax.scan(
            train_body, (train_state,), None, self.config.mappo_epochs
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
            "returned_episode_returns": batch["returned_episode_returns"],
            "returned_episode_lengths": batch["returned_episode_lengths"],
            "returned_episode_wins": batch["returned_episode_wins"],
        }

        return train_state, info
