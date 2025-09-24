from typing import Tuple, Dict, Any, Optional

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
from src.baseline.configs.config import PPOConfig


tfd = tfp.distributions
tfb = tfp.bijectors


class MAPPO(BaseAlgo):
    def __init__(self, config: PPOConfig, env: TABSBattleSimulator):
        super(MAPPO, self).__init__(config, env)
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

        # Instantiate policy and critic functions
        pi = RNNHybridPolicy(
            action_dim=self.action_dim,
            obs_dim=self.observation_dim,
            layer_dim=self.config.layer_dim,
            rngs=rngs,
        )
        critic = RNNCritic(self.state_dim, layer_dim=self.config.layer_dim, rngs=rngs)
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
        pi_optimizer = nnx.Optimizer(
            pi,
            optax.chain(
                optax.clip_by_global_norm(self.config.max_grad_norm),
                optax.adam(learning_rate=learning_rate, eps=1e-5),
            ),
        )
        critic_optimizer = nnx.Optimizer(
            critic,
            optax.chain(
                optax.clip_by_global_norm(self.config.max_grad_norm),
                optax.adam(learning_rate=learning_rate, eps=1e-5),
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
        output = policy(hidden_state, obs)
        next_hidden_state = output[0]
        continuous_distribution, discrete_distribution = policy.get_distribution(*output[1:])

        # Get hybrid action
        discrete_key, continuous_key = jax.random.split(key)
        discrete_actions = discrete_distribution.sample(seed=discrete_key)
        discrete_log_probs = discrete_distribution.log_prob(discrete_actions)
        continuous_actions = jnp.clip(
            continuous_distribution.sample(seed=continuous_key),
            -jnp.pi / 12 + 1e-5,
            jnp.pi / 12 - 1e-5,
        )
        continuous_log_probs = continuous_distribution.log_prob(continuous_actions)

        actions = jnp.stack([continuous_actions, discrete_actions], axis=-1)
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
            rollout_body, initial_carry, None, self.config.rollout_step
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
        rollout_result["last_state"] = last_env_state

        return train_state.replace(key=key), rollout_result

    def train_step(
        self, train_state: TrainState, batch: Dict[str, Any]
    ) -> Tuple[TrainState, Dict[str, Any]]:
        def critic_loss_fn(critic: RNNCritic):
            state_values = jax.vmap(
                lambda feature, done: rnn_result(critic, (self.config.layer_dim,), feature, done),
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
                lambda feature, done: rnn_result(policy, (self.config.layer_dim,), feature, done),
                in_axes=(1, 1),
                out_axes=1,
            )
            output = jax.vmap(lambda obs: policy_vmap(obs, batch["dones"]), out_axes=1)(stacked_obs)

            continuous_distribution, discrete_distribution = policy.get_distribution(*output)
            discrete_action = batch["discrete_actions"]
            discrete_log_pi = discrete_distribution.log_prob(discrete_action)

            continuous_action = batch["continuous_actions"]
            continuous_log_pi = continuous_distribution.log_prob(continuous_action)

            log_pi = discrete_log_pi + continuous_log_pi

            log_ratio = log_pi - batch["log_probs"]

            ratio = jnp.exp(log_ratio)
            loss = jnp.minimum(
                ratio * batch["advantages"],
                jnp.clip(ratio, 1 - self.config.clip_ratio, 1 + self.config.clip_ratio)
                * batch["advantages"],
            ).mean()

            discrete_entropy = discrete_distribution.entropy().mean()
            continuous_entropy = -continuous_log_pi.mean()

            entropy = discrete_entropy + continuous_entropy

            approx_kl = ((ratio - 1) - log_ratio).mean()

            return -(loss + self.config.entropy_coef * entropy), {
                "policy_loss": loss,
                "entropy": entropy,
                "ratio": ratio,
                "ratio_max": ratio.max(),
                "ratio_min": ratio.min(),
                "ratio_mean": ratio.mean(),
                "approx_kl": approx_kl,
                "discrete_entropy": discrete_entropy,
                "continuous_entropy": continuous_entropy,
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
            out_axes=-1,
        )(
            batch["common_reward"],
            batch["dones"],
            batch["state_value"],
            batch["last_value"],
            self.config.gamma,
            self.config.lamda,
        )

        batch.update({"advantages": advantages, "returns": returns})
        batch1 = {
            "advantages": batch["advantages"],
            "continuous_actions": batch["continuous_actions"],
            "discrete_actions": batch["discrete_actions"],
            "log_probs": batch["log_probs"],
            "returns": batch["returns"],
        }

        batch2 = {
            "states": batch["states"],
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
                    lambda idx: jax.tree.map(lambda x: x[:, idx], batch2), out_axes=1
                )(batch_idx)

                batch1_.update(batch2_)
                batch1_["advantages"] = (batch1_["advantages"] - batch1_["advantages"].mean()) / (
                    batch1_["advantages"].std() + 1e-8
                )
                (
                    train_state,
                    train_result,
                ) = self.train_step(train_state, batch1_)
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
            "advantage_max": batch["advantages"].max(),
            "advantage_min": batch["advantages"].min(),
            "advantage_mean": batch["advantages"].mean(),
            "rewards": batch["common_reward"].sum() / self.config.n_env,
            "approx_kl_max": train_result["approx_kl"].max(),
            "approx_kl_min": train_result["approx_kl"].min(),
            "approx_kl_mean": train_result["approx_kl"].mean(),
            "discrete_entropy": train_result["discrete_entropy"].mean(),
            "continuous_entropy": train_result["continuous_entropy"].mean(),
        }

        return train_state, info
