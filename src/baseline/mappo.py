import jax
import jax.numpy as jnp
import tensorflow_probability.substrates.jax as tfp
from src.baseline.utils import NetworkState, get_model, rnn_result, get_gae
from flax.struct import dataclass
from typing import Tuple, Dict, Any
from src.maenv.tabs.tabs_battle_simulator.heuristic_policy import heuristic_policy
from flax import nnx
from src.baseline.module.modules import RNNPolicy, RNNValue
import optax
from functools import partial
import os
import orbax.checkpoint as ocp
from etils import epath
from src.baseline.utils import get_abs_path
import json


@dataclass
class MAPPOState:
    policy_state: NetworkState
    value_state: NetworkState
    key: jax.random.PRNGKey


tfd = tfp.distributions
tfb = tfp.bijectors


class MAPPO:
    def __init__(self, config, env):
        self.config = config
        self.env = env
        self.v_reset = jax.vmap(lambda key: env.reset(key, None))
        self.v_step = jax.vmap(env.step, in_axes=(0, 0, 0))

    def sample_action(
        self,
        train_state: MAPPOState,
        hidden_state: jnp.array,
        obs: jnp.array,
        key: jax.random.PRNGKey,
    ) -> Tuple[MAPPOState, Dict[str, Any]]:
        policy, policy_optimizer = get_model(train_state.policy_state)
        next_hidden_state, logits, mean, log_std = policy(hidden_state, obs)

        discrete_key, continuous_key = jax.random.split(key, 2)

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

    def rollout(self, train_state: MAPPOState) -> Tuple[MAPPOState, Dict[str, Any]]:
        key = train_state.key

        policy, policy_optimizer = get_model(train_state.policy_state)
        value, value_optimizer = get_model(train_state.value_state)

        reset_key, key = jax.random.split(key)
        init_obs, init_env_state = self.v_reset(jax.random.split(reset_key, self.config.n_env))
        init_policy_hidden_state = policy.initialize_carry(
            (self.env.max_n_ally,) + init_obs["unit_0"].shape
        )
        init_value_hidden_state = value.initialize_carry(
            (self.config.n_env, self.env.max_n_ally + init_obs["unit_0"].shape[1])
        )

        def rollout_(carry, _):
            (env_state, obs, policy_hidden_state, value_hidden_state, key) = carry
            stacked_obs = jnp.stack([obs[i] for i in self.env.ally_keys], axis=1)
            state = stacked_obs.reshape(self.config.n_env, -1)
            stacked_obs = jnp.permute_dims(stacked_obs, (1, 0, 2))

            action_key, enemy_key, step_key, reset_key, next_key = jax.random.split(key, 5)

            sample_result = jax.vmap(
                lambda hidden_state, obs, key: self.sample_action(
                    train_state, hidden_state, obs, key
                ),
                in_axes=(0, 0, 0),
            )(policy_hidden_state, stacked_obs, jax.random.split(action_key, self.env.max_n_ally))
            actions = {}
            for i, agent in enumerate(self.env.ally_keys):
                actions[agent] = sample_result["actions"][i]

            #TODO: instead use wrapper
            for i, agent in enumerate(self.env.enemy_keys):
                emeny_action_key, emeny_key = jax.random.split(enemy_key)
                actions[agent] = jax.vmap(heuristic_policy, in_axes=(None, 0, None))(
                    emeny_action_key, obs[agent], self.env.num_agents
                )

            next_obs, next_env_states, rewards, dones, infos = self.v_step(
                jax.random.split(step_key, self.config.n_env), env_state, actions
            )
            next_policy_hidden_state = (
                sample_result["next_hidden_state"] * ~dones["__all__"][None, :]
            )

            next_value_hidden_state, state_value = value(value_hidden_state, state)
            next_value_hidden_state = next_value_hidden_state * ~dones["__all__"]

            sample_result.update(
                {
                    "rewards": rewards,
                    "dones": dones,
                    "observations": obs,
                    "env_state": env_state,
                    "infos": infos,
                    "state_value": state_value,
                    "states": state,
                }
            )

            return (
                next_env_states,
                next_obs,
                next_policy_hidden_state,
                next_value_hidden_state,
                next_key,
            ), sample_result

        (
            (last_env_state, last_obs, last_obs_hidden_state, last_value_hidden_state, last_key),
            rollout_result,
        ) = jax.lax.scan(
            rollout_,
            (init_env_state, init_obs, init_policy_hidden_state, init_value_hidden_state, key),
            None,
            self.config.rollout_step,
        )
        last_obs = jnp.stack([last_obs[i] for i in self.env.ally_keys], axis=1)
        state = last_obs.reshape(self.config.n_env, -1)
        _, last_value = value(last_value_hidden_state, state)
        common_reward = rollout_result["rewards"][:, :, 0]

        rollout_result["common_reward"] = common_reward
        rollout_result["dones"] = rollout_result["dones"]["__all__"]
        rollout_result["last_value"] = last_value

        rollout_result["returned_episode_returns"] = last_env_state.returned_episode_returns
        rollout_result["returned_episode_lengths"] = last_env_state.returned_episode_lengths
        rollout_result["returned_episode_wins"] = last_env_state.returned_episode_wins

        train_state = train_state.replace(key=last_key)
        return train_state, rollout_result

    def init_train_state(self) -> MAPPOState:
        rngs = nnx.Rngs(self.config.seed)
        pi = RNNPolicy(
            action_dim=self.config.action_dim,
            obs_dim=self.config.obs_dim,
            layer_dim=self.config.layer_dim,
            rngs=rngs,
        )
        value = RNNValue(self.config.state_dim, layer_dim=self.config.layer_dim, rngs=rngs)

        pi_optimizer = nnx.Optimizer(
            pi,
            optax.chain(
                optax.clip_by_global_norm(self.config.max_grad_norm),
                optax.adam(learning_rate=self.config.lr),
            ),
        )
        value_optimizer = nnx.Optimizer(
            value,
            optax.chain(
                optax.clip_by_global_norm(self.config.max_grad_norm),
                optax.adam(learning_rate=self.config.lr),
            ),
        )

        (pi_gd, policy_state) = nnx.split((pi, pi_optimizer))
        (value_gd, value_state) = nnx.split((value, value_optimizer))

        return MAPPOState(
            policy_state=NetworkState(pi_gd, policy_state),
            value_state=NetworkState(value_gd, value_state),
            key=jax.random.key(self.config.seed),
        )

    def train_step(
        self, train_state: MAPPOState, batch: Dict[str, Any]
    ) -> Tuple[MAPPOState, Dict[str, Any]]:
        def value_loss_fn(value):
            state_values = jax.vmap(
                lambda feature, done: rnn_result(value, (self.config.layer_dim,), feature, done),
                in_axes=(1, 1),
                out_axes=1,
            )(batch["states"], batch["dones"])[0]
            batch_values = batch["state_value"]
            clip_value = batch_values + jnp.clip(
                state_values - batch_values, -self.config.clip_value, self.config.clip_value
            )
            value_losses = jnp.square(batch["returns"] - state_values)
            clip_losses = jnp.square(batch["returns"] - clip_value)
            value_loss = 0.5 * jnp.maximum(value_losses, clip_losses).mean()

            return value_loss

        def policy_loss(policy):
            stacked_obs = jnp.stack(
                [batch["observations"][key] for key in self.env.ally_keys], axis=0
            )
            policy_vmap = jax.vmap(
                lambda feature, done: rnn_result(policy, (self.config.layer_dim,), feature, done),
                in_axes=(1, 1),
                out_axes=1,
            )
            policy_output = jax.vmap(lambda obs: policy_vmap(obs, batch["dones"]), out_axes=1)(
                stacked_obs
            )
            logits, mean, log_std = policy_output
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
            }

        value, value_optimizer = get_model(train_state.value_state)
        policy, policy_optimizer = get_model(train_state.policy_state)

        v_loss, value_grads = nnx.value_and_grad(value_loss_fn)(value)
        (loss, info), policy_grads = nnx.value_and_grad(policy_loss, has_aux=True)(policy)

        value_optimizer.update(value_grads)
        policy_optimizer.update(policy_grads)

        train_state = train_state.replace(
            policy_state=train_state.policy_state.replace(
                state=nnx.state((policy, policy_optimizer))
            ),
            value_state=train_state.value_state.replace(state=nnx.state((value, value_optimizer))),
        )

        info.update(
            {
                "v_loss": v_loss,
                "ratio_max": info["ratio"].max(),
                "ratio_min": info["ratio"].min(),
                "ratio_mean": info["ratio"].mean(),
            }
        )

        return train_state, info

    def train(self, train_state, step):
        def train_body(carry, _):
            (train_state,) = carry

            train_state, rollout_result = self.rollout(train_state)

            gae, returns = jax.vmap(
                partial(get_gae, lamda=self.config.lamda, gamma=self.config.gamma),
                in_axes=(1, 1, 1, 0),
            )(
                rollout_result["common_reward"],
                rollout_result["dones"],
                rollout_result["state_value"],
                rollout_result["last_value"],
            )

            gae = gae.transpose(1, 0, 2)
            returns = returns.transpose(1, 0, 2) # instead of transpose, use vmap with out_axes=1
            rollout_result.update(
                {
                    "advantages": (gae - gae.mean()) / (gae.std() + 1e-8),
                    "returns": returns,
                }  # gae normalization (is there need to normalize each batch?)
            )

            def ppo_train_body(carry, _):
                (train_state,) = carry

                train_state, train_result = self.train_step(train_state, rollout_result)

                return (train_state,), train_result

            (
                (train_state,),
                train_result,
            ) = jax.lax.scan(ppo_train_body, (train_state,), None, self.config.ppo_epochs)

            ppo_result = {
                "v_loss": train_result["v_loss"].mean(),
                "policy_loss": train_result["policy_loss"].mean(),
                "entropy": train_result["entropy"].mean(),
                "ratio": train_result["ratio"].mean(),
                "ratio_max": train_result["ratio_max"].max(),
                "ratio_min": train_result["ratio_min"].min(),
                "rewards": rollout_result["common_reward"].sum() / self.config.n_env,
                "returned_episode_returns": rollout_result["returned_episode_returns"],
                "returned_episode_lengths": rollout_result["returned_episode_lengths"],
                "returned_episode_wins": rollout_result["returned_episode_wins"],
            }

            return (train_state,), ppo_result

        (
            (train_state,),
            ppo_result,
        ) = jax.lax.scan(train_body, (train_state,), None, step)

        return train_state, ppo_result

    def save_state(self, train_state, path):
        path = get_abs_path(path)

        with ocp.StandardCheckpointer() as checkpointer:
            checkpointer.save(epath.Path(path), train_state)

        # Save config to the checkpoint directory
        config_path = os.path.join(path, "config.json")
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        with open(config_path, "w") as f:
            json.dump(self.config.__dict__, f, indent=2)

    def load_state(self, path, update_config=False):
        path = get_abs_path(path)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Path {path} does not exist")

        if update_config:
            config_path = os.path.join(path, "config.json")
            with open(config_path, "r") as f:
                config = json.load(f)

            for key, value in config.items():
                setattr(self.config, key, value)

        checkpointer = ocp.StandardCheckpointer()
        train_state = checkpointer.restore(epath.Path(path), self.init_train_state())

        return train_state
