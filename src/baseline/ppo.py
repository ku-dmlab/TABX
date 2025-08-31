import jax
import jax.numpy as jnp
import tensorflow_probability.substrates.jax as tfp
from src.baseline.utils import NetworkState, get_model
from typing import Tuple, Dict, Any
from flax.struct import dataclass
from flax import nnx
from src.baseline.module.modules import Policy, Value
import optax
import os
import orbax.checkpoint as ocp
from etils import epath
from src.baseline.utils import get_abs_path
import json
from src.maenv.tabs.tabs_unit_deploy.tabs_unit_deploy import TABSUnitDeploy
from src.maenv.tabs.tabs_unit_comb.tabs_unit_comb import TABSUnitComb

@dataclass
class PPOState:
    policy_state: NetworkState
    value_state: NetworkState
    key: jax.random.PRNGKey


tfd = tfp.distributions
tfb = tfp.bijectors


class PPO:
    def __init__(self, config, env: TABSUnitComb | TABSUnitDeploy):
        self.config = config
        self.env = env
        self.v_reset = jax.vmap(env.reset, in_axes=(0, 0))
        self.v_step = jax.vmap(env.step_env, in_axes=(0, 0, 0))

        self.observation_dim = sum(self.env.observation_space.shape)
        self.action_dim = self.env.action_space.n



    def init_train_state(self) -> PPOState:
        rngs = nnx.Rngs(self.config.seed)
        pi = Policy(
            action_dim=self.action_dim,
            state_dim=self.observation_dim,
            layer_dim=self.config.layer_dim,
            rngs=rngs,
        )
        value = Value(self.observation_dim, layer_dim=self.config.layer_dim, rngs=rngs)

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

        return PPOState(
            policy_state=NetworkState(pi_gd, policy_state),
            value_state=NetworkState(value_gd, value_state),
            key=jax.random.key(self.config.seed),
        )

    def sample_action(self, policy, observation, action_mask, key):
        logits = policy(observation)
        logits = jnp.where(action_mask, -jnp.inf, logits)
        dist = tfd.Categorical(logits=logits)
        actions = dist.sample(seed=key)
        log_probs = dist.log_prob(actions)
        
        result = {
            'actions' : actions,
            'log_probs' : log_probs,
        }
        
        return result

    def rollout(self, ppo_state, scenario):
        policy, policy_optimizer = get_model(ppo_state.policy_state)
        value, value_optimizer = get_model(ppo_state.value_state)

        reset_key, key = jax.random.split(ppo_state.key)
        init_obs, init_env_state = self.v_reset(jax.random.split(reset_key, self.config.n_env), scenario)

        initial_carry = (init_obs, init_env_state, key)

        def rollout_body(carry, _):
            obs, state, key = carry
            sample_key, step_key, next_key = jax.random.split(key, 3)
            sample_result = self.sample_action(policy, obs, state.action_mask, sample_key)
            next_obs, next_state, _, done, _ = self.v_step(jax.random.split(step_key, self.config.n_env), state, sample_result['actions'])

            obs_value = value(obs)

            sample_result.update({
                'values' : obs_value,
                'dones' : done,
                'observations' : obs,
                'action_masks' : state.action_mask,
                'state' : state
            })

            return (next_obs, next_state, next_key), sample_result

        (last_obs, last_state, key), rollout_result = jax.lax.scan(rollout_body, initial_carry, None, self.config.rollout_step)

        last_value = value(last_obs)
        rollout_result["last_value"] = last_value
        rollout_result["last_state"] = last_state

        return ppo_state.replace(key = key), rollout_result

    def train_step(
        self, train_state: PPOState, batch: Dict[str, Any]
    ) -> Tuple[PPOState, Dict[str, Any]]:
        def value_loss_fn(value):
            state_values = value(batch['observations'])
            batch_values = batch["values"]
            clip_value = batch_values + jnp.clip(
                state_values - batch_values, -self.config.clip_value, self.config.clip_value
            )
            value_losses = jnp.square(batch["returns"] - state_values)
            clip_losses = jnp.square(batch["returns"] - clip_value)
            value_loss = 0.5 * jnp.maximum(value_losses, clip_losses).mean()

            return value_loss

        def policy_loss(policy):
            logits = policy(batch['observations'])
            logits = jnp.where(batch['action_masks'], -jnp.inf, logits)
            dist = tfd.Categorical(logits=logits)
            log_pi = dist.log_prob(batch['actions'])

            log_diff = log_pi - batch["log_probs"]
            is_nan_log_diff = jnp.isnan(log_diff)
            log_diff = jnp.where(is_nan_log_diff, 0.0, log_diff)

            ratio = jnp.exp(log_diff).reshape(batch['advantages'].shape)
            loss = (jnp.minimum(
                ratio * batch["advantages"],
                jnp.clip(ratio, 1 - self.config.clip_ratio, 1 + self.config.clip_ratio)
                * batch["advantages"],
            )* (1-is_nan_log_diff.reshape(batch['advantages'].shape))).sum() / (~is_nan_log_diff).sum()

            is_nan_log_pi = jnp.isnan(log_pi)
            entropy = jnp.where(is_nan_log_pi, 0.0, -log_pi * jnp.exp(log_pi)).sum() / (~is_nan_log_pi).sum()


            return -loss + self.config.entropy_coef * entropy, {
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


    def train(self, train_state, batch):
        #TODO : consider batch size
        def train_body(carry, _):
            (train_state,) = carry

            train_state, info = self.train_step(train_state, batch)

            return (train_state,), info

        train_state, info = jax.lax.scan(train_body, (train_state,), None, self.config.ppo_epochs)

        info["policy_loss"] = info["policy_loss"].mean()
        info["entropy"] = info["entropy"].mean()
        info["ratio"] = info["ratio"].mean()
        info["ratio_max"] = info["ratio_max"].max()
        info["ratio_min"] = info["ratio_min"].min()
        info["ratio_mean"] = info["ratio_mean"].mean()
        info["v_loss"] = info["v_loss"].mean()

        return train_state, info
            


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



