from typing import Callable
from typing import Dict as Level

import jax
import jax.numpy as jnp

from src.tabs.wrappers import BaseWrapper


class LevelAutoResetWrapper(BaseWrapper):
    """
    Wrapper for TABS that adds automatic reset by level functionality.
    """

    def __init__(self, env, sample_level: Callable):
        super().__init__(env)
        self.sample_level = sample_level

    def step(self, rng, env_state, action, env_params: Level):
        rng_sample, rng_reset, rng_step = jax.random.split(rng, 3)
        new_level = self.sample_level(env_params, rng_sample)

        reset_obs, reset_state = self.reset(rng_reset, new_level)
        next_obs, next_state, reward, done, info = self.env.step(rng_step, env_state, action)
        ep_done = done["__all__"]
        next_env_state = jax.tree.map(
            lambda x, y: jnp.where(ep_done, x, y),
            reset_state,
            next_state,
        )

        # log state is not reset
        if "log_state" in next_state:
            next_env_state["log_state"] = next_state["log_state"]

        next_obs = jax.tree.map(
            lambda x, y: jnp.where(ep_done, x, y),
            reset_obs,
            next_obs,
        )
        return next_obs, next_env_state, reward, done, info
