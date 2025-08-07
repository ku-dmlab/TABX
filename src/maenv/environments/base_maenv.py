from typing import Tuple, Dict, Optional, Any
import chex

import jax


class BaseMAEnv:
    def __init__(self, num_agents: int, physics_config: Dict[str, float]) -> None:
        self.num_agents = num_agents
        self.physics_config = physics_config

    def get_obs(self, state: Dict[str, Any]) -> chex.Array:
        raise NotImplementedError

    def reset(
        self, key: chex.PRNGKey, cfg: Dict[str, Any]
    ) -> Tuple[Dict[str, chex.Array], Dict[str, Any]]:
        raise NotImplementedError

    def step(
        self,
        key: chex.PRNGKey,
        state: chex.Array,
        action: chex.Array,
        reset_state: Optional[chex.Array] = None,
    ) -> Tuple[Dict[str, chex.Array], Dict[str, Any]]:
        key, key_reset = jax.random.split(key)
        obs_st, states_st, reward, done, info = self.step_env(key, state, action)

        if reset_state is None:
            obs_re, states_re = self.reset(key_reset, states_st.scenario)
        else:
            states_re = reset_state
            obs_re = self.get_obs(states_re)

        states = jax.tree.map(lambda x, y: jax.lax.select(done, x, y), states_re, states_st)
        obs = jax.tree.map(lambda x, y: jax.lax.select(done, x, y), obs_re, obs_st)

        return obs, states, reward, done, info

    def step_env(
        self, key: chex.PRNGKey, state: chex.Array, action: chex.Array
    ) -> Tuple[Dict[str, chex.Array], chex.Array, float, bool, Dict[str, Any]]:
        raise NotImplementedError

    def rollout(self, key: chex.PRNGKey, cfg: Dict[str, Any]) -> float:
        raise NotImplementedError
