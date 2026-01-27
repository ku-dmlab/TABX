from typing import Any, Dict, Tuple

import chex


class BaseMAEnv:
    def __init__(self, num_agents: int) -> None:
        self.num_agents = num_agents

    def get_obs(self, state: Dict[str, Any]) -> chex.Array:
        raise NotImplementedError

    def reset(
        self, key: chex.PRNGKey, cfg: Dict[str, Any]
    ) -> Tuple[Dict[str, chex.Array], Dict[str, Any]]:
        raise NotImplementedError

    def step(
        self, key: chex.PRNGKey, state: chex.Array, action: chex.Array
    ) -> Tuple[Dict[str, chex.Array], chex.Array, float, bool, Dict[str, Any]]:
        raise NotImplementedError

    def rollout(self, key: chex.PRNGKey, cfg: Dict[str, Any]) -> float:
        raise NotImplementedError

    def action_space(self, agent: str):
        return self.action_spaces[agent]

    def observation_space(self, agent: str):
        return self.observation_spaces[agent]
