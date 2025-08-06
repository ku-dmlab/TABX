from typing import Tuple, Dict, Any
import chex


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
        self, key: chex.PRNGKey, state: chex.Array, action: chex.Array
    ) -> Tuple[Dict[str, chex.Array], chex.Array, float, bool, Dict[str, Any]]:
        raise NotImplementedError

    def rollout(self, key: chex.PRNGKey, cfg: Dict[str, Any]) -> float:
        raise NotImplementedError
