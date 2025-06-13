from typing import Tuple, Dict, Any
import chex


class BaseMAEnv:
    def __init__(self, config):
        self.config = config

    def reset(
        self, key: chex.PRNGKey, env_params: Dict[str, chex.Array | float] = None
    ) -> Tuple[Dict[str, chex.Array], Dict[str, Any]]:
        raise NotImplementedError

    def step(
        self, key: chex.PRNGKey, state: chex.Array, action: chex.Array
    ) -> Tuple[Dict[str, chex.Array], chex.Array, float, bool, Dict[str, Any]]:
        raise NotImplementedError
