import os
import json
from etils import epath
from typing import Tuple, Dict, Any, overload

import chex
import jax
import orbax.checkpoint as ocp

from src.baseline.utils import TrainState, get_abs_path
from src.maenv.tabs.scenarios import Scenario
from src.maenv.environments.base_maenv import BaseMAEnv


class BaseAlgo:
    def __init__(self, config, env: BaseMAEnv):
        self.config = config
        self.env = env

    @overload
    def init_train_state(self, key: jax.random.PRNGKey) -> TrainState:
        pass

    @overload
    def sample_action(
        self, train_state: TrainState, obs: chex.Array, key: jax.random.PRNGKey
    ) -> Dict[str, Any]:
        pass

    def rollout(
        self, train_state: TrainState, scenario: Scenario
    ) -> Tuple[TrainState, Dict[str, Any]]:
        pass

    def train_step(
        self, train_state: TrainState, batch: Dict[str, Any]
    ) -> Tuple[TrainState, Dict[str, Any]]:
        pass

    @overload
    def train(
        self, train_state: TrainState, batch: Dict[str, Any]
    ) -> Tuple[TrainState, Dict[str, Any]]:
        pass

    def save_state(self, train_state: TrainState, path: str):
        path = get_abs_path(path)

        with ocp.StandardCheckpointer() as checkpointer:
            checkpointer.save(epath.Path(path), train_state)

        # Save config to the checkpoint directory
        config_path = os.path.join(path, "config.json")
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        with open(config_path, "w") as f:
            json.dump(self.config.__dict__, f, indent=2)

    def load_state(self, path: str, update_config: bool = False):
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
