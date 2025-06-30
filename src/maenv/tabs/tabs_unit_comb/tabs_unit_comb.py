from functools import partial
from typing import Dict

import jax
import jax.numpy as jnp
import chex
from flax import struct

from src.maenv.environments.base_maenv import BaseMAEnv
from src.maenv.environments.spaces import Discrete
from src.maenv.tabs.units import get_all_unit_spec, get_all_unit_names
from src.maenv.tabs.scenarios import Scenario


@struct.dataclass
class State:
    budget: int
    all_unit_spec: Dict[str, Dict[str, chex.Array]]
    current_unit_list: chex.Array
    enemy_unit_comp: chex.Array


class TABSUnitComb(BaseMAEnv):
    def __init__(self, scenario: Scenario = None) -> None:
        self._min_budget = min(get_all_unit_spec()[0])
        if scenario:
            self.max_budget = scenario.budget
            self.enemy_unit_comp = scenario.enemy_unit_comp
        else:
            self.max_budget = 800
            self.enemy_unit_comp = jnp.array([10, 0, 0, 0, 0, 0, 0])  # 10 farmers
        self.num_units = len(get_all_unit_names())

        self.action_space = Discrete(num_categories=self.num_units)  # unit id
        # No need to define observation space

    def get_obs(self, state):
        """
        Return observation including
            - current budget
            - current purchased unit list
            - all units' prices
            - enemy unit composition
        """

        return jnp.concatenate(
            (
                jnp.array([state.budget]),
                state.current_unit_list,
                state.all_unit_spec[0],
                state.enemy_unit_comp,
            )
        )

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key):
        state = State(
            budget=self.max_budget,
            all_unit_spec=get_all_unit_spec(),
            current_unit_list=jnp.zeros(self.num_units, dtype=jnp.int32),
            enemy_unit_comp=self.enemy_unit_comp,
        )
        return self.get_obs(state), state

    @partial(jax.jit, static_argnums=(0,))
    def step(self, key, state, action):
        mask = jnp.zeros(self.num_units, dtype=jnp.bool_)
        mask = mask.at[action].set(True)

        purchase_valid = state.all_unit_spec[0] <= state.budget
        purchase_valid = purchase_valid & mask

        new_unit_list = jnp.where(
            purchase_valid, state.current_unit_list + 1, state.current_unit_list
        )
        new_budget = jnp.where(purchase_valid, state.budget - state.all_unit_spec[0], state.budget)

        # Update state
        state = state.replace(
            budget=new_budget[action],
            current_unit_list=new_unit_list,
        )

        # NOTE: Reward would be computed by the result of battle with this unit combination.
        reward = None

        # Episodes will continue until no more units can be purchased.
        done = jnp.where(state.budget < self._min_budget, 1, 0)

        return self.get_obs(state), state, reward, done, {}
