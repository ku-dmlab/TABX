from functools import partial
from typing import Dict

import jax
import jax.numpy as jnp
import chex
from flax import struct

from src.maenv.environments.base_maenv import BaseMAEnv
from src.maenv.environments.spaces import Discrete, Box
from src.maenv.tabs.units import get_all_unit_spec, get_all_unit_names
from src.maenv.tabs.scenarios import Scenario


@struct.dataclass
class State:
    budget: int
    all_unit_spec: Dict[str, Dict[str, chex.Array]]
    current_unit_list: chex.Array
    enemy_unit_comp: chex.Array
    action_mask: chex.Array
    scenario: Scenario


class TABSUnitComb(BaseMAEnv):
    def __init__(self) -> None:
        self._min_budget = min(get_all_unit_spec()[0])
        self.num_units = len(get_all_unit_names())

        max_agents = 50
        self.action_space = Discrete(num_categories=self.num_units)  # unit id
        self.observation_space = Box(
            low=0, high=max_agents, shape=(1 + self.num_units * 3), dtype=jnp.float32
        )

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

    def reset(self, key, scenario: Scenario):
        # assert scenario.budget >= self._min_budget
        chex.assert_equal(scenario.enemy_unit_comp.shape, (self.num_units,))
        state = State(
            budget=scenario.budget,
            all_unit_spec=get_all_unit_spec(),
            current_unit_list=jnp.zeros(self.num_units, dtype=jnp.int32),
            enemy_unit_comp=scenario.enemy_unit_comp,
            action_mask=jnp.where(scenario.budget >= get_all_unit_spec()[0], True, False),
            scenario=scenario,
        )
        return self.get_obs(state), state

    def step_env(self, key, state, action):
        all_prices = state.all_unit_spec[0]
        purchase_valid = all_prices[action] <= state.budget
        new_unit_list = jnp.where(
            purchase_valid, state.current_unit_list + 1, state.current_unit_list
        )
        new_budget = (state.budget - all_prices[action] * purchase_valid).astype(jnp.int32)
        action_mask = jnp.where(new_budget >= all_prices, True, False)
        # Update state
        state = state.replace(
            budget=new_budget,
            current_unit_list=new_unit_list.astype(jnp.int32),
            action_mask=action_mask,
        )
        # NOTE: Reward would be computed by the result of battle with this unit combination.
        reward = None

        # Episodes will continue until no more units can be purchased.
        done = jnp.where(state.budget < self._min_budget, 1, 0)

        return self.get_obs(state), state, reward, done, {}
