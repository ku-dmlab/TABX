import jax.numpy as jnp
import chex
from flax import struct

from src.maenv.environments.base_maenv import BaseMAEnv
from src.maenv.environments.spaces import Discrete, Box
from src.maenv.tabs.scenarios import Scenario, TABSConf


@struct.dataclass
class State:
    budget: int
    all_price: chex.Array
    current_unit_list: chex.Array
    enemy_unit_comp: chex.Array
    unit_comp_mask: chex.Array
    action_mask: chex.Array
    scenario: Scenario


class TABSUnitComb(BaseMAEnv):
    def __init__(self, cfg: TABSConf) -> None:
        self.max_num_units = cfg.max_num_units
        self.max_agents = cfg.max_agents

        self.action_space = Discrete(num_categories=self.max_num_units)  # unit id
        self.observation_space = Box(
            low=0, high=self.max_agents, shape=(1 + self.max_num_units * 3), dtype=jnp.float32
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
                state.all_price,
                state.enemy_unit_comp,
            )
        )

    def reset(self, key, scenario: Scenario):
        self.num_units = jnp.sum(scenario.unit_comp_mask)
        # assert scenario.budget >= self._min_budget
        chex.assert_equal(scenario.enemy_unit_comp.shape, (self.max_num_units,))
        action_mask = (
            jnp.where(scenario.budget >= scenario.price, True, False) * scenario.unit_comp_mask
        )
        state = State(
            budget=scenario.budget,
            all_price=scenario.price,
            current_unit_list=jnp.zeros(self.max_num_units, dtype=jnp.int32),
            enemy_unit_comp=scenario.enemy_unit_comp,
            unit_comp_mask=scenario.unit_comp_mask,
            action_mask=action_mask,
            scenario=scenario,
        )
        return self.get_obs(state), state

    def step_env(self, key, state, action):
        purchase_valid = state.all_price[action] <= state.budget
        new_unit_list = jnp.where(
            purchase_valid, state.current_unit_list + 1, state.current_unit_list
        )
        new_budget = (state.budget - state.all_price[action] * purchase_valid).astype(jnp.int32)
        action_mask = jnp.where(new_budget >= state.all_price, True, False) * state.unit_comp_mask
        # Update state
        state = state.replace(
            budget=new_budget,
            current_unit_list=new_unit_list.astype(jnp.int32),
            action_mask=action_mask,
        )
        # NOTE: Reward would be computed by the result of battle with this unit combination.
        reward = None

        # Episodes will continue until no more units can be purchased.
        done = jnp.where(
            state.budget < jnp.min(jnp.where(state.all_price > 0, state.all_price, jnp.inf)), 1, 0
        )

        return self.get_obs(state), state, reward, done, {}
