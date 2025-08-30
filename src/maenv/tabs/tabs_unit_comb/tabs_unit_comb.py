import jax.numpy as jnp
import chex
from flax import struct

from src.maenv.environments.base_maenv import BaseMAEnv
from src.maenv.environments.spaces import Discrete, Box
from src.maenv.tabs.scenarios import Scenario, TABSConf


@struct.dataclass
class State:
    budget: chex.Array
    all_price: chex.Array
    all_spec: chex.Array
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
            low=0,
            high=jnp.inf,
            shape=(1 + self.max_num_units * 3 + self.max_num_units**2),
            dtype=jnp.float32,
        )

    def get_obs(self, state):
        """
        Return observation including
            - Current budget
            - Current purchased unit list
            - All units' price
            - All units' spec
            - Enemy unit composition
        """

        return jnp.concatenate(
            (
                state.budget,
                state.current_unit_list,
                state.all_price,
                state.all_spec.flatten(),
                state.enemy_unit_comp,
            )
        )

    def reset(self, key, scenario: Scenario):
        # TODO: implement random enemy unit composition
        self.num_units = jnp.sum(scenario.unit_comp_mask)
        # assert scenario.budget >= self._min_budget
        chex.assert_equal(scenario.enemy_unit_comp.shape, (self.max_num_units,))
        action_mask = 1 - (
            jnp.where(scenario.budget >= scenario.price, True, False) * scenario.unit_comp_mask
        )

        all_spec = jnp.vstack(
            (
                scenario.health,
                scenario.body_radius,
                scenario.body_weight,
                scenario.speed,
                scenario.attack_damage,
                scenario.attack_range,
                scenario.attack_cooldown,
            )
        )

        state = State(
            budget=scenario.budget,
            all_price=scenario.price,
            all_spec=all_spec,
            current_unit_list=jnp.zeros(self.max_num_units, dtype=jnp.int32),
            enemy_unit_comp=scenario.enemy_unit_comp,
            unit_comp_mask=scenario.unit_comp_mask,
            action_mask=action_mask.astype(jnp.float32),
            scenario=scenario,
        )
        return self.get_obs(state), state

    def step_env(self, key, state, action):
        mask = jnp.zeros(self.max_num_units, dtype=jnp.bool_)
        mask = mask.at[action].set(True)

        valid_max_agent = jnp.where(
            jnp.sum(state.current_unit_list) + 1 <= self.max_agents, 1, 0
        ).astype(jnp.bool_)
        purchase_valid = state.all_price <= state.budget
        purchase_valid = purchase_valid & mask & valid_max_agent

        new_unit_list = jnp.where(
            purchase_valid, state.current_unit_list + 1, state.current_unit_list
        )
        new_budgets = jnp.where(purchase_valid, state.budget - state.all_price, state.budget)
        budget = new_budgets[action].astype(jnp.int32)[None]

        action_mask = 1-(
            jnp.where(budget >= state.all_price, True, False)
            * state.unit_comp_mask
            * valid_max_agent
        )
        # Update state
        state = state.replace(
            budget=budget,
            current_unit_list=new_unit_list.astype(jnp.int32),
            action_mask=action_mask.astype(jnp.float32),
        )
        # NOTE: Reward would be computed by the result of battle with this unit combination.
        reward = None

        # Episodes will continue until no more units can be purchased.
        done = jnp.where(
            jnp.logical_not(valid_max_agent)
            | (state.budget < jnp.min(jnp.where(state.all_price > 0, state.all_price, jnp.inf))),
            1,
            0,
        )

        return self.get_obs(state), state, reward, done, {}
