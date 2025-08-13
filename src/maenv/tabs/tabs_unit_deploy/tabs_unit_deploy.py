from typing import Tuple

import jax
import jax.numpy as jnp
import chex
from flax import struct

from src.maenv.environments.base_maenv import BaseMAEnv
from src.maenv.environments.spaces import Discrete, Box
from src.maenv.tabs.scenarios import Scenario, TABSConf
from src.maenv.tabs.tabs_unit_deploy.utils import convert_unit_layer, conv_lower_right_padding


@struct.dataclass
class State:
    next_unit: chex.Array  # The unit to be deploy next
    remaining_units: chex.Array  # Remaining units queue
    unit_comp_mask: chex.Array  # Avail unit types
    battle_field: chex.Array  # (H_max x W_max) matrix facing enemies
    battle_field_mask: (
        chex.Array
    )  # (H_max x W_max) matrix representing available deployment spots for the next unit
    enemy_battle_field: chex.Array  # (H_max x W_max) matrix facing allies
    enemy_battle_field_mask: chex.Array  # (H_max x W_max) mask for available enemy battle field
    space_occupied_spec: chex.Array
    scenario: Scenario


class TABSUnitDeploy(BaseMAEnv):
    def __init__(self, cfg: TABSConf) -> None:
        self.max_num_units = cfg.max_num_units
        self.max_agents = cfg.max_agents

        self.max_field_height = cfg.max_field_height
        self.max_field_width = cfg.max_field_width
        self.battle_field = jnp.zeros(
            (self.max_field_height, self.max_field_width), dtype=jnp.float32
        )
        max_field_size = self.max_field_height * self.max_field_width

        self.action_space = Discrete(num_categories=max_field_size)
        self.observation_space = Box(
            low=0,
            high=self.max_agents,
            shape=(1 + self.max_num_units + 2 * self.max_num_units * max_field_size,),
            dtype=jnp.float32,
        )

        self._step = 0

    def get_obs(self, state: State) -> chex.Array:
        """
        Return observation includes:
            - The unit id to be deploy next.
            - The list of units still available for deployment, including the next unit id.
            - The current state of the deployed units on the battlefield.
        """

        enemy_battle_field = convert_unit_layer(state.enemy_battle_field, self.max_num_units)
        ally_battle_field = convert_unit_layer(state.battle_field, self.max_num_units)

        battle_field = jnp.vstack((enemy_battle_field, ally_battle_field))
        obs = jnp.concatenate((state.next_unit, state.remaining_units, battle_field.flatten()))

        return obs

    def _get_deploy_mask(
        self,
        next_unit: chex.Array,
        mask: chex.Array,
        space_occupied_spec: chex.Array,
    ) -> chex.Array:
        # Get the occupied space size of the next unit
        space_sizes = jnp.sqrt(space_occupied_spec[:]).astype(jnp.int32)

        masks = jax.vmap(
            lambda size, kernel: jnp.where(
                size > 1,
                jnp.where(
                    conv_lower_right_padding(mask, kernel) >= size**2, True, False
                ),  # For Mammoth
                mask,  # Others
            ),
            in_axes=(0, None),
        )(space_sizes, jnp.ones((2, 2)))

        return masks[next_unit - 1][0].astype(jnp.bool_)

    def _get_place_mask(
        self,
        next_unit: chex.Array,
        mask: chex.Array,
        space_occupied_spec: chex.Array,
    ) -> chex.Array:
        space_sizes = jnp.sqrt(space_occupied_spec[:]).astype(jnp.int32)

        masks = jax.vmap(
            lambda size, kernel: jnp.where(
                size > 1,
                jax.scipy.signal.convolve2d(mask, kernel, mode="same").astype(jnp.bool_),
                mask,
            ),
            in_axes=(0, None),
        )(space_sizes, jnp.ones((2, 2)))

        return masks[next_unit - 1][0].astype(jnp.bool_)

    def reset(self, key, scenario: Scenario):
        # TODO: implement random enemy deployment
        enemy_battle_field = scenario.enemy_battle_field
        enemy_battle_field_mask = scenario.enemy_battle_field_mask

        next_unit = jnp.array(
            [jnp.argmax(scenario.ally_unit_comp * scenario.unit_comp_mask != 0) + 1]
        )  # ID starts from 1
        battle_field = jnp.zeros_like(scenario.battle_field)
        battle_field_mask = self._get_deploy_mask(
            next_unit, scenario.battle_field_mask, scenario.space_occupied
        )

        state = State(
            next_unit=next_unit,
            remaining_units=scenario.ally_unit_comp,
            unit_comp_mask=scenario.unit_comp_mask,
            battle_field=battle_field,  # ally battle field
            battle_field_mask=battle_field_mask.astype(jnp.float32),  # ally battle field mask
            enemy_battle_field=enemy_battle_field,
            enemy_battle_field_mask=enemy_battle_field_mask,
            space_occupied_spec=scenario.space_occupied,
            scenario=scenario,
        )
        return self.get_obs(state), state

    def step_env(self, key, state, action):
        # Deploy the next unit
        h, w = action // self.max_field_width, action % self.max_field_width
        try_deploy = jnp.zeros_like(state.battle_field, dtype=jnp.bool_)
        try_deploy = try_deploy.at[h, w].set(True)

        # Get the condition for deployment
        cond_deploy = try_deploy & state.battle_field_mask.astype(jnp.bool_)
        try_deploy = self._get_place_mask(state.next_unit, try_deploy, state.space_occupied_spec)
        battle_field = jnp.where(
            cond_deploy.any() & (cond_deploy | try_deploy), state.next_unit[0], state.battle_field
        )  # Deployed battle field

        # Remove the deployed unit from the list
        _cond_unit = jax.nn.one_hot(state.next_unit - 1, state.remaining_units.size)
        remaining_units = jnp.where(
            cond_deploy.any(),
            jnp.where(_cond_unit, state.remaining_units - 1, state.remaining_units),
            state.remaining_units,
        )

        # Get a mask for the next unit after deployment
        next_unit = jnp.array([jnp.argmax(remaining_units * state.unit_comp_mask != 0) + 1])
        battle_field_mask = jnp.where(
            cond_deploy.any(),
            self._get_deploy_mask(next_unit, jnp.logical_not(try_deploy), state.space_occupied_spec)
            & jnp.logical_not(battle_field).astype(jnp.bool_),
            state.battle_field_mask,
        )

        # Update state
        state = state.replace(
            next_unit=next_unit,
            remaining_units=remaining_units.flatten(),
            battle_field=battle_field.astype(jnp.float32),
            battle_field_mask=battle_field_mask.astype(jnp.float32),
        )

        # NOTE: Reward would be computed by the result of battle with this unit deployment.
        reward = None

        # Episode will continue until no more units can be deployed.
        done = jnp.where(state.remaining_units.sum(), 0, 1)

        return self.get_obs(state), state, reward, done, {}
