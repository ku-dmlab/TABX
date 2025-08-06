from functools import partial

import jax
import jax.numpy as jnp
import chex
from flax import struct

from src.maenv.environments.base_maenv import BaseMAEnv
from src.maenv.environments.spaces import Discrete
from src.maenv.tabs.units import get_all_unit_spec
from src.maenv.tabs.scenarios import Scenario
from src.maenv.tabs.tabs_unit_deploy.utils import (
    convert_unit_layer,
    get_valid_battle_field_mask,
    conv_lower_right_padding,
)


@struct.dataclass
class State:
    next_unit: chex.Array  # The unit to be deploy next
    remaining_units: chex.Array  # Remaining units queue
    battle_field: chex.Array  # N x M matrix facing enemies
    enemy_battle_field: chex.Array  # N x M matrix facing allies
    battle_field_mask: chex.Array  # N x M matrix representing available deployment spots.


class TABSUnitDeploy(BaseMAEnv):
    def __init__(self, scenario: Scenario = None) -> None:
        self._min_budget = min(get_all_unit_spec()[0])
        if scenario:
            # Check whether enemies are successfully deployed
            is_valid, _ = get_valid_battle_field_mask(scenario.enemy_battle_field)
            if not is_valid:
                raise ValueError
            self.enemy_battle_field = scenario.enemy_battle_field
            self.ally_unit_comp = scenario.ally_unit_comp
            assert scenario.field_width * scenario.field_height >= jnp.sum(self.ally_unit_comp)
            self.field_height = scenario.field_height
            self.field_width = scenario.field_width
        else:
            self.enemy_battle_field = jnp.concatenate(
                (jnp.ones((2, 5), dtype=jnp.float32), jnp.zeros((2, 5), dtype=jnp.float32)), axis=0
            )  # 10 farmers
            self.ally_unit_comp = jnp.array([10, 0, 0, 0, 0, 0, 0], dtype=jnp.float32)  # 10 farmers
            self.field_height = 4
            self.field_width = 5

        self.action_space = Discrete(num_categories=self.field_height * self.field_width)
        # No need to define observation space

        self._step = 0

    def get_obs(self, state: State) -> chex.Array:
        """
        Return observation includes:
            - The unit id to be deploy next.
            - The list of units still available for deployment, including the next unit id.
            - The current state of the deployed units on the battlefield.
        """

        enemy_battle_field = convert_unit_layer(state.enemy_battle_field)
        ally_battle_field = convert_unit_layer(state.battle_field)

        battle_field = jnp.vstack((enemy_battle_field, ally_battle_field))
        obs = jnp.concatenate((state.next_unit, state.remaining_units, battle_field.flatten()))

        return obs

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key):
        # TODO: implement random enemy deployment.
        state = State(
            next_unit=jnp.array([jnp.argmax(self.ally_unit_comp != 0) + 1]),
            remaining_units=self.ally_unit_comp,
            battle_field=jnp.zeros((self.field_height, self.field_width), dtype=jnp.float32),
            enemy_battle_field=jnp.flip(self.enemy_battle_field, axis=0),
            battle_field_mask=jnp.ones((self.field_height, self.field_width), dtype=jnp.bool_),
        )
        return self.get_obs(state), state

    @partial(jax.jit, static_argnums=(0,))
    def step(self, key, state, action):
        # Deploy the next unit
        h, w = action // self.field_width, action % self.field_width
        try_deploy = jnp.zeros_like(state.battle_field, dtype=jnp.bool_)
        try_deploy = try_deploy.at[h, w].set(True)

        # Get the occupied space size of the next unit
        space_occupied_spec = get_all_unit_spec()[-1]
        n = jnp.sqrt(space_occupied_spec[state.next_unit - 1][0]).astype(int)
        # Constrain deployment on contours by the next unit size
        contour_deploy = (
            jnp.zeros_like(try_deploy, dtype=jnp.bool_)
            .at[
                jnp.tile(jnp.arange(self.field_height - n + 1), self.field_width - n + 1),
                jnp.tile(jnp.arange(self.field_width - n + 1), self.field_height - n + 1),
            ]
            .set(True)
        )

        _cond_deploy = try_deploy & contour_deploy & state.battle_field_mask
        _deployed = state.battle_field.at[h, w].set(state.next_unit[0])
        battle_field = jnp.where(_cond_deploy, _deployed, state.battle_field)

        # Remove the deployed unit from the list
        _cond_unit = jax.nn.one_hot(state.next_unit - 1, state.remaining_units.size)
        remaining_units = jnp.where(_cond_unit, state.remaining_units - 1, state.remaining_units)

        # Get a mask for the next unit after deployment
        next_unit = jnp.array([jnp.argmax(remaining_units != 0) + 1])
        m = jnp.sqrt(space_occupied_spec[next_unit - 1][0]).astype(int)
        deployed_mask = jnp.where(
            state.battle_field > 0, True, False
        )  # Deployed units before deployment
        deploy_mask = jnp.logical_not(
            conv_lower_right_padding(
                try_deploy | deployed_mask, jnp.ones((m, m), dtype=jnp.bool_)
            ).astype(jnp.bool_)
        )
        # Constrain deployment on contours by the next unit size
        deploy_mask = deploy_mask.at[jnp.arange(self.field_width), self.field_width - m + 1 :].set(
            False
        )
        deploy_mask = deploy_mask.at[
            self.field_height - m + 1 :, jnp.arange(self.field_height)
        ].set(False)

        battle_field_mask = deploy_mask & state.battle_field_mask

        # Update state
        state = state.replace(
            next_unit=next_unit,
            remaining_units=remaining_units.flatten(),
            battle_field=battle_field,
            battle_field_mask=battle_field_mask,
        )

        # NOTE: Reward would be computed by the result of battle with this unit deployment.
        reward = None

        # Episode will continue until no more units can be deployed.
        done = jnp.where(state.remaining_units.sum(), 0, 1)

        return self.get_obs(state), state, reward, done, {}
