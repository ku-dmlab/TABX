import unittest

import jax
import jax.numpy as jnp

from src.maenv.tabs.tabs_unit_deploy.tabs_unit_deploy import TABSUnitDeploy
from src.maenv.tabs.units import get_all_unit_names
from src.maenv.tabs.scenarios import TABSConf, Scenario, generate_scenario, default_tabs_conf


class TestTABSUnitDeploy(unittest.TestCase):
    def test_reset(self):
        self.reset_predefined_scenario1("20farmers")
        self.reset_predefined_scenario1("1theking")
        self.reset_predefined_scenario1("4archer_1mammoth")

    def test_select_action(self):
        rng = jax.random.PRNGKey(0)
        rng, _rng = jax.random.split(rng)
        self.select_action1(_rng)
        rng, _rng = jax.random.split(rng)
        self.select_same_action1(_rng)

    def test_space_occupied_for_next_unit(self):
        rng = jax.random.PRNGKey(0)
        rng, _rng = jax.random.split(rng)
        self.space_occupied_for_next_unit1(rng)
        rng, _rng = jax.random.split(rng)
        self.space_occupied_for_next_unit2(rng)
        rng, _rng = jax.random.split(rng)
        self.space_occupied_for_next_unit3(rng)

    def test_space_occupied(self):
        rng = jax.random.PRNGKey(0)
        rng, _rng = jax.random.split(rng)
        self.space_occupied1(_rng)
        rng, _rng = jax.random.split(rng)
        self.space_occupied2(_rng)

    def reset_predefined_scenario1(self, scenario_name):
        tabs_conf = TABSConf(
            scenario_name=scenario_name,
            max_agents=20,
            max_num_units=len(get_all_unit_names()),
            max_field_height=4,
            max_field_width=5,
        )
        env = TABSUnitDeploy(tabs_conf)
        scenario = generate_scenario(tabs_conf)

        o, s = env.reset(0, scenario)
        self.assertEqual(env.observation_space.shape, o.shape)

    def select_same_action1(self, rng):
        env = TABSUnitDeploy(default_tabs_conf)
        scenario = generate_scenario(default_tabs_conf)

        rng1, rng2, rng3, rng4 = jax.random.split(rng, 4)
        o, s = env.reset(rng1, scenario)
        action = jnp.array(0)
        o, s, _, _, _ = env.step(rng2, s, action)
        o, s, _, _, _ = env.step(rng3, s, action)
        o, s, _, _, _ = env.step(rng4, s, action)

        expected_battle_field = jnp.zeros(
            (default_tabs_conf.max_field_height, default_tabs_conf.max_field_width)
        )
        expected_battle_field = expected_battle_field.at[0, 0].set(1)
        self.assertTrue(
            jnp.array_equal(s.battle_field, expected_battle_field),
            f"\ntrue:\n{expected_battle_field}\nout:\n{s.battle_field}",
        )

        expected_battle_field_mask = jnp.logical_not(expected_battle_field).astype(jnp.float32)
        self.assertTrue(
            jnp.array_equal(s.battle_field_mask, expected_battle_field_mask),
            f"\ntrue:\n{expected_battle_field_mask}\nout:\n{s.battle_field_mask}",
        )

    def select_action1(self, rng):
        env = TABSUnitDeploy(default_tabs_conf)
        scenario = generate_scenario(default_tabs_conf)

        rng1, rng2, rng3 = jax.random.split(rng, 3)
        o, s = env.reset(rng1, scenario)
        action = jnp.array(0)
        o, s, _, _, _ = env.step(rng2, s, action)
        action = jnp.array(1)
        o, s, _, _, _ = env.step(rng3, s, action)

        expected_battle_field = jnp.zeros(
            (default_tabs_conf.max_field_height, default_tabs_conf.max_field_width)
        )
        expected_battle_field = expected_battle_field.at[0, :2].set(1)
        self.assertTrue(
            jnp.array_equal(s.battle_field, expected_battle_field),
            f"\ntrue:\n{expected_battle_field}\nout:\n{s.battle_field}",
        )

        expected_battle_field_mask = jnp.logical_not(expected_battle_field).astype(jnp.float32)
        self.assertTrue(
            jnp.array_equal(s.battle_field_mask, expected_battle_field_mask),
            f"\ntrue:\n{expected_battle_field_mask}\nout:\n{s.battle_field_mask}",
        )

    def space_occupied_for_next_unit1(self, rng):
        """archer -> mammoth"""
        env = TABSUnitDeploy(default_tabs_conf)
        scenario = generate_scenario(default_tabs_conf)
        scenario = scenario.replace(ally_unit_comp=jnp.array([0, 1, 0, 0, 1, 0, 0]))

        rng1, rng2 = jax.random.split(rng)
        o, s = env.reset(rng1, scenario)
        action = jnp.array(3)
        o, s, _, _, _ = env.step(rng2, s, action)

        expected_battle_field_mask = jnp.array(
            [[1, 1, 0, 0, 0], [1, 1, 1, 1, 0], [1, 1, 1, 1, 0], [0, 0, 0, 0, 0]]
        ).astype(jnp.float32)

        self.assertTrue(
            jnp.array_equal(s.battle_field_mask, expected_battle_field_mask),
            f"\ntrue:\n{expected_battle_field_mask}\nout:\n{s.battle_field_mask}",
        )

    def space_occupied_for_next_unit2(self, rng):
        """mammoth -> mammoth"""
        env = TABSUnitDeploy(default_tabs_conf)
        scenario = generate_scenario(default_tabs_conf)
        scenario = scenario.replace(ally_unit_comp=jnp.array([0, 0, 0, 0, 2, 0, 0]))

        rng1, rng2 = jax.random.split(rng)
        o, s = env.reset(rng1, scenario)
        action = jnp.array(3)
        o, s, _, _, _ = env.step(rng2, s, action)

        expected_battle_field_mask = jnp.array(
            [[1, 1, 0, 0, 0], [1, 1, 0, 0, 0], [1, 1, 1, 1, 0], [0, 0, 0, 0, 0]]
        ).astype(jnp.float32)

        self.assertTrue(
            jnp.array_equal(s.battle_field_mask, expected_battle_field_mask),
            f"\ntrue:\n{expected_battle_field_mask}\nout:\n{s.battle_field_mask}",
        )

    def space_occupied_for_next_unit3(self, rng):
        """mammoth -> deadeye"""
        env = TABSUnitDeploy(default_tabs_conf)
        scenario = generate_scenario(default_tabs_conf)
        scenario = scenario.replace(ally_unit_comp=jnp.array([0, 0, 0, 0, 1, 1, 0]))

        rng1, rng2 = jax.random.split(rng)
        o, s = env.reset(rng1, scenario)
        action = jnp.array(7)
        o, s, _, _, _ = env.step(rng2, s, action)

        expected_battle_field_mask = jnp.array(
            [[1, 1, 1, 1, 1], [1, 1, 0, 0, 1], [1, 1, 0, 0, 1], [1, 1, 1, 1, 1]]
        ).astype(jnp.float32)

        self.assertTrue(
            jnp.array_equal(s.battle_field_mask, expected_battle_field_mask),
            f"\ntrue:\n{expected_battle_field_mask}\nout:\n{s.battle_field_mask}",
        )

    def space_occupied_for_next_unit4(self, rng):
        """farmmer -> farmmer"""
        env = TABSUnitDeploy(default_tabs_conf)
        scenario = generate_scenario(default_tabs_conf)
        scenario = scenario.replace(ally_unit_comp=jnp.array([2, 0, 0, 0, 0, 0, 0]))

        rng1, rng2 = jax.random.split(rng)
        o, s = env.reset(rng1, scenario)
        action = jnp.array(7)
        o, s, _, _, _ = env.step(rng2, s, action)

        expected_battle_field_mask = jnp.array(
            [[1, 1, 1, 1, 1], [1, 1, 0, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1]]
        ).astype(jnp.float32)

        self.assertTrue(
            jnp.array_equal(s.battle_field_mask, expected_battle_field_mask),
            f"\ntrue:\n{expected_battle_field_mask}\nout:\n{s.battle_field_mask}",
        )

    def space_occupied1(self, rng):
        env = TABSUnitDeploy(default_tabs_conf)
        scenario = generate_scenario(default_tabs_conf)
        scenario = scenario.replace(ally_unit_comp=jnp.array([0, 0, 0, 0, 2, 0, 0]))

        rng1, rng2, rng3 = jax.random.split(rng, 3)
        o, s = env.reset(rng1, scenario)
        action = jnp.array(0)
        o, s, _, _, _ = env.step(rng2, s, action)
        action = jnp.array(1)
        o, s, _, _, _ = env.step(rng3, s, action)

        expected_battle_field = jnp.zeros(
            (default_tabs_conf.max_field_height, default_tabs_conf.max_field_width)
        )
        expected_battle_field = expected_battle_field.at[0, 0].set(5).astype(jnp.float32)
        self.assertTrue(
            jnp.array_equal(s.battle_field, expected_battle_field),
            f"\ntrue:\n{expected_battle_field}\nout:\n{s.battle_field}",
        )

        expected_battle_field_mask = jnp.array(
            [[0, 0, 1, 1, 0], [0, 0, 1, 1, 0], [1, 1, 1, 1, 0], [0, 0, 0, 0, 0]], dtype=jnp.float32
        )

        self.assertTrue(
            jnp.array_equal(s.battle_field_mask, expected_battle_field_mask),
            f"\ntrue:\n{expected_battle_field_mask}\nout:\n{s.battle_field_mask}",
        )

    def space_occupied2(self, rng):
        env = TABSUnitDeploy(default_tabs_conf)
        scenario = generate_scenario(default_tabs_conf)
        scenario = scenario.replace(ally_unit_comp=jnp.array([0, 0, 0, 0, 2, 0, 3]))

        o, s = env.reset(0, scenario)
        actions = jnp.array([0, 11, 1, 4])
        rngs = jax.random.split(rng, len(actions))
        for i in range(len(actions)):
            action = actions[i]
            o, s, _, _, _ = env.step(rngs[i], s, action)

        expected_battle_field = jnp.zeros(
            (default_tabs_conf.max_field_height, default_tabs_conf.max_field_width)
        )
        expected_battle_field = jnp.array(
            [[5, 0, 0, 0, 7], [0, 0, 0, 0, 0], [0, 5, 0, 0, 0], [0, 0, 0, 0, 0]], dtype=jnp.float32
        )
        self.assertTrue(
            jnp.array_equal(s.battle_field, expected_battle_field),
            f"\ntrue:\n{expected_battle_field}\nout:\n{s.battle_field}",
        )

        expected_battle_field_mask = jnp.array(
            [[0, 0, 1, 1, 0], [0, 0, 1, 1, 1], [1, 0, 0, 1, 1], [1, 0, 0, 1, 1]], dtype=jnp.float32
        )

        self.assertTrue(
            jnp.array_equal(s.battle_field_mask, expected_battle_field_mask),
            f"\ntrue:\n{expected_battle_field_mask}\nout:\n{s.battle_field_mask}",
        )


if __name__ == "__main__":
    """
    refer: https://docs.python.org/3/library/unittest.html
    cli: python -m unittest test_tabs_unit_deploy
    """

    unittest.main()
