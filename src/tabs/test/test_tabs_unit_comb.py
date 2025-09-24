import unittest

import jax
import jax.numpy as jnp

from src.tabs.tabs_unit_comb.tabs_unit_comb import TABSUnitComb
from src.tabs.units import get_all_unit_names
from src.tabs.scenarios import TABSConfig, Scenario, generate_scenario, default_tabs_conf


class TestTABSUnitComb(unittest.TestCase):
    def test_reset(self):
        self.reset_predefined_scenario1("10F")
        self.reset_predefined_scenario1("1K")
        self.reset_predefined_scenario1("4A1M")

    def test_purchase(self):
        rng = jax.random.PRNGKey(0)
        rng, _rng = jax.random.split(rng)
        self.purchase1(_rng)
        rng, _rng = jax.random.split(rng)
        self.purchase2(_rng)
        rng, _rng = jax.random.split(rng)
        self.purchase3(rng)

    def reset_predefined_scenario1(self, scenario_name):
        tabs_conf = TABSConfig(
            scenario_name=scenario_name,
            max_agents=10,
            max_num_units=len(get_all_unit_names()),
            max_field_height=4,
            max_field_width=5,
        )
        env = TABSUnitComb(tabs_conf)
        scenario = generate_scenario(tabs_conf)

        o, s = env.reset(0, scenario)
        expected_obs = jnp.concatenate(
            (
                jnp.array([scenario.budget]),
                jnp.zeros(tabs_conf.max_num_units),
                scenario.price,
                scenario.enemy_unit_comp,
            )
        )
        self.assertTrue(jnp.array_equal(o, expected_obs))

    def purchase1(self, rng):
        env = TABSUnitComb(default_tabs_conf)
        scenario = generate_scenario(default_tabs_conf)

        rng, _rng = jax.random.split(rng)
        o, s = env.reset(rng, scenario)
        action = jnp.array(3)  # buy BombThrower
        o, s, _, _, _ = env.step(_rng, s, action)

        self.assertTrue(
            s.budget == scenario.budget - scenario.price[action],
            f"\ntrue: {scenario.budget - scenario.price[action]}\nout: {s.budget}",
        )
        true = jnp.array([0, 0, 0, 1, 0, 0, 0], dtype=jnp.int32)
        self.assertTrue(
            jnp.array_equal(s.current_unit_list, true),
            f"\ntrue: {true}\nout: {s.current_unit_list}",
        )
        true = jnp.array([1, 1, 0, 1, 0, 1, 1], dtype=jnp.float32)
        self.assertTrue(
            jnp.array_equal(s.action_mask, true),
            f"\ntrue: {true}\nout: {s.action_mask}",
        )

    def purchase2(self, rng):
        env = TABSUnitComb(default_tabs_conf)
        scenario = generate_scenario(default_tabs_conf)

        rng, _rng = jax.random.split(rng)
        o, init_s = env.reset(rng, scenario)
        action = jnp.array(4)
        o, s, _, _, _ = env.step(_rng, init_s, action)

        self.assertTrue(s.budget == scenario.budget, f"\ntrue: {scenario.budget}\nout: {s.budget}")
        self.assertTrue(
            jnp.array_equal(s.current_unit_list, init_s.current_unit_list),
            f"\ntrue: {init_s.current_unit_list}\nout: {s.current_unit_list}",
        )
        self.assertTrue(
            jnp.array_equal(s.action_mask, init_s.action_mask),
            f"\ntrue: {init_s.action_mask}\nout: {s.action_mask}",
        )

    def purchase3(self, rng):
        tabs_conf = default_tabs_conf
        tabs_conf = tabs_conf.replace(max_agents=1)
        env = TABSUnitComb(tabs_conf)
        scenario = generate_scenario(tabs_conf)

        rng, _rng, _rng1 = jax.random.split(rng, 3)
        o, init_s = env.reset(rng, scenario)

        action = jnp.array(0)
        o, s1, _, d, _ = env.step_env(_rng, init_s, action)
        self.assertFalse(d.astype(jnp.bool_))
        o, s, _, d, _ = env.step_env(_rng1, s1, action)
        self.assertTrue(d.astype(jnp.bool_))

        self.assertTrue(
            s1.budget == s.budget,
            f"\ntrue: {s1.budget}\nout: {s.budget}",
        )
        self.assertTrue(
            jnp.array_equal(s1.current_unit_list, s.current_unit_list),
            f"\ntrue: {s1.current_unit_list}\nout: {s.current_unit_list}",
        )
        self.assertTrue(
            jnp.array_equal(s.action_mask, jnp.zeros_like(s.action_mask)),
            f"\ntrue: {jnp.zeros_like(s.action_mask)}\nout: {s.action_mask}",
        )


if __name__ == "__main__":
    """
    refer: https://docs.python.org/3/library/unittest.html
    cli: python -m unittest test_tabs_unit_comb
    """

    unittest.main()
