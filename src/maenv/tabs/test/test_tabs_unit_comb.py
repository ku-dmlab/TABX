import unittest

import jax.numpy as jnp

from src.maenv.tabs.tabs_unit_comb.tabs_unit_comb import TABSUnitComb
from src.maenv.tabs.units import get_all_unit_names
from src.maenv.tabs.scenarios import TABSConf, Scenario, generate_scenario


class TestTABSUnitComb(unittest.TestCase):
    def test_reset(self):
        self.reset_predefined_scenario1()

    def reset_predefined_scenario1(self):
        tabs_conf = TABSConf(
            scenario_name="4archer_1mammoth",
            max_agents=20,
            max_num_units=len(get_all_unit_names()),
            max_field_height=4,
            max_field_width=5,
            scenario=None,
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


if __name__ == "__main__":
    """
    refer: https://docs.python.org/3/library/unittest.html
    cli: python -m unittest test_tabs_unit_comb
    """

    unittest.main()
