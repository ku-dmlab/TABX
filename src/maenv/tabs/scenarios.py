import chex
from flax import struct
import jax.numpy as jnp
from src.maenv.tabs.units import UnitID


@struct.dataclass
class Scenario:
    budget: int
    enemy_unit_types: chex.Array


MAP_NAME_TO_SCENARIO = {
    "20farmers": Scenario(budget=1600, enemy_unit_types=jnp.zeros((20,))),
    "1theking": Scenario(budget=1600, enemy_unit_types=jnp.array([UnitID.TheKing])),
    "1mammoth_4archer": Scenario(
        budget=2000,
        enemy_unit_types=jnp.array(
            [UnitID.Mammoth, UnitID.Archer, UnitID.Archer, UnitID.Archer, UnitID.Archer]
        ),
    ),
}


def map_name_to_scenario(map_name: str):
    """Return pre-defined scenario"""
    return MAP_NAME_TO_SCENARIO[map_name]


def register_scenario(map_name: str, scenario: Scenario):
    MAP_NAME_TO_SCENARIO[map_name] = scenario
