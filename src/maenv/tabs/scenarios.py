import chex
from flax import struct
import jax.numpy as jnp


@struct.dataclass
class Scenario:
    budget: int
    enemy_type_comp: chex.Array


MAP_NAME_TO_SCENARIO = {
    "20farmers": Scenario(budget=1600, enemy_type_comp=jnp.array([20, 0, 0, 0, 0, 0, 0])),
    "1theking": Scenario(budget=1600, enemy_type_comp=jnp.array([0, 0, 1, 0, 0, 0, 0])),
    "4archer_1mammoth": Scenario(
        budget=2000,
        enemy_type_comp=jnp.array([0, 4, 0, 0, 1, 0, 0]),
    ),
}


def map_name_to_scenario(map_name: str):
    """Return pre-defined scenario"""
    return MAP_NAME_TO_SCENARIO[map_name]


def register_scenario(map_name: str, scenario: Scenario):
    MAP_NAME_TO_SCENARIO[map_name] = scenario
