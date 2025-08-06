import chex
from flax import struct
import jax.numpy as jnp


@struct.dataclass
class Scenario:
    budget: int
    enemy_unit_comp: chex.Array
    ally_unit_comp: chex.Array
    enemy_battle_field: chex.Array
    field_height: int
    field_width: int


MAP_NAME_TO_SCENARIO = {
    "20farmers": Scenario(
        budget=1600,
        enemy_unit_comp=jnp.array([20, 0, 0, 0, 0, 0, 0]),
        ally_unit_comp=jnp.array([20, 0, 0, 0, 0, 0, 0]),
        enemy_battle_field=jnp.ones((4, 5), dtype=jnp.float32),
        field_height=4,
        field_width=5,
    ),
    "1theking": Scenario(
        budget=1600,
        enemy_unit_comp=jnp.array([0, 0, 1, 0, 0, 0, 0]),
        ally_unit_comp=jnp.array([0, 0, 1, 0, 0, 0, 0]),
        enemy_battle_field=jnp.concatenate(
            (jnp.array([[0, 0, 3, 0, 0]]), jnp.zeros((3, 5), dtype=jnp.float32)), axis=0
        ),
        field_height=4,
        field_width=5,
    ),
    "4archer_1mammoth": Scenario(
        budget=2000,
        enemy_unit_comp=jnp.array([0, 4, 0, 0, 1, 0, 0]),
        ally_unit_comp=jnp.array([0, 4, 0, 0, 1, 0, 0]),
        enemy_battle_field=jnp.array(
            [[0, 0, 5, 0, 0], [0, 2, 0, 0, 2], [2, 0, 0, 0, 2], [0, 0, 0, 0, 0]], dtype=jnp.float32
        ),
        field_height=4,
        field_width=5,
    ),
}


def map_name_to_scenario(map_name: str):
    """Return pre-defined scenario"""
    return MAP_NAME_TO_SCENARIO[map_name]


def register_scenario(map_name: str, scenario: Scenario):
    MAP_NAME_TO_SCENARIO[map_name] = scenario
