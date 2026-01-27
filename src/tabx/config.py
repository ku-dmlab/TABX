from dataclasses import dataclass

import chex
import jax.numpy as jnp
from flax import struct

from src.tabx.constants import ALL_UNIT_NAMES


@dataclass(frozen=True)
class TABXConfig:
    max_n_ally: int = 10  # The maximum number of ally agents
    max_n_enemy: int = 10  # The maximum number of enemy agents
    max_n_zone: int = 4  # The maximum number of zone
