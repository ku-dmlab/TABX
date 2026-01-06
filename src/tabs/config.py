from dataclasses import dataclass

import chex
import jax.numpy as jnp
from flax import struct

from src.tabs.constants import ALL_UNIT_NAMES


@dataclass(frozen=True)
class TABSConfig:
    max_n_ally: int = 10  # The maximum number of ally agents
    max_n_enemy: int = 10  # The maximum number of enemy agents
    max_n_zone: int = 4  # The maximum number of zone


@struct.dataclass
class TABSHeuristicConfig:
    epsilon: chex.Array = struct.field(
        default_factory=lambda: jnp.array([0.1])
    )  # Probability of taking random action inheuristic policy (0.0-1.0)
    aggressive_threshold: chex.Array = struct.field(
        default_factory=lambda: jnp.array([0.3])
    )  # Threshold for aggressive behavior in heuristic policy (0.0-1.0)
    rotate_noise_scale: chex.Array = struct.field(default_factory=lambda: jnp.array([0.5]))
    healer_rotate_noise_scale: chex.Array = struct.field(default_factory=lambda: jnp.array([0.1]))
    healer_aggressive_threshold: chex.Array = struct.field(
        default_factory=lambda: jnp.array([0.85])
    )
    assasin_speed: chex.Array = struct.field(
        default_factory=lambda: jnp.array([1.4])
    )  # Threshold for determining assassin unit which has high speed
    ranger_attack_range: chex.Array = struct.field(
        default_factory=lambda: jnp.array([10.0])
    )  # Attack range of ranger
