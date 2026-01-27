import chex
import jax.numpy as jnp
from flax import struct


@struct.dataclass
class TABXHeuristicParam:
    epsilon: chex.Array = struct.field(
        default_factory=lambda: jnp.array([0.1])
    )  # Probability of taking random action inheuristic policy (0.0-1.0)
    aggressive_threshold: chex.Array = struct.field(
        default_factory=lambda: jnp.array([0.3])
    )  # Threshold for aggressive behavior in heuristic policy (0.0-1.0)
    healer_aggressive_threshold: chex.Array = struct.field(
        default_factory=lambda: jnp.array([0.85])
    )
    assasin_speed: chex.Array = struct.field(
        default_factory=lambda: jnp.array([1.4])
    )  # Threshold for determining assassin unit which has high speed
    ranger_attack_range: chex.Array = struct.field(
        default_factory=lambda: jnp.array([10.0])
    )  # Attack range of ranger
