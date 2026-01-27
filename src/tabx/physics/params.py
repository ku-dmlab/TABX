import jax.numpy as jnp
from flax import struct


@struct.dataclass
class PhysicsParams:
    dt: float = struct.field(default_factory=lambda: jnp.array([0.5]))
    percent: float = struct.field(default_factory=lambda: jnp.array([0.5]))
    slop: float = struct.field(default_factory=lambda: jnp.array([0.01]))
    restitution: float = struct.field(default_factory=lambda: jnp.array([0.8]))
