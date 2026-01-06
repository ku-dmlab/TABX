from collections import namedtuple

import jax.numpy as jnp


class BoxCollider(namedtuple("BoxCollider", ["width", "height"])):
    width: float
    height: float


class CircleCollider(namedtuple("CircleCollider", ["radius"])):
    radius: float


class Transform(namedtuple("Transform", ["position", "rotation"])):
    position: jnp.array
    rotation: jnp.array


class Ellipse(namedtuple("Ellipse", ["position", "axes"])):
    position: jnp.array
    axes: jnp.array


class RigidBody(namedtuple("RigidBody", ["mass", "velocity", "acceleration", "is_kinematic"])):
    velocity: jnp.array
    mass: jnp.array
    acceleration: jnp.array
    is_kinematic: jnp.array

    def __new__(
        cls,
        mass,
        velocity=jnp.array([0.0, 0.0]),
        acceleration=jnp.array([0.0, 0.0]),
        is_kinematic=jnp.array([False]),
    ):
        return super().__new__(cls, mass, velocity, acceleration, is_kinematic)

    def update(self, config):
        velocity = self.velocity + self.acceleration * config.dt
        return self._replace(velocity=velocity * (1 - self.is_kinematic))
