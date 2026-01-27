from dataclasses import dataclass

import chex
import jax.numpy as jnp
from flax import struct


@struct.dataclass
class LastVisibleTarget:
    abs_position: chex.Array = jnp.array([0.0, 0.0])
    ever_visible: chex.Array = jnp.array([False])


@dataclass
class OwnState:
    health: chex.Array
    max_health_ratio: chex.Array
    position: chex.Array
    rotation: chex.Array
    attack_range: chex.Array
    attack_damage: chex.Array
    cooldown_ratio: chex.Array
    radius: chex.Array
    speed: chex.Array
    sight_angle: chex.Array

    @property
    def is_healer(self):
        return self.attack_damage < 0

    @property
    def is_on_cooldown(self):
        return self.cooldown_ratio < 1.0

    @classmethod
    def from_obs(cls, obs: chex.Array) -> "OwnState":
        own = obs[:14]

        return cls(
            health=own[0],
            max_health_ratio=own[1],
            position=own[2:4],
            rotation=own[4] * 2 * jnp.pi,
            attack_range=own[5],
            attack_damage=own[6],
            cooldown_ratio=own[8],
            radius=own[9],
            speed=own[13],
            sight_angle=own[11],
        )


@dataclass
class ZoneState:
    zone_type: chex.Array
    position: chex.Array
    axes: chex.Array
    damage: chex.Array

    @classmethod
    def from_obs(cls, obs: chex.Array, num_agents: int, num_zones: int) -> "OtherState":
        if num_zones == 0:
            return cls(
                zone_type=jnp.array([[0]]),
                position=jnp.array([[0.0, 0.0]]),
                axes=jnp.array([[0.0, 0.0]]),
                damage=jnp.array([[0.0]]),
            )
        other = obs[14 + (num_agents - 1) * 16 :].reshape(num_zones, -1)

        return cls(
            zone_type=other[:, 0],
            position=other[:, 1:3],
            axes=other[:, 3:5],
            damage=other[:, 5],
        )


@dataclass
class OtherState:
    health: chex.Array
    health_ratio: chex.Array
    rel_pos: chex.Array
    rotation: chex.Array
    attack_range: chex.Array
    attack_damage: chex.Array
    is_ally: chex.Array
    is_attackable: chex.Array
    max_hp: chex.Array
    radius: chex.Array
    speed: chex.Array

    @property
    def is_alive(self):
        return self.health > 0

    @classmethod
    def from_obs(cls, obs: chex.Array, num_agents: int) -> "OtherState":
        other = obs[14 : 14 + (num_agents - 1) * 16].reshape(num_agents - 1, -1)

        return cls(
            health=other[:, 0],
            health_ratio=other[:, 1],
            rel_pos=other[:, 2:4],
            rotation=other[:, 4] * 2 * jnp.pi,
            attack_range=other[:, 5],
            attack_damage=other[:, 6],
            is_ally=other[:, 13].astype(jnp.bool_),
            is_attackable=other[:, 14].astype(jnp.bool_),
            radius=other[:, 9],
            speed=other[:, 15],
            max_hp=jnp.where(
                other[:, 13].astype(jnp.bool_) | (other[:, 0] == 0),
                jnp.inf,
                other[:, 0]
                / other[:, 1],  # hp / normalized_hp = max_hp since normalized_hp = hp / max_hp
            ),
        )


@chex.dataclass
class ParsedObservation:
    own: OwnState
    other: OtherState
    zone: ZoneState

    @classmethod
    def from_obs(cls, obs: chex.Array, num_agents: int, num_zones: int) -> "ParsedObservation":
        return cls(
            own=OwnState.from_obs(obs),
            other=OtherState.from_obs(obs, num_agents),
            zone=ZoneState.from_obs(obs, num_agents, num_zones),
        )
