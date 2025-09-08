
# TABS Battle Simulator

This module provides a simple, vectorized battle simulation environment for multiple units, inspired by TABS. Implemented with JAX for efficient computation, it is designed for research and prototyping of multi-agent environments. The environment allows you to run battle simulations.

## Overview

The core components of the battle simulator are:

- **UnitStatus**: Stores the state and combat attributes of a unit (health, attack, cooldown, sight, etc.).
- **DefaultUnit**: Represents a unit in the environment, including its transform, physics, collider, team, and status.
- **GameManager**: Manages the global state, including reward, done flag, timestep, and target assignments.

## Getting Started

Below is an example of how to instantiate units and the game manager, update the target matrix, and perform an action:

```python
import jax.numpy as jnp
from src.tabs.tabs_battle_simulator.battle_simulator import DefaultUnit, GameManager, UnitStatus
from src.physics import Transform, RigidBody, CircleCollider


unit1 = DefaultUnit(
    transform=Transform(position=jnp.array([1.0, 2.0]), rotation=jnp.array([0.0])),
    rigidbody=RigidBody(mass=jnp.array([1.0]), velocity=jnp.array([0.0, 0.0]), acceleration=jnp.array([0.0, 0.0]), is_kinematic=jnp.array([False])),
    collider=CircleCollider(radius=jnp.array([1.0])),
    team=jnp.array([0]),
    pos_limit=jnp.array([-10.0, 10.0]),
    status=UnitStatus(id=jnp.array([0]), health=jnp.array([100.0]), attack_damage=jnp.array([10.0]), attack_range=jnp.array([1000.0]), attack_cooldown=jnp.array([0.0]), cooldown=jnp.array([0.0]), sight_angle=jnp.array([4 * jnp.pi]), sight_radius=jnp.array([10.0])),
    attacking=jnp.array([False])
)

unit2 = DefaultUnit(
    transform=Transform(position=jnp.array([3.0, 4.0]), rotation=jnp.array([jnp.pi / 4])),
    rigidbody=RigidBody(mass=jnp.array([1.0]), velocity=jnp.array([0.0, 0.0]), acceleration=jnp.array([0.0, 0.0]), is_kinematic=jnp.array([False])),
    collider=CircleCollider(radius=jnp.array([1.0])),
    team=jnp.array([1]),
    pos_limit=jnp.array([-10.0, 10.0]),
    status=UnitStatus(id=jnp.array([1]), health=jnp.array([100.0]), attack_damage=jnp.array([10.0]), attack_range=jnp.array([1.0]), attack_cooldown=jnp.array([0.0]), cooldown=jnp.array([0.0]), sight_angle=jnp.array([jnp.pi /2]), sight_radius=jnp.array([10.0])), 
    attacking=jnp.array([False])
)

unit3 = DefaultUnit(
    transform=Transform(position=jnp.array([2.0, 2.0]), rotation=jnp.array([jnp.pi / 2])),
    rigidbody=RigidBody(mass=jnp.array([1.0]), velocity=jnp.array([0.0, 0.0]), acceleration=jnp.array([0.0, 0.0]), is_kinematic=jnp.array([False])),
    collider=CircleCollider(radius=jnp.array([1.0])),
    team=jnp.array([1]),
    pos_limit=jnp.array([-10.0, 10.0]),
    status=UnitStatus(id=jnp.array([1]), health=jnp.array([100.0]), attack_damage=jnp.array([10.0]), attack_range=jnp.array([1.0]), attack_cooldown=jnp.array([0.0]), cooldown=jnp.array([0.0]), sight_angle=jnp.array([jnp.pi / 4]), sight_radius=jnp.array([10.0])),
    attacking=jnp.array([False])
)

game_manager = GameManager(
    reward=jnp.array([0.0]),
    done=jnp.array([False]),
    timestep=jnp.array([0]),
    target=jnp.array([-1])
)

objects = {
    'unit1' : unit1,
    'unit2' : unit2,
    'unit3' : unit3,
    'game_manager' : game_manager
}

objects['game_manager'] = objects['game_manager'].update_distance_matrix(objects)
objects['unit1'].act(objects, jnp.array([0.0]), config={
    'dt' : 1.0
})
```