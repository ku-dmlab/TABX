
# Totally Accelerated Battle Simulator (TABS)
This environment provides a simple, vectorized battle simulation environment for multiple units, inspired by [TABS](https://store.steampowered.com/app/508440/Totally_Accurate_Battle_Simulator/). Implemented with JAX for efficient computation, it is designed for research and prototyping of multi-agent environments. The environment allows you to run battle simulations.

## Overview

The core components of the battle simulator are:

- **UnitStatus**: Stores the state and combat attributes of a unit (health, attack, cooldown, sight, etc.).
- **DefaultUnit**: Represents a unit in the environment, including its transform, physics, collider, team, and status.
- **GameManager**: Manages the global state, including reward, done flag, timestep, and target assignments.

## Getting Started
Begin by instantiating a `TABSConfig`, generating a scenario, and initializing the base environment:
```python
import jax

from src.tabs import TABS
from src.tabs.scenarios import generate_scenario
from src.tabs.config import TABSConfig, PhysicsParams, TABSHeuristicParam
from src.tabs.wrappers.wrappers import (
    TABSEnemyHeuristicWrapper,
    TABSAutoResetWrapper,
    TABSLogWrapper,
)

tabs_config = TABSConfig(scenario_name="2F1K2A1H_tight")
scenario = generate_scenario(tabs_config)

env = TABS(cfg=tabs_config)
env = TABSLogWrapper(env)
env = TABSEnemyHeuristicWrapper(env)
env = TABSAutoResetWrapper(env)
```

Some components require parameters during `env.reset()`. Create an env_params dictionary:
```python
env_params = {
    "scenario": scenario,
    "physics_params": PhysicsParams(),
    "heuristic_params": TABSHeuristicParam(),
}
```
You can then pass these parameters directly during environment reset,
```python
obs, env_state = env.reset(jax.random.PRNGKey(0), env_params)
```

A ready-to-run example is included:
```
uv run run_example.py
```
This script demonstrates the full setup, including environment creation, wrapper application, and a minimal control loop.