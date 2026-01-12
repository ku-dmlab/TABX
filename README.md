# Totally Accelerated Battle Simulator in JAX (TABX)
**Totally Accelerated Battle Simulator in JAX (TABX)** is a rapid, flexible, and easily configurable sandbox for generalization in MARL. It allows researchers to generate various scenarios tailored to specific research objectives by offering a diverse set of environmental parameters.

We recommend using the provided Docker devcontainer and uv Python package to ensure a consistent development environment.
```
# Create a virtual environment
$ uv venv

# Activate the environment
$ uv sync
```

## Observation and Action Spaces
Agents possess a partial, **fan-shaped** observation field aligned with their heading, providing a perspective analogous to a first-person view. Within this field, agents observe the status (e.g., remaining health) and attributes of all visible units and zones.

Based on these observations, agents select from six discrete actions: four directional movements, an attack/heal action, and a rotation. The attack/heal action is only available once the unit’s cooldown has expired.

```python
class UnitAction:
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3
    ATTACK = 4
    TURN_RIGHT = 5
    TURN_LEFT = 6
    IDLE = 7
```

## Non-targeting Mechanism and Zones
A distinguishing aspect of TABX is its unit interaction system, which utilizes non-targeted attack and healing mechanisms. Unlike many simulators that rely on explicit targeting, TABX agents execute actions within a spatial hurtbox—an interaction occurs only if the recipient is positioned within this defined field.

We incorporate three environmental zones (lava, bush, and swamp). These zones dynamically modulate agent attributes, such as health and velocity, or introduce asymmetric visibility between units, forcing agents to adapt their tactics based on the terrain.

| Zone | Effect Category| Technical Impact|
|---|---|---|
|Lava |Health |Continuous damage (HP depletion)|
|Bush |Visibility |Asymmetric concealment (Stealth)|
|Swamp |Velocity |Movement speed penalty (Slow)|

## Rewards
An episode terminates when all units of one team are incapacitated. At this point, a global binary reward is issued to all agents:
- Win (+1): All enemy units eliminated.
- Lose (-1): All ally units eliminated.

## Baseline Algorithms
We provide implementations of five MARL algorithms and five UED algorithms, available in the `baseline` directory.
| Category | Algorithm | Reference |
| --- | --- | --- |
| MARL | IPPO | De Witt et al. (2020) |
| | MAPPO | Yu et al. (2022) |
| | IQL | Tampuu et al. (2017) |
| | VDN | Sunehang et al. (2017) |
| | QMIX | Rashid et al. (2020) |
| UED | DR | Jakobi (1997); Sadeghi & Levien (2016) |
| | PLR | Jiang et al. (2021a) |
| | Robust PLR | Jiang et al. (2021b) |
| | ACCEL | Parker-Holder et al. (2022) |
| | SFL | Rutherford et al. (2024a) |

Configuration files are managed using Tyro. Every files include `wandb` logging by default. Logging can be disabled through the configuration file.