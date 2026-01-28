## Quick Start

The following example shows how to run 5 parallel environments using JAX's vectorization:

```python
import jax
from src.tabx import TABX, build_batched_env_params_and_config
from src.tabx.wrappers.wrappers import TABXAutoResetWrapper, TABXEnemyHeuristicWrapper, TABXLogWrapper

# Build batched environments with challenge scenario (5 parallel environments)
env_params, config = build_batched_env_params_and_config(scenario_names="elbow", physics_param_names="default", heuristic_param_names="novice", n_repeat=5)
env = TABX(cfg=config)
env = TABXLogWrapper(env)
env = TABXEnemyHeuristicWrapper(env)
env = TABXAutoResetWrapper(env)

# Build batched environments with combined unit and zone scenarios (5 parallel environments)
env_params, config = build_batched_env_params_and_config(scenario_names="2F1M2Avs2S1K_2L", physics_param_names="default" heuristic_param_names="medium", n_repeat=5)
env = TABX(cfg=config)
env = TABXLogWrapper(env)
env = TABXEnemyHeuristicWrapper(env)
env = TABXAutoResetWrapper(env)

# Vectorized reset and step
v_reset = jax.vmap(env.reset, in_axes=(0, 0))
v_step = jax.vmap(env.step, in_axes=(0, 0, 0, 0))

rng = jax.random.PRNGKey(0)
obs, state = v_reset(jax.random.split(rng, 5), env_params)

# Run episode
actions = {agent: env.action_spaces[agent].sample(rng) for agent in env.ally_keys}
obs, state, rewards, dones, infos = v_step(rng_keys, state, actions, env_params)
```

## Visualization

Create animated visualizations of battle episodes:

```python
from src.tabx.visualize import Visualizer

# Collect episode states
state_seq = []  # List of environment states from each timestep

# Create and save visualization
visualizer = Visualizer(env, state_seq)
visualizer.animate(save_fname="battle.gif", view=False)
```
## Heuristic Parameters

Predefined heuristic opponent difficulty levels with configurable behavior:

- **Epsilon**: Probability of taking random actions (1.0 = fully random, 0.0 = fully deterministic)
- **Aggressive Threshold**: Distance threshold for ranged units to kite away from enemies (as ratio of attack range)
- **Healer Aggressive Threshold**: Distance threshold for healers to kite (as ratio of attack range)
- **Assassin Speed**: Speed threshold to identify assassin units (attack enemy backs)
- **Ranger Attack Range**: Range threshold to identify ranged units (enable kiting behavior)

| Level | Epsilon | Aggressive Threshold | Healer Aggressive Threshold | Assassin Speed | Ranger Attack Range |
|-------|---------|---------------------|----------------------------|----------------|-------------------|
| `random` | 1.0 | 0.0 | 0.85 | 1.4 | 10.0 |
| `novice` | 0.5 | 0.1 | 0.85 | 1.4 | 10.0 |
| `medium` | 0.2 | 0.3 | 0.85 | 1.4 | 10.0 |
| `advanced` | 0.1 | 0.5 | 0.85 | 1.4 | 10.0 |
| `expert` | 0.01 | 0.7 | 0.85 | 1.4 | 10.0 |

## Predefined Scenarios

TABX includes a variety of predefined challenge scenarios with different tactical situations:

### Challenge Scenarios

| Scenario | Snapshot |
|----------|----------|
| `ambush` | ![ambush](scenarios/snaps/challenges/ambush.png) |
| `bypass` | ![bypass](scenarios/snaps/challenges/bypass.png) |
| `clover` | ![clover](scenarios/snaps/challenges/clover.png) |
| `crossfire` | ![crossfire](scenarios/snaps/challenges/crossfire.png) |
| `elbow` | ![elbow](scenarios/snaps/challenges/elbow.png) |
| `encirclement` | ![encirclement](scenarios/snaps/challenges/encirclement.png) |
| `grid` | ![grid](scenarios/snaps/challenges/grid.png) |
| `pingpong` | ![pingpong](scenarios/snaps/challenges/pingpong.png) |
| `ribbon` | ![ribbon](scenarios/snaps/challenges/ribbon.png) |
| `superking` | ![superking](scenarios/snaps/challenges/superking.png) |
| `vsrangers` | ![vsrangers](scenarios/snaps/challenges/vsrangers.png) |

### Zone Scenarios

| Scenario | Snapshot |
|----------|----------|
| `1S` | ![1S](scenarios/snaps/zones/1S.png) |
| `2L` | ![2L](scenarios/snaps/zones/2L.png) |
| `2L2B2S` | ![2L2B2S](scenarios/snaps/zones/2L2B2S.png) |
| `3B` | ![3B](scenarios/snaps/zones/3B.png) |

### Unit Scenarios

| Scenario | Snapshot |
|----------|----------|
| `1F1M3A1Hvs2F1S1K1A1H` | ![1F1M3A1Hvs2F1S1K1A1H](scenarios/snaps/units/1F1M3A1Hvs2F1S1K1A1H.png) |
| `2F1M2Avs2S1K` | ![2F1M2Avs2S1K](scenarios/snaps/units/2F1M2Avs2S1K.png) |
| `4F1S1K2A1Pvs2M1C1P` | ![4F1S1K2A1Pvs2M1C1P](scenarios/snaps/units/4F1S1K2A1Pvs2M1C1P.png) |
| `5F1S1A1Dvs7F1S1D1H` | ![5F1S1A1Dvs7F1S1D1H](scenarios/snaps/units/5F1S1A1Dvs7F1S1D1H.png) |

