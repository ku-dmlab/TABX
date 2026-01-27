# Totally Accelerated Battle Simulator (TABX)
**Totally Accelerated Battle Simulator (TABX)**, is inspired by the popular strategic simulation game Totally Accurate Battle Simulator (2021, Landfall Games). TABX provides a complex, multi-stage environment suite implemented in JAX, designed for accelerated training and scalable experimentation.

TABX is structured into three sequential stages:
- `TABXUnitComb`: Agents select unit compositions under a given budget.
- `TABXUnitDeploy`: The chosen units are spatially arranged on the battlefield.
- `TABXBattleSimulator`: Agents control units in real-time combat.

## Baseline Algorithms
We provide implementations of four reinforcement learning algorithms, available in the `baseline` directory. Configuration files are managed using Tyro.

Every files include `wandb` logging by default. Logging can be disabled through the configuration file.

## Installation
We recommend using the provided Docker devcontainer and uv Python package to ensure a consistent development environment.
```
# Create a virtual environment
$ uv venv

# Activate the environment
$ uv sync
```