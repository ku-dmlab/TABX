# TABSUnitDeploy Environment

TABSUnitDeploy is an environment implemented in JAX for evaluating agents in unit deployment tasks. The goal is to determine the optimal placement of units for a given scenario. The agent must understand strategic positioning and assess enemy strength to decide where to deploy the next unit from the available unit list.


The environment is initialized with a scenario that defines the battlefield size and enemy deployment. If no scenario is specified, a default configuration is used: 10 farmers on a 5 $\times$ 4 battlefield. Upon initialization, the environment returns an observation containing:
- The ID of the unit to be deployed next
- A list of units remaining for deployment
- The current state of the battlefield with deployed units.

The agent is expected to output a position for the next unit to be deployed. The episode ends when all available units have been placed.

## Getting Started
```python
import jax
from src.maenv.tabs.tabs_unit_deploy.tabs_unit_deploy import TABSUnitDeploy
from src.maenv.tabs.scenarios import MAP_NAME_TO_SCENARIO

def run(scenario_name, seed):
    scenario = MAP_NAME_TO_SCENARIO[scenario_name]
    env = TABSUnitComb(scenario=scenario)

    rng = jax.random.PRNGKey(seed)

    _rng, rng = jax.random.split(rng)
    obs, env_state = env.reset(rng)

    _rng, rng = jax.random.split(rng)
    action = env.action_space.sample(rng)

    _rng, rng = jax.random.split(rng)
    obs, env_state, reward, done, info = env.step(rng, env_state, action)

if __name__ == "__main__":
    scenario_name = "20farmers"
    seed = 0
    run(scenario_name, seed)
```