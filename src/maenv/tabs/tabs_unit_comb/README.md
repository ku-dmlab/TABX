# TABSUnitComb Envrionment

This environment, TABSUnitComb, implemented in JAX, evaluates an agent to determine the optimal unit combination for a specific scenario. The agent should understand unit specifications and the features of predefined (or custom) enemy compositions, and purchase appropriate units within the given budget.


The environment is instantiated with a scenario that specifies a budget and an enemy unit composition. If no scenario is provided, a default configuration (an 800 budget and 10 farmers) is applied. Upon initialization, the environment returns an observation that includes the current budget, the list of purchased units, the prices of all available units, and the enemy unit composition. The agent is expected to output a unit ID to purchase. The episode continues until the agent can no longer purchase any units due to an exhausted budget.

## Getting Started
```python
import jax
from src.maenv.tabs.tabs_unit_comb.tabs_unit_comb import TASBUnitComb
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