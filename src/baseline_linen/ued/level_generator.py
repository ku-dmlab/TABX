from typing import Dict

import jax
import jax.numpy as jnp
import chex

from src.tabs.scenarios import Scenario
from src.tabs.config import TABSHeuristicConfig

FREE_PARAM_TYPES = {"unit_spec": 0, "heuristic_config": 1}


def randomize_unit_specs(env_params: Dict, rng: chex.PRNGKey) -> Dict:
    # NOTE: For each spec, minval and maxval represent the minimum and maximum values of the predefined units, respectively.
    #       You need to change this for the evaluation phase with unseen unit specs.
    unit_spec_ranges = {"health": [25, 685], "speed": [0.5, 1.4], "attack_damage": [-7, 80]}

    def _randomize_unit_specs(scenario: Scenario, rng: chex.PRNGKey) -> Scenario:
        n_units = scenario.health.shape[0]
        # Randomly set unit specifications (health, speed, attack_damage)
        rngs = jax.random.split(rng, 3)
        scenario = scenario.replace(
            health=jax.random.uniform(
                rngs[0],
                shape=(n_units,),
                minval=unit_spec_ranges["health"][0],
                maxval=unit_spec_ranges["health"][1],
            ).astype(jnp.float32),
            speed=jax.random.uniform(
                rngs[1],
                shape=(n_units,),
                minval=unit_spec_ranges["speed"][0],
                maxval=unit_spec_ranges["speed"][1],
            ).astype(jnp.float32),
            attack_damage=jax.random.uniform(
                rngs[2],
                shape=(n_units,),
                minval=unit_spec_ranges["attack_damage"][0],
                maxval=unit_spec_ranges["attack_damage"][1],
            ).astype(jnp.float32),
        )

        return scenario

    env_params = {
        "scenario": _randomize_unit_specs(env_params["scenario"], rng),
        "physics_params": env_params["physics_params"],
        "heuristic_params": env_params["heuristic_params"],
    }

    return env_params


def randomize_heuristic_config(env_params: Dict, rng: chex.PRNGKey) -> Dict:
    # NOTE: Each element represents the probability of taking a random action and aggressive behavior, respectively.
    #       You need to change this for the evaluation phase with unseen configuration.
    heuristic_config_ranges = {"epsilon": [0.0, 1.0], "aggressive_threshold": [0.0, 1.0]}

    def _randomize_heuristic_config(
        config: TABSHeuristicConfig, rng: chex.PRNGKey
    ) -> TABSHeuristicConfig:
        # Randomly set heuristic policy configuration (epsilon, aggressive_threshold)
        rngs = jax.random.split(rng)
        config = config.replace(
            epsilon=jax.random.uniform(
                rngs[0],
                shape=(1,),
                minval=heuristic_config_ranges["epsilon"][0],
                maxval=heuristic_config_ranges["epsilon"][1],
            ).astype(jnp.float32),
            aggressive_threshold=jax.random.uniform(
                rngs[1],
                shape=(1,),
                minval=heuristic_config_ranges["aggressive_threshold"][0],
                maxval=heuristic_config_ranges["aggressive_threshold"][1],
            ).astype(jnp.float32),
        )

        return config

    env_params["heuristic_params"] = _randomize_heuristic_config(
        env_params["heuristic_params"], rng
    )

    return env_params


def generate_level(env_params: Dict, free_param_type: int, rng: chex.PRNGKey) -> Dict:
    return jax.lax.switch(
        free_param_type, [randomize_unit_specs, randomize_heuristic_config], env_params, rng
    )
