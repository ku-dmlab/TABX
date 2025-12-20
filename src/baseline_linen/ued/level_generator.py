from typing import Callable
from typing import Dict as Level

import jax
import jax.numpy as jnp
import chex

from src.tabs.scenarios import Scenario
from src.tabs.config import TABSHeuristicConfig

FREE_PARAM_TYPES = {"unit_spec": 0, "heuristic_config": 1}


# NOTE: For each spec, minval and maxval represent the minimum and maximum values of the predefined units, respectively.
#       You need to change this for the evaluation phase with unseen unit specs.
unit_spec_ranges = {"health": [25, 685], "speed": [0.5, 1.4], "attack_damage": [-7, 80]}


def randomize_unit_specs(env_params: Level, rng: chex.PRNGKey) -> Level:
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


# NOTE: Each element represents the probability of taking a random action and aggressive behavior, respectively.
#       You need to change this for the evaluation phase with unseen configuration.
heuristic_config_ranges = {"epsilon": [0.0, 1.0], "aggressive_threshold": [0.0, 1.0]}


def randomize_heuristic_config(env_params: Level, rng: chex.PRNGKey) -> Level:
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

    env_params = {
        "scenario": env_params["scenario"],
        "physics_params": env_params["physics_params"],
        "heuristic_params": _randomize_heuristic_config(env_params["heuristic_params"], rng),
    }

    return env_params


def level_generator(free_param_type: int) -> Callable:
    def generate_level(env_params: Level, rng: chex.PRNGKey) -> Level:
        return jax.lax.switch(
            free_param_type, [randomize_unit_specs, randomize_heuristic_config], env_params, rng
        )

    return generate_level


def mutate_unit_spec(env_params: Level, rng: chex.PRNGKey, _) -> Level:
    def _mutate_unit_spec(scenario: Scenario, rng: chex.PRNGKey) -> Scenario:
        # Add noise
        rngs = jax.random.split(rng, 3)
        scenario = scenario.replace(
            health=jnp.clip(
                scenario.health + jax.random.uniform(rngs[0], minval=-0.1, maxval=0.1),
                min=unit_spec_ranges["health"][0],
                max=unit_spec_ranges["health"][1],
            ),
            speed=jnp.clip(
                scenario.speed + jax.random.uniform(rngs[1], minval=-0.1, maxval=0.1),
                min=unit_spec_ranges["speed"][0],
                max=unit_spec_ranges["speed"][1],
            ),
            attack_damage=jnp.clip(
                scenario.attack_damage + jax.random.uniform(rngs[2], minval=-0.1, maxval=0.1),
                min=unit_spec_ranges["attack_damage"][0],
                max=unit_spec_ranges["attack_damage"][1],
            ),
        )
        return scenario

    env_params = {
        "scenario": _mutate_unit_spec(env_params["scenario"], rng),
        "physics_params": env_params["physics_params"],
        "heuristic_params": env_params["heuristic_params"],
    }

    return env_params


def mutate_heuristic_config(env_params: Level, rng: chex.PRNGKey, _) -> Level:
    def _mutate_heuristic_config(
        config: TABSHeuristicConfig, rng: chex.PRNGKey
    ) -> TABSHeuristicConfig:
        # Add noise
        rngs = jax.random.split(rng)
        config = config.replace(
            epsilon=jnp.clip(
                config.epsilon + jax.random.uniform(rngs[0], minval=-0.1, maxval=0.1),
                min=heuristic_config_ranges["epsilon"][0],
                max=heuristic_config_ranges["epsilon"][1],
            ),
            aggressive_threshold=jnp.clip(
                config.aggressive_threshold + jax.random.uniform(rngs[1], minval=-0.1, maxval=0.1),
                min=heuristic_config_ranges["aggressive_threshold"][0],
                max=heuristic_config_ranges["aggressive_threshold"][1],
            ),
        )

        return config

    env_params = {
        "scenario": env_params["scenario"],
        "physics_params": env_params["physics_params"],
        "heuristic_params": _mutate_heuristic_config(env_params["heuristic_params"], rng),
    }

    return env_params


def mutate_level_generator(free_param_type: int) -> Callable:
    def mutate_level(env_params: Level, rng: chex.PRNGKey, num_edits: int) -> Level:
        return jax.lax.switch(
            free_param_type, [mutate_unit_spec, mutate_heuristic_config], env_params, rng, num_edits
        )

    return mutate_level
