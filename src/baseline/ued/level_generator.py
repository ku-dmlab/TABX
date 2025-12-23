from enum import IntEnum
from functools import partial
from typing import Callable
from typing import Dict as Level

import chex
import jax
import jax.numpy as jnp

from src.tabs.config import TABSHeuristicConfig
from src.tabs.scenarios import Scenario, ZoneScenario

FREE_PARAM_TYPES = {"zone": 0, "unit_spec": 1, "heuristic_config": 2}

zone_ranges = {
    "zone_type": [0, 2],
    "position_x": [15, 45],  # battle field width: 60.5
    "position_y": [10, 30],  # battle field height: 39
    "axes_x": [2, 10],
    "axes_y": [2, 5],
    "damage": [2, 20],
}


def randomize_zone(env_params: Level, rng: chex.PRNGKey) -> Level:
    def _randomize_zone(zone_scenario: ZoneScenario, rng: chex.PRNGKey) -> ZoneScenario:
        n_zones = zone_scenario.zone_type.shape[0]
        # Randomly set zone specifications (zone_type, ellipse, damage)
        rngs = jax.random.split(rng, 6)
        zone_scenario = zone_scenario.replace(
            zone_type=jax.random.randint(
                rngs[0],
                shape=(n_zones, 1),
                minval=zone_ranges["zone_type"][0],
                maxval=zone_ranges["zone_type"][1],
            ),
            position=jnp.array(
                [
                    jax.random.uniform(
                        rngs[1],
                        shape=(n_zones),
                        minval=zone_ranges["position_x"][0],
                        maxval=zone_ranges["position_x"][1],
                    ),
                    jax.random.uniform(
                        rngs[2],
                        shape=(n_zones),
                        minval=zone_ranges["position_y"][0],
                        maxval=zone_ranges["position_y"][1],
                    ),
                ]
            ).T,
            axes=jnp.array(
                [
                    jax.random.uniform(
                        rngs[3],
                        shape=(n_zones),
                        minval=zone_ranges["axes_x"][0],
                        maxval=zone_ranges["axes_x"][1],
                    ),
                    jax.random.uniform(
                        rngs[4],
                        shape=(n_zones),
                        minval=zone_ranges["axes_y"][0],
                        maxval=zone_ranges["axes_y"][1],
                    ),
                ]
            ).T,
            damage=jax.random.uniform(
                rngs[5],
                shape=(n_zones, 1),
                minval=zone_ranges["damage"][0],
                maxval=zone_ranges["damage"][1],
            ),
        )
        return zone_scenario

    env_params = {
        "scenario": env_params["scenario"],
        "zone_scenario": _randomize_zone(env_params["zone_scenario"], rng),
        "physics_params": env_params["physics_params"],
        "heuristic_params": env_params["heuristic_params"],
    }
    return env_params


# NOTE: For each spec, minval and maxval represent the minimum and maximum values of the predefined units, respectively.
#       You need to change this for the evaluation phase with unseen unit specs.
unit_spec_ranges = {"healths": [25, 685], "speeds": [0.5, 1.4], "attack_damages": [-7, 80]}


def randomize_unit_specs(env_params: Level, rng: chex.PRNGKey) -> Level:
    def _randomize_unit_specs(scenario: Scenario, rng: chex.PRNGKey) -> Scenario:
        n_units = scenario.healths.shape[0]
        # Randomly set unit specifications (health, speed, attack_damage)
        rngs = jax.random.split(rng, 3)
        scenario = scenario.replace(
            healths=jax.random.uniform(
                rngs[0],
                shape=(n_units, 1),
                minval=unit_spec_ranges["healths"][0],
                maxval=unit_spec_ranges["healths"][1],
            ).astype(jnp.float32),
            speeds=jax.random.uniform(
                rngs[1],
                shape=(n_units, 1),
                minval=unit_spec_ranges["speeds"][0],
                maxval=unit_spec_ranges["speeds"][1],
            ).astype(jnp.float32),
            attack_damages=jax.random.uniform(
                rngs[2],
                shape=(n_units, 1),
                minval=unit_spec_ranges["attack_damages"][0],
                maxval=unit_spec_ranges["attack_damages"][1],
            ).astype(jnp.float32),
        )

        return scenario

    env_params = {
        "scenario": _randomize_unit_specs(env_params["scenario"], rng),
        "zone_scenario": env_params["zone_scenario"],
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
        "zone_scenario": env_params["zone_scenario"],
        "physics_params": env_params["physics_params"],
        "heuristic_params": _randomize_heuristic_config(env_params["heuristic_params"], rng),
    }

    return env_params


def level_generator(free_param_type: int) -> Callable:
    def generate_level(env_params: Level, rng: chex.PRNGKey) -> Level:
        return jax.lax.switch(
            free_param_type,
            [randomize_zone, randomize_unit_specs, randomize_heuristic_config],
            env_params,
            rng,
        )

    return generate_level


def mutate_zone(env_params: Level, rng: chex.PRNGKey, num_edits=3) -> Level:
    class Mutations(IntEnum):
        NO_OP = 0
        ADD_NOISE = 1
        ROTATE = 2
        CHANGE_TYPE = 3

    def _no_op(zone_scenario: ZoneScenario, rng: chex.PRNGKey) -> ZoneScenario:
        return zone_scenario

    def _add_noise(zone_scenario: ZoneScenario, rng: chex.PRNGKey) -> ZoneScenario:
        # Add noise
        rngs = jax.random.split(rng, 5)
        zone_scenario = zone_scenario.replace(
            position=jnp.array(
                [
                    jnp.clip(
                        zone_scenario.position[:, 0]
                        + jax.random.uniform(rngs[0], minval=-4, maxval=4),
                        min=zone_ranges["position_x"][0],
                        max=zone_ranges["position_x"][1],
                    ),
                    jnp.clip(
                        zone_scenario.position[:, 1]
                        + jax.random.uniform(rngs[1], minval=-4, maxval=4),
                        min=zone_ranges["position_y"][0],
                        max=zone_ranges["position_y"][1],
                    ),
                ]
            ).T,
            axes=jnp.array(
                [
                    jnp.clip(
                        zone_scenario.axes[:, 0] + jax.random.uniform(rngs[2], minval=-2, maxval=2),
                        min=zone_ranges["axes_x"][0],
                        max=zone_ranges["axes_x"][1],
                    ),
                    jnp.clip(
                        zone_scenario.axes[:, 1] + jax.random.uniform(rngs[3], minval=-2, maxval=2),
                        min=zone_ranges["axes_y"][0],
                        max=zone_ranges["axes_y"][1],
                    ),
                ]
            ).T,
            damage=jnp.clip(
                zone_scenario.damage + jax.random.uniform(rngs[4], minval=-4, maxval=4),
                min=zone_ranges["damage"][0],
                max=zone_ranges["damage"][1],
            ),
        )
        return zone_scenario

    def _rotate(zone_scenario: ZoneScenario, rng: chex.PRNGKey) -> ZoneScenario:
        # Swap axes size
        swapped = jnp.roll(zone_scenario.axes, shift=1, axis=1)
        zone_scenario = zone_scenario.replace(
            axes=jnp.array(
                [
                    jnp.clip(
                        swapped[:, 0],
                        min=zone_ranges["axes_x"][0],
                        max=zone_ranges["axes_x"][1],
                    ),
                    jnp.clip(
                        swapped[:, 1],
                        min=zone_ranges["axes_y"][0],
                        max=zone_ranges["axes_y"][1],
                    ),
                ]
            ).T
        )
        return zone_scenario

    def _change_type(zone_scenario: ZoneScenario, rng: chex.PRNGKey) -> ZoneScenario:
        # Randomly reassign zone type
        n_zones = zone_scenario.zone_type.shape[0]
        zone_scenario = zone_scenario.replace(
            zone_type=jax.random.randint(
                rng,
                shape=(n_zones, 1),
                minval=zone_ranges["zone_type"][0],
                maxval=zone_ranges["zone_type"][1],
            )
        )
        return zone_scenario

    def _mutate_zone(zone_scenario: ZoneScenario, rng: chex.PRNGKey) -> ZoneScenario:
        def _mutate(zone_scenario, step):
            rng, mutation = step

            def _apply(zone_scenario, rng):
                zone_scenario = jax.lax.switch(
                    mutation, [_no_op, _add_noise, _rotate, _change_type], zone_scenario, rng
                )

                return zone_scenario

            return jax.lax.cond(
                mutation != -1, _apply, lambda *_: zone_scenario, zone_scenario, rng
            ), None

        rng, nrng, *mrngs = jax.random.split(rng, num_edits + 2)
        mutations = jax.random.choice(
            nrng, jnp.arange(len(Mutations)), shape=(num_edits,), p=jnp.array([0.1, 0.3, 0.3, 0.3])
        )

        zone_scenario, _ = jax.lax.scan(_mutate, zone_scenario, (jnp.array(mrngs), mutations))

        return zone_scenario

    env_params = {
        "scenario": env_params["scenario"],
        "zone_scenario": _mutate_zone(env_params["zone_scenario"], rng),
        "physics_params": env_params["physics_params"],
        "heuristic_params": env_params["heuristic_params"],
    }

    return env_params


def mutate_unit_spec(env_params: Level, rng: chex.PRNGKey) -> Level:
    def _mutate_unit_spec(scenario: Scenario, rng: chex.PRNGKey) -> Scenario:
        # Add noise
        rngs = jax.random.split(rng, 3)
        scenario = scenario.replace(
            healths=jnp.clip(
                scenario.healths + jax.random.uniform(rngs[0], minval=-0.1, maxval=0.1),
                min=unit_spec_ranges["healths"][0],
                max=unit_spec_ranges["healths"][1],
            ),
            speeds=jnp.clip(
                scenario.speeds + jax.random.uniform(rngs[1], minval=-0.1, maxval=0.1),
                min=unit_spec_ranges["speeds"][0],
                max=unit_spec_ranges["speeds"][1],
            ),
            attack_damages=jnp.clip(
                scenario.attack_damages + jax.random.uniform(rngs[2], minval=-0.1, maxval=0.1),
                min=unit_spec_ranges["attack_damages"][0],
                max=unit_spec_ranges["attack_damages"][1],
            ),
        )
        return scenario

    env_params = {
        "scenario": _mutate_unit_spec(env_params["scenario"], rng),
        "zone_scenario": env_params["zone_scenario"],
        "physics_params": env_params["physics_params"],
        "heuristic_params": env_params["heuristic_params"],
    }

    return env_params


def mutate_heuristic_config(env_params: Level, rng: chex.PRNGKey) -> Level:
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
        "zone_scenario": env_params["zone_scenario"],
        "physics_params": env_params["physics_params"],
        "heuristic_params": _mutate_heuristic_config(env_params["heuristic_params"], rng),
    }

    return env_params


def mutate_level_generator(free_param_type: int) -> Callable:
    def mutate_level(env_params: Level, rng: chex.PRNGKey) -> Level:
        return jax.lax.switch(
            free_param_type,
            [mutate_zone, mutate_unit_spec, mutate_heuristic_config],
            env_params,
            rng,
        )

    return mutate_level
