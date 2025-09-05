import tyro
from dataclasses import dataclass
import json
import os
from datetime import datetime


@dataclass
class Config:
    # Env configuration
    scenario: str = "8A_vs_1A1M1H"
    max_field_height: int = 4
    max_field_width: int = 5

    # Evaluation configuration
    n_env: int = 4096 * 4  # The number of environments to run in parallel
    rollout_step: int = (
        1024  # The number of rollouts to run in parallel for TABSUnitComb and TABSUnitDeploy
    )
    seed: int = 42

    # Heuristic policy configuration
    epsilon: float = 0.1
    aggressive_threshold: float = 0.75
    gpu_id: int = 0


if __name__ == "__main__":
    config = tyro.cli(Config)

    import os

    os.environ["CUDA_VISIBLE_DEVICES"] = str(config.gpu_id)

    import jax
    import jax.numpy as jnp
    from src.maenv.tabs.tabs_battle_simulator.tabs_battle_simulator import TABSBattleSimulator
    from src.maenv.tabs.wrappers.wrapper import (
        TABSBattleSimulatorHeuristicWrapper,
        TABSBattleSimulatorLogWrapper,
    )
    from src.maenv.tabs.scenarios import TABSConf, generate_scenario
    from functools import partial

    scenario = generate_scenario(TABSConf(scenario_name=config.scenario))
    tabs_config = TABSConf(
        scenario_name=config.scenario,
        max_n_ally=int(scenario.ally_unit_comp.sum().item()),
        max_n_enemy=int(scenario.enemy_unit_comp.sum().item()),
        max_field_height=config.max_field_height,
        max_field_width=config.max_field_width,
    )

    env = TABSBattleSimulator(tabs_config)
    env = TABSBattleSimulatorHeuristicWrapper(
        env,
        epsilon=config.epsilon,
        aggressive_threshold=config.aggressive_threshold,
        heuristic_units="all",
    )
    env = TABSBattleSimulatorLogWrapper(env, reset_when_done=False)

    repated_scenario = jax.tree.map(lambda x: jnp.repeat(x[None], config.n_env, axis=0), scenario)
    v_reset = jax.vmap(env.reset, in_axes=(0, 0))
    v_step = jax.vmap(
        partial(env.step, action={}),
        in_axes=(
            0,
            0,
        ),
    )

    def rollout_body(carry, _):
        state, key = carry
        key, next_key = jax.random.split(key)

        obs, state, reward, done, info = v_step(jax.random.split(key, config.n_env), state)
        return (state, next_key), None

    init_obs, init_state = v_reset(
        jax.random.split(jax.random.key(config.seed), config.n_env), repated_scenario
    )
    carry = (init_state, jax.random.key(config.seed))
    (last_state, _), _ = jax.lax.scan(rollout_body, carry, None, config.rollout_step)

    win_rate = last_state.returned_episode_wins.mean(axis=0)  # [n_team, 1]
    returns = last_state.episode_returns.mean(axis=0)  # [n_team, 1]
    lengths = last_state.episode_lengths.mean()  # scalar

    # Save results to JSON file
    # Create results directory if it doesn't exist
    results_dir = "heuristic_evaluation_results"
    os.makedirs(results_dir, exist_ok=True)

    # Prepare results dictionary
    results = {
        "timestamp": datetime.now().isoformat(),
        "scenario": config.scenario,
        "epsilon": config.epsilon,
        "aggressive_threshold": config.aggressive_threshold,
        "n_env": config.n_env,
        "rollout_step": config.rollout_step,
        "max_field_height": config.max_field_height,
        "max_field_width": config.max_field_width,
        "seed": config.seed,
        "win_rate": win_rate.tolist(),
        "returns": returns.tolist(),
        "lengths": float(lengths),
    }

    # Generate filename with configuration parameters
    filename = f"{config.scenario}_eps{config.epsilon}_aggr{config.aggressive_threshold}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    filepath = os.path.join(results_dir, filename)

    # Save to JSON file
    with open(filepath, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to: {filepath}")
