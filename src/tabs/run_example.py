import jax
import jax.numpy as jnp

from src.tabs import TABS
from src.tabs.config import TABSHeuristicConfig
from src.tabs.physics import build_batched_physics_params
from src.tabs.scenarios import build_batched_scenarios
from src.tabs.wrappers.wrappers import (
    TABSAutoResetWrapper,
    TABSEnemyHeuristicWrapper,
    TABSLogWrapper,
)

if __name__ == "__main__":
    n_envs = 5
    num_steps = 10
    scenario_name = "elbow"

    vscenario, zone_scenario, tabs_config = build_batched_scenarios(
        scenario_names=scenario_name, n_repeat=n_envs
    )
    physics_params = build_batched_physics_params(physics_param_names="default", n_repeat=n_envs)
    env = TABS(cfg=tabs_config)
    env = TABSLogWrapper(env)
    env = TABSEnemyHeuristicWrapper(env)
    env = TABSAutoResetWrapper(env)

    v_reset = jax.vmap(env.reset, in_axes=(0, 0))
    v_step = jax.vmap(env.step, in_axes=(0, 0, 0, 0))

    rng = jax.random.PRNGKey(0)
    rng, _rng = jax.random.split(rng)

    env_params = jax.tree.map(
        lambda x: jnp.repeat(x[None], n_envs, axis=0),
        {
            "heuristic_params": TABSHeuristicConfig(),
        },
    ) | {
        "scenario": vscenario,
        "zone_scenario": zone_scenario,
        "physics_params": physics_params,
    }

    init_obs, init_state = v_reset(jax.random.split(_rng, n_envs), env_params)

    def _run(carry, _):
        obs, env_state, rng = carry

        # Random policy
        rng, action_rng = jax.random.split(rng)
        env_actions = {}
        for i, agent in enumerate(env.ally_keys):
            action_rngs = jax.random.split(action_rng, n_envs + 1)
            action_rng = action_rngs[0]
            env_actions[agent] = jax.vmap(env.action_spaces[env.unit_keys[0]].sample, in_axes=(0))(
                action_rngs[1:]
            )

        rng, _rng = jax.random.split(rng)
        next_obs, next_state, rewards, dones, infos = v_step(
            jax.random.split(_rng, n_envs), env_state, env_actions, env_params
        )

        transition = {
            "done": dones,
            "action": env_actions,
            "reward": rewards,
            "obs": obs,
            "info": infos,
        }

        return (next_obs, next_state, rng), transition

    rng, _rng = jax.random.split(rng)
    carry, trajs = jax.lax.scan(_run, (init_obs, init_state, _rng), None, num_steps)
    print(trajs)
