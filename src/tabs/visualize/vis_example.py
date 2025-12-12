import jax

from src.tabs import TABS
from src.tabs.visualize import Visualizer
from src.tabs.scenarios import generate_scenario
from src.tabs.wrappers import TABSEnemyHeuristicWrapper
from src.tabs.config import TABSConfig, PhysicsParams, TABSHeuristicConfig

if __name__ == "__main__":
    num_steps = 120
    scenario_name = "2F1K2A1H_tight"

    tabs_conf = TABSConfig(scenario_name=scenario_name)
    scenario = generate_scenario(tabs_conf)
    tabs_config = TABSConfig(
        scenario_name=scenario_name,
        max_n_ally=int(scenario.ally_unit_comp.sum().item()),
        max_n_enemy=int(scenario.enemy_unit_comp.sum().item()),
    )

    env = TABS(cfg=tabs_conf)
    env = TABSEnemyHeuristicWrapper(env)

    rng = jax.random.PRNGKey(0)
    rng, _rng = jax.random.split(rng)

    env_params = {
        "scenario": scenario,
        "physics_params": PhysicsParams(),
        "heuristic_params": TABSHeuristicConfig(),
    }
    obs, env_state = env.reset(_rng, env_params)

    def rollout_body(carry, _):
        (env_state, rng) = carry
        # Random policy
        rng, step_rng, action_rng = jax.random.split(rng, 3)
        env_actions = {}
        for _, agent in enumerate(env.ally_keys):
            rng, action_rng = jax.random.split(action_rng)
            env_actions[agent] = env.action_spaces[env.unit_keys[0]].sample(rng)

        obs, next_state, reward, done, info = jax.jit(env.step)(step_rng, env_state, env_actions)

        return (next_state, rng), next_state

    carry, stacked = jax.lax.scan(rollout_body, (env_state, rng), None, num_steps)
    state_seq = [jax.tree.map(lambda x: x[i], stacked) for i in range(num_steps)]

    visualizer = Visualizer(env, tabs_conf.scenario_name, state_seq)
    visualizer.animate(save_fname="tabs.gif", view=False)
