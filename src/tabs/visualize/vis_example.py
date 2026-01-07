import jax

from src.tabs import TABS, build_batched_env_params_and_config
from src.tabs.visualize import Visualizer
from src.tabs.wrappers import TABSEnemyHeuristicWrapper

if __name__ == "__main__":
    num_steps = 120
    seed = 0
    scenario_name = "2F1M2Avs2S1K_2L2B2S"

    env_params, tabs_config = build_batched_env_params_and_config(scenario_names=scenario_name)

    env = TABS(cfg=tabs_config)
    env = TABSEnemyHeuristicWrapper(env)

    rng = jax.random.PRNGKey(seed)
    rng, _rng = jax.random.split(rng)

    obs, env_state = env.reset(_rng, env_params)

    def rollout_body(carry, _):
        (env_state, rng) = carry
        # Random policy
        rng, step_rng, action_rng = jax.random.split(rng, 3)
        env_actions = {}
        for _, agent in enumerate(env.ally_keys):
            rng, action_rng = jax.random.split(action_rng)
            env_actions[agent] = env.action_spaces[env.unit_keys[0]].sample(rng)

        obs, next_state, reward, done, info = env.step(step_rng, env_state, env_actions)

        return (next_state, rng), next_state

    _, stacked = jax.lax.scan(rollout_body, (env_state, rng), None, num_steps)
    state_seq = [jax.tree.map(lambda x: x[i], stacked) for i in range(num_steps)]

    visualizer = Visualizer(env, state_seq)
    visualizer.animate(save_fname=f"{scenario_name}.gif", view=False)
