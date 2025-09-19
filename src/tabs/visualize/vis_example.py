import jax

from src.tabs import TABSUnitComb
from src.tabs.scenarios import generate_scenario
from src.tabs.config import TABSConfig
from src.tabs.visualize import UnitCombVisualizer

if __name__ == "__main__":
    num_steps = 10

    tabs_conf = TABSConfig()
    env = TABSUnitComb(tabs_conf)
    scenario = generate_scenario(tabs_conf)

    rng = jax.random.PRNGKey(0)
    rng, _rng = jax.random.split(rng)

    state_seq = []

    obs, env_state = env.reset(_rng, scenario)
    state_seq.append(env_state)

    for i in range(num_steps):
        # Random policy
        rng, step_rng, action_rng = jax.random.split(rng, 3)
        action = env.action_space.sample(action_rng)

        obs, env_state, reward, done, info = env.step(step_rng, env_state, action)

        state_seq.append(env_state)

    print(len(state_seq))

    visualizer = UnitCombVisualizer(env, tabs_conf.scenario_name, state_seq)
    visualizer.animate(save_fname="output.gif", view=False)
