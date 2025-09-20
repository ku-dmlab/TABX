import jax

from src.tabs import TABSUnitComb, TABSUnitDeploy
from src.tabs.scenarios import generate_scenario
from src.tabs.config import TABSConfig
from src.tabs.visualize import UnitCombVisualizer, UnitDeployVisualizer

if __name__ == "__main__":
    num_steps = 10

    tabs_conf = TABSConfig()
    scenario = generate_scenario(tabs_conf)

    env = TABSUnitComb(tabs_conf)

    rng = jax.random.PRNGKey(0)
    rng, _rng = jax.random.split(rng)

    obs, env_state = env.reset(_rng, scenario)

    state_seq = []
    state_seq.append(env_state)
    for i in range(num_steps):
        # Random policy
        rng, step_rng, action_rng = jax.random.split(rng, 3)
        action = env.action_space.sample(action_rng)

        obs, env_state, reward, done, info = env.step(step_rng, env_state, action)

        state_seq.append(env_state)

    visualizer = UnitCombVisualizer(env, tabs_conf.scenario_name, state_seq)
    visualizer.animate(save_fname="tabsunitcomb.gif", view=False)

    env = TABSUnitDeploy(tabs_conf)

    rng, _rng = jax.random.split(rng)

    scenario = scenario.replace(ally_unit_comp=state_seq[-1].current_unit_list)
    obs, env_state = env.reset(_rng, scenario)

    state_seq = []
    state_seq.append(env_state)
    for i in range(num_steps):
        # Random policy
        rng, step_rng, action_rng = jax.random.split(rng, 3)
        action = env.action_space.sample(action_rng)

        obs, env_state, reward, done, info = env.step(step_rng, env_state, action)

        state_seq.append(env_state)

    visualizer = UnitDeployVisualizer(env, tabs_conf.scenario_name, state_seq)
    visualizer.animate(save_fname="tabsunitdeploy.gif", view=False)
