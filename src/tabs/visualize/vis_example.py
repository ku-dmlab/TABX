import jax

from src.tabs import TABSUnitComb, TABSUnitDeploy, TABSBattleSimulator
from src.tabs.wrappers import TABSBattleSimulatorHeuristicWrapper
from src.tabs.scenarios import generate_scenario
from src.tabs.config import TABSConfig, TABSHeuristicConfig
from src.tabs.visualize import UnitCombVisualizer, UnitDeployVisualizer, BattleSimulatorVisualizer

if __name__ == "__main__":
    num_steps = 10
    bs_num_steps = 120

    tabs_conf = TABSConfig()
    scenario = generate_scenario(tabs_conf)

    rng = jax.random.PRNGKey(0)
    rng, _rng = jax.random.split(rng)

    # TABSUnitComb
    env = TABSUnitComb(tabs_conf)
    obs, env_state = env.reset(_rng, scenario)

    def rollout_body(carry, _):
        (env_state, rng) = carry
        # Random policy
        rng, step_rng, action_rng = jax.random.split(rng, 3)
        action = env.action_space.sample(action_rng)

        obs, next_state, reward, done, info = jax.jit(env.step)(step_rng, env_state, action)

        return (next_state, rng), next_state

    carry, stacked = jax.lax.scan(rollout_body, (env_state, rng), None, bs_num_steps)

    state_seq = [jax.tree.map(lambda x: x[i], stacked) for i in range(bs_num_steps)]

    visualizer = UnitCombVisualizer(env, tabs_conf.scenario_name, state_seq)
    visualizer.animate(save_fname="tabsunitcomb.gif", view=False)

    # TABSUnitDeploy
    env = TABSUnitDeploy(tabs_conf)
    rng, _rng = jax.random.split(rng)

    scenario = scenario.replace(ally_unit_comp=state_seq[-1].current_unit_list)
    obs, env_state = env.reset(_rng, scenario)

    def rollout_body(carry, _):
        (env_state, rng) = carry
        # Random policy
        rng, step_rng, action_rng = jax.random.split(rng, 3)
        action = env.action_space.sample(action_rng)

        obs, next_state, reward, done, info = jax.jit(env.step)(step_rng, env_state, action)

        return (next_state, rng), next_state

    carry, stacked = jax.lax.scan(rollout_body, (env_state, rng), None, bs_num_steps)

    state_seq = [jax.tree.map(lambda x: x[i], stacked) for i in range(bs_num_steps)]

    visualizer = UnitDeployVisualizer(env, tabs_conf.scenario_name, state_seq)
    visualizer.animate(save_fname="tabsunitdeploy.gif", view=False)

    # TABSBattleSimulator
    env = TABSBattleSimulator(tabs_conf)
    env = TABSBattleSimulatorHeuristicWrapper(env, "enemy", TABSHeuristicConfig())
    rng, _rng = jax.random.split(rng)

    scenario = scenario.replace(battle_field=state_seq[-1].battle_field)
    obs, env_state = env.reset(_rng, scenario)

    def rollout_body(carry, _):
        (env_state, rng) = carry
        # Random policy
        rng, step_rng, action_rng = jax.random.split(rng, 3)
        env_actions = {}
        for _, agent in enumerate(env.ally_keys):
            rng, action_rng = jax.random.split(action_rng)
            env_actions[agent] = env.action_space.sample(rng)

        obs, next_state, reward, done, info = jax.jit(env.step)(step_rng, env_state, env_actions)

        return (next_state, rng), next_state

    carry, stacked = jax.lax.scan(rollout_body, (env_state, rng), None, bs_num_steps)

    state_seq = [jax.tree.map(lambda x: x[i], stacked) for i in range(bs_num_steps)]

    visualizer = BattleSimulatorVisualizer(env, tabs_conf.scenario_name, state_seq)
    visualizer.animate(save_fname="tabsbattlesimulator.gif", view=False)
