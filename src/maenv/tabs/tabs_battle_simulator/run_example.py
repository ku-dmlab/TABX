import jax

from src.maenv.tabs.tabs_battle_simulator.tabs_battle_simulator import TABSBattleSimulator
from src.maenv.tabs.scenarios import TABSConf, generate_scenario
from src.maenv.utils import Transition
from src.maenv.tabs.wrappers.wrapper import (
    TABSBattleSimulatorAutoResetWrapper,
    TABSBattleSimulatorHeuristicWrapper,
)

if __name__ == "__main__":
    n_envs = 5
    num_steps = 10

    tabs_conf = TABSConf()
    env = TABSBattleSimulator(tabs_conf)
    scenario = generate_scenario(tabs_conf)
    env = TABSBattleSimulatorHeuristicWrapper(env, "enemy")
    env = TABSBattleSimulatorAutoResetWrapper(env, scenario)

    v_reset = jax.vmap(env.reset, in_axes=(0, None))
    v_step = jax.vmap(env.step, in_axes=(0, 0, 0))

    rng = jax.random.PRNGKey(0)
    rng, _rng = jax.random.split(rng)

    init_obs, init_state = v_reset(jax.random.split(_rng, n_envs), scenario)

    def _run(carry, _):
        obs, env_state, rng = carry

        # Random policy
        rng, action_rng = jax.random.split(rng)
        env_actions = {}
        for i, agent in enumerate(env.ally_keys):
            action_rngs = jax.random.split(action_rng, n_envs + 1)
            action_rng = action_rngs[0]
            env_actions[agent] = jax.vmap(env.action_space.sample, in_axes=(0))(action_rngs[1:])

        rng, _rng = jax.random.split(rng)
        next_obs, next_state, rewards, dones, infos = v_step(
            jax.random.split(_rng, n_envs), env_state, env_actions
        )

        transition = Transition(
            done=dones,
            action=env_actions,
            reward=rewards,
            obs=obs,
            info=infos,
            unavail_action=None,
        )

        return (next_obs, next_state, rng), transition

    rng, _rng = jax.random.split(rng)
    carry, trajs = jax.lax.scan(_run, (init_obs, init_state, _rng), None, num_steps)
