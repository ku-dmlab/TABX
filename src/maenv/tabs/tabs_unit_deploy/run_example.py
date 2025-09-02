import jax
from src.maenv.tabs.tabs_unit_deploy.tabs_unit_deploy import TABSUnitDeploy
from src.maenv.tabs.scenarios import default_tabs_conf, generate_scenario
from src.maenv.utils import Transition

if __name__ == "__main__":
    n_envs = 5
    num_steps = 10

    env = TABSUnitDeploy(default_tabs_conf)
    scenario = generate_scenario(default_tabs_conf)

    v_reset = jax.vmap(env.reset, in_axes=(0, None))
    v_step = jax.vmap(env.step, in_axes=(0, 0, 0))

    rng = jax.random.PRNGKey(0)
    rng, _rng = jax.random.split(rng)

    init_obs, init_state = v_reset(jax.random.split(_rng, n_envs), scenario)

    def _run(carry, _):
        obs, env_state, rng = carry

        # Random policy
        rng, action_rng = jax.random.split(rng)
        action_rngs = jax.random.split(action_rng, n_envs)
        actions = jax.vmap(env.action_space.sample, in_axes=(0))(action_rngs)

        rng, _rng = jax.random.split(rng)
        next_obs, next_state, rewards, dones, infos = v_step(
            jax.random.split(_rng, n_envs), env_state, actions
        )

        transition = Transition(
            done=dones,
            action=actions,
            reward=rewards,
            obs=obs,
            info=infos,
            unavail_action=env_state.unavail_action,
        )

        return (next_obs, next_state, rng), transition

    rng, _rng = jax.random.split(rng)
    carry, trajs = jax.lax.scan(_run, (init_obs, init_state, _rng), None, num_steps)
