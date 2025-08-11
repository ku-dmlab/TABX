import jax
from src.maenv.tabs.tabs_unit_comb.tabs_unit_comb import TABSUnitComb
from src.maenv.tabs.scenarios import MAP_NAME_TO_SCENARIO
from src.maenv.utils import Transition

if __name__ == "__main__":
    n_envs = 5
    num_steps = 10

    env = TABSUnitComb()
    scenario = MAP_NAME_TO_SCENARIO["4archer_1mammoth"]

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
            avail_action=env_state.action_mask,
        )

        return (next_obs, next_state, rng), transition

    rng, _rng = jax.random.split(rng)
    _run((init_obs, init_state, _rng), None)
    carry, trajs = jax.lax.scan(_run, (init_obs, init_state, _rng), None, num_steps)
