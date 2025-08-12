import jax

from src.maenv.tabs.tabs_unit_comb.tabs_unit_comb import TABSUnitComb
from src.maenv.tabs.tabs_unit_deploy.tabs_unit_deploy import TABSUnitDeploy
from src.maenv.tabs.scenarios import default_tabs_conf, generate_scenario
from src.maenv.utils import Transition


def make_run(n_envs, num_steps):
    # Get scenario
    scenario = generate_scenario(default_tabs_conf)

    # Instantiate envs
    comb_env = TABSUnitComb(default_tabs_conf)
    deploy_env = TABSUnitDeploy(default_tabs_conf)

    comb_reset = jax.vmap(comb_env.reset, in_axes=(0, None))
    comb_step = jax.vmap(comb_env.step, in_axes=(0, 0, 0))
    deploy_reset = jax.vmap(deploy_env.reset, in_axes=(0, 0))
    deploy_step = jax.vmap(deploy_env.step, in_axes=(0, 0, 0))

    def run(rng):
        def _run_comb(carry, _):
            obs, env_state, rng = carry

            # Random policy
            rng, action_rng = jax.random.split(rng)
            action_rngs = jax.random.split(action_rng, n_envs)
            actions = jax.vmap(comb_env.action_space.sample, in_axes=(0))(action_rngs)

            rng, _rng = jax.random.split(rng)
            next_obs, next_state, rewards, dones, infos = comb_step(
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

        rng, init_rng, _rng = jax.random.split(rng, 3)
        init_obs, init_state = comb_reset(jax.random.split(init_rng, n_envs), scenario)
        carry, traj_comb = jax.lax.scan(_run_comb, (init_obs, init_state, _rng), None, num_steps)
        _, next_state, rng = carry

        def _run_deploy(carry, _):
            obs, env_state, rng = carry

            # Random policy
            rng, action_rng = jax.random.split(rng)
            action_rngs = jax.random.split(action_rng, n_envs)
            actions = jax.vmap(deploy_env.action_space.sample, in_axes=(0))(action_rngs)

            rng, _rng = jax.random.split(rng)
            next_obs, next_state, rewards, dones, infos = deploy_step(
                jax.random.split(_rng, n_envs), env_state, actions
            )

            transition = Transition(
                done=dones,
                action=actions,
                reward=rewards,
                obs=obs,
                info=infos,
                avail_action=env_state.battle_field_mask.flatten(),
            )

            return (next_obs, next_state, rng), transition

        rng, init_rng, _rng = jax.random.split(rng, 3)
        scenarios = jax.vmap(lambda s, x: s.replace(ally_unit_comp=x), in_axes=(None, 0))(
            scenario, next_state.current_unit_list
        )
        init_obs, init_state = deploy_reset(jax.random.split(init_rng, n_envs), scenarios)
        carry, traj_deploy = jax.lax.scan(
            _run_deploy, (init_obs, init_state, _rng), None, num_steps
        )
        _, next_state, _ = carry

        scenarios = jax.vmap(
            lambda s, x, m: s.replace(battle_field=x, battle_field_mask=m), in_axes=(None, 0, 0)
        )(scenario, next_state.battle_field, next_state.battle_field_mask)

        return {"traj_comb": traj_comb, "traj_deploy": traj_deploy}

    return run


def main():
    n_envs = 5
    num_steps = 10
    seed = 0

    rng = jax.random.PRNGKey(seed)
    with jax.disable_jit(False):
        run_jit = jax.jit(make_run(n_envs, num_steps))
        out = run_jit(rng)

    # print(out)


if __name__ == "__main__":
    main()
