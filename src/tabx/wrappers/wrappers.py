from typing import Any, Dict, List

import chex
import jax
import jax.numpy as jnp
from flax import struct

from src.tabx import TABX
from src.tabx.heuristic_policy import LastVisibleTarget, heuristic_policy
from src.tabx.heuristic_policy.params import TABXHeuristicParam
from src.tabx.scenarios import VectorizedScenario


class BaseWrapper:
    def __getattr__(self, name: str):
        return getattr(self.env, name)

    def __init__(self, env: TABX):
        self.env = env


class TABXEnemyHeuristicWrapper(BaseWrapper):
    def __init__(
        self,
        env: TABX,
    ):
        super().__init__(env)

        self.action_spaces = {
            agent: action_space_
            for (agent, action_space_) in self.env.action_spaces.items()
            if agent in self.env.ally_keys
        }

        self.observation_spaces = {
            agent: observation_space
            for (agent, observation_space) in self.env.observation_spaces.items()
            if agent in self.env.ally_keys
        }

        self.agents = self.env.ally_keys
        self.num_agents = len(self.agents)

    def filter_obs(self, obs):
        target_obs = {}
        for obs_key in obs.keys():
            if obs_key in self.env.enemy_keys:
                continue
            target_obs[obs_key] = obs[obs_key]
        return target_obs

    def reset(self, key, env_params: Dict[str, Any]):
        if "heuristic_params" not in env_params:
            raise ValueError("The heuristic_params is not in env_params.")
        obs, state = self.env.reset(key, env_params)
        target_obs = self.filter_obs(obs)
        return target_obs, state | {"heuristic_params": env_params["heuristic_params"]} | {
            "last_visible_targets": {unit: LastVisibleTarget() for unit in self.env.enemy_keys}
        }

    def step(self, key, state, action):
        obs = self.env.get_obs(state["state"])
        new_last_visible_targets = {}
        # Add enemy actions based on heuristic policy
        for unit in self.env.enemy_keys:
            heuristic_key, key = jax.random.split(key)
            action[unit], new_last_visible_targets[unit] = heuristic_policy(
                heuristic_key,
                obs[unit],
                state["last_visible_targets"][unit],
                self.env.num_agents,
                self.env.max_n_zone,
                state["heuristic_params"],
                state["physics_params"],
            )
        obs, next_state, reward, done, info = self.env.step(key, state, action)
        target_obs = self.filter_obs(obs)
        done = self.filter_obs(done) | {"__all__": done["__all__"]}
        rewards = {agent: reward[0] for agent in self.agents}
        rewards["__all__"] = reward[0]

        return (
            target_obs,
            next_state
            | {"heuristic_params": state["heuristic_params"]}
            | {"last_visible_targets": new_last_visible_targets},
            rewards,
            done,
            info,
        )

    def get_avail_actions(self, state):
        avail_actions = self.env.get_avail_actions(state)
        return {agent: avail_actions[agent] for agent in self.env.ally_keys}


class TABXHeuristicWrapper(BaseWrapper):
    """
    Wrapper for TABX that adds heuristic policy to specified units.

    This wrapper allows certain units to be controlled by a heuristic policy instead of
    requiring manual actions. It can filter observations to only return non-heuristic
    units and optionally filter rewards to only ally teams.

    Args:
        env: TABX environment to wrap
        heuristic_units: Units to control with heuristic policy
            - "all": All units in the environment
            - "enemy": Only enemy units
            - List[str]: Specific list of unit keys
        heuristic_obs: Whether to include heuristic units in observation output
            - True: Return observations for all units
            - False: Return observations only for non-heuristic units
        only_ally_reward: Whether to filter rewards to ally team only
            - True: Return rewards only for ally team
            - False: Return rewards for all teams

    Returns:
        Wrapped environment with heuristic policy applied to specified units
    """

    def __init__(
        self,
        env: TABX,
        heuristic_units: List[str] | str = "enemy",
        heuristic_config: TABXHeuristicParam = TABXHeuristicParam(),
        heuristic_obs: bool = False,
        only_ally_reward: bool = True,
    ):
        super().__init__(env)

        if isinstance(heuristic_units, str):
            if heuristic_units == "all":
                self.heuristic_units = self.env.ally_keys + self.env.enemy_keys
            elif heuristic_units == "enemy":
                self.heuristic_units = self.env.enemy_keys
            else:
                raise ValueError(f"Invalid heuristic units: {heuristic_units}")
        elif isinstance(heuristic_units, list):
            self.heuristic_units = heuristic_units
        else:
            raise ValueError(f"Invalid heuristic units: {heuristic_units}")

        self.heuristic_config = heuristic_config
        self.heuristic_obs = heuristic_obs
        self.only_ally_reward = only_ally_reward
        self.non_heuristic_units = [
            unit for unit in self.env.unit_keys if unit not in self.heuristic_units
        ]

    def filter_obs(self, obs):
        target_obs = {}
        for unit in self.non_heuristic_units:
            target_obs[unit] = obs[unit]
        return target_obs

    def reset(self, key, senario: VectorizedScenario):
        obs, state = self.env.reset(key, senario)
        target_obs = self.filter_obs(obs)
        return target_obs, state

    def step(self, key, state, action):
        obs = self.env.get_obs(state)
        # Add enemy actions based on heuristic policy
        for unit in self.heuristic_units:
            heuristic_key, key = jax.random.split(key)
            action[unit] = heuristic_policy(
                heuristic_key,
                obs[unit],
                self.env.num_agents,
                self.heuristic_config,
            )
        obs, next_state, reward, done, info = self.env.step(key, state, action)
        target_obs = self.filter_obs(obs)
        if self.only_ally_reward:
            reward = reward[0]
        return target_obs, next_state, reward, done, info


class TABXAutoResetWrapper(BaseWrapper):
    """
    Wrapper for TABX that adds automatic reset functionality.
    """

    def __init__(self, env: TABX):
        super().__init__(env)

    def step(self, key, state, action, env_params: Dict[str, Any] = None):
        reset_obs, reset_state = self.reset(key, env_params)
        next_obs, next_state, reward, done, info = self.env.step(key, state, action)
        ep_done = done["__all__"]
        next_env_state = jax.tree.map(
            lambda x, y: jnp.where(ep_done, x, y),
            reset_state,
            next_state,
        )

        # log state is not reset
        if "log_state" in next_state:
            next_env_state["log_state"] = next_state["log_state"]

        next_obs = jax.tree.map(
            lambda x, y: jnp.where(ep_done, x, y),
            reset_obs,
            next_obs,
        )
        return next_obs, next_env_state, reward, done, info


@struct.dataclass
class LogEnvState:
    episode_returns: chex.Array
    episode_lengths: chex.Array
    returned_episode_returns: chex.Array
    returned_episode_lengths: chex.Array
    returned_episode_wins: chex.Array
    first_kills: chex.Array
    cumulative_is_attackings: chex.Array
    cumulative_damage_dealts: chex.Array
    cumulative_attack_success: chex.Array
    returned_first_kills: chex.Array
    returned_attack_success_rates: chex.Array
    returned_cumulative_is_attackings: chex.Array
    returned_cumulative_damage_dealts: chex.Array
    returned_cumulative_attack_success: chex.Array


class TABXEnemyAllyFlipWrapper(BaseWrapper):
    def __init__(self, env: TABX):
        super().__init__(env)
        self.ally_keys = env.enemy_keys
        self.enemy_keys = env.ally_keys
        self.agents = self.ally_keys + self.enemy_keys

    def step(self, key, state, action):
        obs, next_state, reward, done, info = self.env.step(key, state, action)

        fliped_reward = reward[::-1]

        n_ally = len(self.env.ally_keys)

        def get_fliped_info(info: jnp.ndarray):
            return jnp.concatenate([info[n_ally:], info[:n_ally]])

        fliped_info = {
            "damage_dealt": get_fliped_info(info["damage_dealt"]),
            "disabled_units": get_fliped_info(info["disabled_units"]),
            "done_reward": info["done_reward"][::-1],
            "is_attacking": get_fliped_info(info["is_attacking"]),
            "is_win": info["is_win"][::-1],
            "team_dead_units": info["team_dead_units"][::-1],
            "timestep": info["timestep"],
            "truncation": info["truncation"],
        }

        return obs, next_state, fliped_reward, done, fliped_info


# ref : https://github.com/FLAIROx/JaxMARL/blob/main/jaxmarl/wrappers/baselines.py
class TABXLogWrapper(BaseWrapper):
    """
    Wrapper for TABX that logs the episode returns, lengths, and wins.

    Note: When done but no reset occurs, returned_episode_returns and lengths may not be accurate. Set reset_when_done to False if you don't want to reset when done.
    """

    def __init__(self, env: TABX, reset_when_done: bool = True):
        super().__init__(env)

        self.reset_when_done = reset_when_done

    def reset(self, key, env_params: Dict[str, Any]):
        obs, state = self.env.reset(key, env_params)

        log_state = LogEnvState(
            episode_returns=jnp.zeros((self.env.max_team,)),
            episode_lengths=jnp.zeros((self.env.max_team,)),
            returned_episode_returns=jnp.zeros((self.env.max_team,)),
            returned_episode_lengths=jnp.zeros((self.env.max_team,)),
            returned_episode_wins=jnp.zeros((self.env.max_team,)),
            first_kills=jnp.zeros((self.env.max_team,)),
            cumulative_is_attackings=jnp.zeros((self.env.max_n_ally + self.env.max_n_enemy,)),
            cumulative_damage_dealts=jnp.zeros((self.env.max_n_ally + self.env.max_n_enemy,)),
            cumulative_attack_success=jnp.zeros((self.env.max_n_ally + self.env.max_n_enemy,)),
            returned_first_kills=jnp.zeros((self.env.max_team,)),
            returned_attack_success_rates=jnp.zeros((self.env.max_n_ally + self.env.max_n_enemy,)),
            returned_cumulative_is_attackings=jnp.zeros(
                (self.env.max_n_ally + self.env.max_n_enemy,)
            ),
            returned_cumulative_damage_dealts=jnp.zeros(
                (self.env.max_n_ally + self.env.max_n_enemy,)
            ),
            returned_cumulative_attack_success=jnp.zeros(
                (self.env.max_n_ally + self.env.max_n_enemy,)
            ),
        )

        return obs, state | {"log_state": log_state}

    def step(self, key, _state: Dict[str, Any], action):
        obs, next_state, reward, done, info = self.env.step(key, _state, action)

        ep_done = done["__all__"]

        state = _state["log_state"]

        if self.reset_when_done:
            new_episode_return = state.episode_returns + reward
            net_epsiode_length = state.episode_lengths + 1

            team_dead_units = info["team_dead_units"]

            new_first_kill = jnp.where(
                jnp.any(state.first_kills), state.first_kills, team_dead_units > 0
            )
            new_cumulative_is_attackings = state.cumulative_is_attackings + info["is_attacking"]
            new_cumulative_damage_dealts = state.cumulative_damage_dealts + info["damage_dealt"]
            new_cumulative_attack_success = state.cumulative_attack_success + (
                jnp.abs(info["damage_dealt"]) > 1e-6
            )

            attack_success_rates = new_cumulative_attack_success / new_cumulative_is_attackings

            log_state = LogEnvState(
                episode_returns=new_episode_return * (1 - ep_done),
                episode_lengths=net_epsiode_length * (1 - ep_done),
                returned_episode_returns=state.returned_episode_returns * (1 - ep_done)
                + new_episode_return * ep_done,
                returned_episode_lengths=state.returned_episode_lengths * (1 - ep_done)
                + net_epsiode_length * ep_done,
                returned_episode_wins=info["is_win"] * ep_done
                + state.returned_episode_wins * (1 - ep_done),
                first_kills=new_first_kill * (1 - ep_done),
                cumulative_is_attackings=new_cumulative_is_attackings * (1 - ep_done),
                cumulative_damage_dealts=new_cumulative_damage_dealts * (1 - ep_done),
                cumulative_attack_success=new_cumulative_attack_success * (1 - ep_done),
                returned_first_kills=state.returned_first_kills * (1 - ep_done)
                + new_first_kill * ep_done,
                returned_attack_success_rates=jnp.where(
                    ep_done, attack_success_rates, state.returned_attack_success_rates
                ),
                returned_cumulative_is_attackings=state.returned_cumulative_is_attackings
                * (1 - ep_done)
                + new_cumulative_is_attackings * ep_done,
                returned_cumulative_damage_dealts=state.returned_cumulative_damage_dealts
                * (1 - ep_done)
                + new_cumulative_damage_dealts * ep_done,
                returned_cumulative_attack_success=state.returned_cumulative_attack_success
                * (1 - ep_done)
                + new_cumulative_attack_success * ep_done,
            )

        else:
            new_episode_return = state.episode_returns + reward * (1 - ep_done)
            net_epsiode_length = state.episode_lengths + 1 * (1 - ep_done)
            new_cumulative_is_attackings = state.cumulative_is_attackings + info["is_attacking"]
            new_cumulative_damage_dealts = state.cumulative_damage_dealts + info["damage_dealt"]
            team_dead_units = info["team_dead_units"]
            new_cumulative_attack_success = state.cumulative_attack_success + (
                jnp.abs(info["damage_dealt"]) > 1e-6
            )
            attack_success_rates = new_cumulative_attack_success / new_cumulative_is_attackings

            new_first_kill = jnp.where(
                jnp.any(state.first_kills), state.first_kills, team_dead_units > 0
            )

            log_state = LogEnvState(
                episode_returns=new_episode_return,
                episode_lengths=net_epsiode_length,
                returned_episode_returns=new_episode_return,
                returned_episode_lengths=net_epsiode_length,
                returned_episode_wins=info["is_win"].astype(jnp.float32),
                first_kills=new_first_kill,
                cumulative_is_attackings=new_cumulative_is_attackings,
                cumulative_damage_dealts=new_cumulative_damage_dealts,
                cumulative_attack_success=new_cumulative_attack_success,
                returned_first_kills=new_first_kill,
                returned_attack_success_rates=attack_success_rates,
                returned_cumulative_is_attackings=new_cumulative_is_attackings,
                returned_cumulative_damage_dealts=new_cumulative_damage_dealts,
                returned_cumulative_attack_success=new_cumulative_attack_success,
            )

        info["episode_returns"] = log_state.episode_returns
        info["episode_lengths"] = log_state.episode_lengths
        info["returned_episode_returns"] = log_state.returned_episode_returns
        info["returned_episode_lengths"] = log_state.returned_episode_lengths
        info["returned_episode_wins"] = log_state.returned_episode_wins
        info["returned_first_kills"] = log_state.returned_first_kills
        info["returned_attack_success_rates"] = log_state.returned_attack_success_rates
        info["returned_cumulative_is_attackings"] = log_state.returned_cumulative_is_attackings
        info["returned_cumulative_damage_dealts"] = log_state.returned_cumulative_damage_dealts

        return obs, next_state | {"log_state": log_state}, reward, done, info
