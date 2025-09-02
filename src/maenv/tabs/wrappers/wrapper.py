from typing import Dict, Any
from src.maenv.tabs.tabs_battle_simulator.tabs_battle_simulator import TABSBattleSimulator
from src.maenv.tabs.tabs_battle_simulator.heuristic_policy import heuristic_policy
from src.maenv.tabs.scenarios import Scenario
import jax.numpy as jnp
import chex
import jax
from flax import struct
from typing import List


class BattleSimulatorWrapper:
    def __getattr__(self, name: str):
        return getattr(self.env, name)

    def __init__(self, env: TABSBattleSimulator):
        self.env = env


class BattleSimulatorHeuristicWrapper(BattleSimulatorWrapper):
    """
    Wrapper for BattleSimulator that adds heuristic policy to the units.
    heuristic_units: List[str] | str, epsilon: float = 0.1
    heuristic_units: "all" | "enemy" | List[str]
    epsilon: float = 0.1
    - all: all units
    - enemy: enemy units
    - List[str]: list of units
    """

    def __init__(
        self, env: TABSBattleSimulator, heuristic_units: List[str] | str, epsilon: float = 0.1
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

        self.epsilon = epsilon

    def reset(self, key, senario: Scenario):
        obs, state = self.env.reset(key, senario)
        return obs, state

    def step(self, key, state, action):
        obs = self.env.get_obs(state)
        # add actions based on heuristic policy
        for unit in self.heuristic_units:
            heuristic_key, key = jax.random.split(key)
            action[unit] = heuristic_policy(
                heuristic_key, obs[unit], self.env.num_agents, self.epsilon
            )

        obs, next_state, reward, done, info = self.env.step(key, state, action)
        return obs, next_state, reward, done, info


class BattleSimulatorAutoResetWrapper(BattleSimulatorWrapper):
    """
    Wrapper for BattleSimulator that adds automatic reset functionality.
    fixed_scenario: Scenario = None
    - None: random scenario
    - Scenario: fixed scenario if you want to use a fixed scenario. If you want to use an explicit scenario when resetting, set to None.
    """

    def __init__(self, env: TABSBattleSimulator, fixed_senario: Scenario = None):
        super().__init__(env)
        self.fixed_senario = fixed_senario

    def reset(self, key, senario: Scenario = None):
        if self.fixed_senario is None:
            obs, state = self.env.reset(key, senario)
        else:
            obs, state = self.env.reset(key, self.fixed_senario)
        return obs, state

    def step(self, key, state, action, scenario=None):
        if self.fixed_senario is None:
            reset_obs, reset_state = self.reset(key, scenario)
        else:
            reset_obs, reset_state = self.reset(key, self.fixed_senario)

        next_obs, next_state, reward, done, info = self.env.step(key, state, action)

        ep_done = done["__all__"]

        next_state = jax.tree.map(
            lambda x, y: jnp.where(ep_done, x, y),
            reset_state,
            next_state,
        )
        next_obs = jax.tree.map(
            lambda x, y: jnp.where(ep_done, x, y),
            reset_obs,
            next_obs,
        )

        return next_obs, next_state, reward, done, info


@struct.dataclass
class LogEnvState:
    env_state: Dict[str, Any]
    episode_returns: chex.Array
    episode_lengths: chex.Array
    returned_episode_returns: chex.Array
    returned_episode_lengths: chex.Array
    returned_episode_wins: chex.Array


# ref : https://github.com/FLAIROx/JaxMARL/blob/main/jaxmarl/wrappers/baselines.py
class BattleSimulatorLogWrapper(BattleSimulatorWrapper):
    """
    Wrapper for BattleSimulator that logs the episode returns, lengths, and wins.

    Note: When done but no reset occurs, returned_episode_returns and lengths may not be accurate. Set reset_when_done to False if you don't want to reset when done.
    """

    def __init__(self, env: TABSBattleSimulator, reset_when_done: bool = True):
        super().__init__(env)

        self.reset_when_done = reset_when_done

    def reset(self, key, senario: Scenario):
        obs, state = self.env.reset(key, senario)

        log_state = LogEnvState(
            env_state=state,
            episode_returns=jnp.zeros((self.env.max_team, 1)),
            episode_lengths=jnp.zeros((self.env.max_team, 1)),
            returned_episode_returns=jnp.zeros((self.env.max_team, 1)),
            returned_episode_lengths=jnp.zeros((self.env.max_team, 1)),
            returned_episode_wins=jnp.zeros((self.env.max_team, 1)),
        )

        return obs, log_state

    def step(self, key, state: LogEnvState, action):
        obs, next_state, reward, done, info = self.env.step(key, state.env_state, action)

        ep_done = done["__all__"]

        if self.reset_when_done:
            new_episode_return = state.episode_returns + reward
            net_epsiode_length = state.episode_lengths + 1

            log_state = LogEnvState(
                env_state=next_state,
                episode_returns=new_episode_return * (1 - ep_done),
                episode_lengths=net_epsiode_length * (1 - ep_done),
                returned_episode_returns=state.returned_episode_returns * (1 - ep_done)
                + new_episode_return * ep_done,
                returned_episode_lengths=state.returned_episode_lengths * (1 - ep_done)
                + net_epsiode_length * ep_done,
                returned_episode_wins=info["done_reward"] * ep_done
                + state.returned_episode_wins * (1 - ep_done),
            )

        else:
            new_episode_return = state.episode_returns + reward * (1 - ep_done)
            net_epsiode_length = state.episode_lengths + 1 * (1 - ep_done)

            log_state = LogEnvState(
                env_state=next_state,
                episode_returns=new_episode_return,
                episode_lengths=net_epsiode_length,
                returned_episode_returns=state.returned_episode_returns,
                returned_episode_lengths=state.returned_episode_lengths,
                returned_episode_wins=info["done_reward"].astype(jnp.float32),
            )

        info["episode_returns"] = log_state.episode_returns
        info["episode_lengths"] = log_state.episode_lengths
        info["returned_episode_returns"] = log_state.returned_episode_returns
        info["returned_episode_lengths"] = log_state.returned_episode_lengths
        info["returned_episode_wins"] = log_state.returned_episode_wins

        return obs, log_state, reward, done, info
