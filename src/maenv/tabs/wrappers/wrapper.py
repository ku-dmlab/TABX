from typing import Dict, Any
from src.maenv.tabs.tabs_battle_simulator.tabs_battle_simulator import BattleSimulator
from src.maenv.tabs.scenarios import Scenario
import jax.numpy as jnp
import chex
import jax
from flax import struct


class BattleSimulatorWrapper:
    def __getattr__(self, name: str):
        return getattr(self.env, name)

    def __init__(self, env: BattleSimulator):
        self.env = env


class BattleSimulatorAutoResetWrapper(BattleSimulatorWrapper):
    def __init__(self, env: BattleSimulator, fixed_senario: Scenario = None):
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
    def __init__(self, env: BattleSimulator):
        super().__init__(env)

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

        info["returned_episode_returns"] = log_state.returned_episode_returns
        info["returned_episode_lengths"] = log_state.returned_episode_lengths
        info["returned_episode_wins"] = log_state.returned_episode_wins

        return obs, log_state, reward, done, info
