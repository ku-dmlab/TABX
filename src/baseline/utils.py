from typing import Any

import chex
import jax
import jax.numpy as jnp
from flax.training.train_state import TrainState

from src.tabs.constants import ALL_UNIT_NAMES
from src.tabs.tabs import TABS


@chex.dataclass(frozen=True)
class Timestep:
    obs: dict
    actions: dict
    rewards: dict
    dones: dict
    avail_actions: dict


class CustomTrainState(TrainState):
    target_network_params: Any
    timesteps: int = 0
    n_updates: int = 0
    grad_steps: int = 0


def batchify(x: dict, agent_list, num_actors):
    x = jnp.stack([x[a] for a in agent_list])
    return x.reshape((num_actors, -1))


def unbatchify(x: jnp.ndarray, agent_list, num_envs, num_actors):
    x = x.reshape((num_actors, num_envs, -1))
    return {a: x[i] for i, a in enumerate(agent_list)}


def get_battle_metric(env: TABS, last_state):
    log_state = last_state["log_state"]
    n_env = last_state["log_state"].returned_episode_returns.shape[0]

    battle_metric = {
        "returned_cumulative_is_attackings": log_state.returned_cumulative_is_attackings,
        "returned_cumulative_damage_dealts": log_state.returned_cumulative_damage_dealts,
        "returned_cumulative_attack_success": log_state.returned_cumulative_attack_success,
    }

    ally_is_disabled = []

    def get_is_disabled(state):
        return jnp.stack([state["state"][unit].status.is_disabled for unit in env.ally_keys])

    def get_unit_ids(state):
        return jnp.stack([state["state"][unit].status.unit_id for unit in env.ally_keys])

    ally_battle_metric = jax.tree.map(lambda x: x[:, : env.max_n_ally], battle_metric)
    ally_is_disabled = jax.vmap(get_is_disabled)(last_state)
    ally_unit_ids = jax.vmap(get_unit_ids)(last_state)

    def unit_condition_sum(target_unit_id, unit_ids, is_disabled, values):
        return (
            (unit_ids == target_unit_id) * (1 - is_disabled) * values.reshape(is_disabled.shape)
        ).sum()

    unit_battle_metric = jax.tree.map(
        lambda x: jax.vmap(unit_condition_sum, in_axes=(0, None, None, None))(
            jnp.arange(env.max_n_ally + env.max_n_enemy), ally_unit_ids, ally_is_disabled, x
        ),
        ally_battle_metric,
    )

    unit_counts = jax.vmap(unit_condition_sum, in_axes=(0, None, None, None))(
        jnp.arange(env.max_n_ally + env.max_n_enemy),
        ally_unit_ids,
        ally_is_disabled,
        jnp.ones_like(ally_is_disabled),
    )

    unit_battle_metric["attack_success_rate"] = (
        unit_battle_metric["returned_cumulative_attack_success"]
        / unit_battle_metric["returned_cumulative_is_attackings"]
    )

    unit_specific_metric = {}
    for key, value in unit_battle_metric.items():
        for i, name in enumerate(ALL_UNIT_NAMES):
            unit_specific_metric[f"{key}/{name}"] = value[i] / (
                unit_counts[i] if key != "attack_success_rate" else 1
            )

    team_fight_metric = {
        "cumulative_is_attackings": unit_battle_metric["returned_cumulative_is_attackings"].sum()
        / n_env,
        "cumulative_damage_dealts": jnp.nansum(
            unit_battle_metric["returned_cumulative_damage_dealts"]
            * (unit_battle_metric["returned_cumulative_damage_dealts"] > 0)
        )
        / n_env,
        "cumulative_heal_amount": jnp.nansum(
            unit_battle_metric["returned_cumulative_damage_dealts"]
            * (unit_battle_metric["returned_cumulative_damage_dealts"] < 0)
        )
        / n_env,
        "attack_success_rate": jnp.nansum(unit_battle_metric["returned_cumulative_attack_success"])
        / jnp.nansum(unit_battle_metric["returned_cumulative_is_attackings"]),
        "first_kill_rate": log_state.returned_first_kills[:, 0].mean(),
    }

    return (
        unit_specific_metric
        | team_fight_metric
        | {
            "episode_returns": log_state.returned_episode_returns[:, 0].mean(),
            "episode_lengths": log_state.returned_episode_lengths[:, 0].mean(),
            "episode_wins": log_state.returned_episode_wins[:, 0].mean(),
        }
    )
