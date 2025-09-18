import jax
import jax.numpy as jnp

from functools import partial
from src.baseline.algorithm.base_algo import BaseAlgo
from src.tabs.scenarios import get_vectorized_scenario, VectorizedScenario
from src.tabs.constants import ALL_UNIT_NAMES


def get_comb_metric(config, last_state):
    comb_metric = {
        "budget": last_state.budget.mean(),
    }
    for i, name in enumerate(ALL_UNIT_NAMES):
        comb_metric[f"unit_count/{name}"] = last_state.current_unit_list.mean(axis=0)[i]
    return comb_metric


def get_deploy_metric(config, last_state):
    deploy_metric = {
        "unit_deploy": last_state.battle_field[0],
    }
    return deploy_metric


def get_battle_metric(config, last_state, scenarios_bs):
    battle_metric = {
        "returned_cumulative_is_attackings": last_state.returned_cumulative_is_attackings,
        "returned_cumulative_damage_dealts": last_state.returned_cumulative_damage_dealts,
        "returned_cumulative_attack_success": last_state.returned_cumulative_attack_success,
    }

    v_vectorize_scenario = jax.vmap(
        partial(
            get_vectorized_scenario,
            n_ally=config.tabs.max_n_ally,
            n_enemy=config.tabs.max_n_enemy,
        )
    )
    vectorized_scenario: VectorizedScenario = v_vectorize_scenario(scenarios_bs)

    ally_battle_metric = jax.tree.map(lambda x: x[:, : config.tabs.max_n_ally], battle_metric)
    ally_is_disabled = vectorized_scenario.is_disabled[:, : config.tabs.max_n_ally]
    ally_unit_ids = vectorized_scenario.unit_ids[:, : config.tabs.max_n_ally]

    def unit_condition_sum(target_unit_id, unit_ids, is_disabled, values):
        return ((unit_ids == target_unit_id) * (1 - is_disabled) * values).sum()

    unit_battle_metric = jax.tree.map(
        lambda x: jax.vmap(unit_condition_sum, in_axes=(0, None, None, None))(
            jnp.arange(config.tabs.max_num_units), ally_unit_ids, ally_is_disabled, x
        ),
        ally_battle_metric,
    )

    unit_counts = jax.vmap(unit_condition_sum, in_axes=(0, None, None, None))(
        jnp.arange(config.tabs.max_num_units),
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
        / config.n_env,
        "cumulative_damage_dealts": jnp.nansum(
            unit_battle_metric["returned_cumulative_damage_dealts"]
            * (unit_battle_metric["returned_cumulative_damage_dealts"] > 0)
        )
        / config.n_env,
        "cumulative_heal_amount": jnp.nansum(
            unit_battle_metric["returned_cumulative_damage_dealts"]
            * (unit_battle_metric["returned_cumulative_damage_dealts"] < 0)
        )
        / config.n_env,
        "attack_success_rate": jnp.nansum(unit_battle_metric["returned_cumulative_attack_success"])
        / jnp.nansum(unit_battle_metric["returned_cumulative_is_attackings"]),
        "first_kill_rate": last_state.returned_first_kills[:, 0].mean(),
    }

    return unit_specific_metric | team_fight_metric


def get_metric(config, rollout_result_comb, rollout_result_deploy, rollout_result_bs, scenarios_bs):
    battle_metric = get_battle_metric(config, rollout_result_bs["last_state"], scenarios_bs)
    comb_metric = get_comb_metric(config, rollout_result_comb["last_state"])
    deploy_metric = get_deploy_metric(config, rollout_result_deploy["last_state"])
    metric = comb_metric | deploy_metric | battle_metric
    metric.update(
        {
            "episode_returns": rollout_result_bs["returned_episode_returns"][:, 0].mean(),
            "episode_lengths": rollout_result_bs["returned_episode_lengths"][:, 0].mean(),
            "episode_wins": rollout_result_bs["returned_episode_wins"][:, 0].mean(),
            "reward_sum": rollout_result_bs["common_reward"].sum() / config.n_env,
        }
    )
    return metric


def rollout_tabs(
    unit_comb_agent: BaseAlgo,
    unit_deploy_agent: BaseAlgo,
    battle_agent: BaseAlgo,
    train_state_comb,
    train_state_deploy,
    train_state_bs,
    scenarios_comb,
    config,
):
    train_state_comb, rollout_result_comb = unit_comb_agent.rollout(
        train_state_comb, scenarios_comb
    )
    scenarios_deploy = scenarios_comb.replace(
        ally_unit_comp=rollout_result_comb["last_state"].current_unit_list
    )
    train_state_deploy, rollout_result_deploy = unit_deploy_agent.rollout(
        train_state_deploy, scenarios_deploy
    )
    scenarios_bs = scenarios_deploy.replace(
        battle_field=rollout_result_deploy["last_state"].battle_field
    )
    train_state_bs, rollout_result_bs = battle_agent.rollout(train_state_bs, scenarios_bs)

    metric = get_metric(
        config, rollout_result_comb, rollout_result_deploy, rollout_result_bs, scenarios_bs
    )
    rollout_result_comb["rewards"] = (
        rollout_result_comb["dones"] * rollout_result_bs["returned_episode_returns"][None, :, 0]
    )
    rollout_result_deploy["rewards"] = (
        rollout_result_deploy["dones"] * rollout_result_bs["returned_episode_returns"][None, :, 0]
    )
    return (train_state_comb, train_state_deploy, train_state_bs), {
        "rollout_result_comb": rollout_result_comb,
        "rollout_result_deploy": rollout_result_deploy,
        "rollout_result_bs": rollout_result_bs,
        "metric": metric,
    }
