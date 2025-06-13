import os

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
import numpy as np
from collections import namedtuple
from IPython.display import HTML
from src.maenv.physics import *
from src.maenv.util import *


class Paddle(
    namedtuple(
        "Paddle",
        ["rigidbody", "collider", "action_idx", "pos_max", "pos_min", "v_max", "team", "turn"],
    )
):
    rigidbody: RigidBody
    collider: BoxCollider
    action_idx: jnp.array
    pos_max: jnp.array
    pos_min: jnp.array
    v_max: jnp.array
    turn: jnp.array  # 0, 1
    team: jnp.array  # 0, 1

    def update(self, config):
        next_rigidbody = self.rigidbody.update(config)
        next_rigidbody = next_rigidbody._replace(
            position=jnp.clip(next_rigidbody.position, self.pos_min, self.pos_max),
            velocity=jnp.clip(next_rigidbody.velocity, -self.v_max, self.v_max),
        )

        return self._replace(
            rigidbody=next_rigidbody,
            collider=self.collider,
            action_idx=self.action_idx,
            pos_max=self.pos_max,
            pos_min=self.pos_min,
            v_max=self.v_max,
        )

    def on_action(self, objects, info):
        action = info[self.action_idx]  # 0: up, 1: down
        next_velocity = jnp.array([self.rigidbody.velocity[0], (action == 1) - 0.5]) * 2.0
        return self._replace(rigidbody=self.rigidbody._replace(velocity=next_velocity))


class Ball(namedtuple("Ball", ["rigidbody", "collider", "turn", "touch_count", "v_max"])):
    rigidbody: RigidBody
    collider: CircleCollider
    touch_count: jnp.array
    turn: jnp.array
    v_max: jnp.array

    def update(self, config):
        rigidbody = self.rigidbody.update(config)
        rigidbody = rigidbody._replace(
            velocity=jnp.clip(rigidbody.velocity, -self.v_max, self.v_max)
        )
        return self._replace(rigidbody=rigidbody)

    def on_collision(self, objects, is_collision, name):
        if "agent" in name:
            is_right_turn = objects[name].turn == self.turn
            notify(
                objects,
                "ball_touch_paddle",
                (is_right_turn, objects[name].team, objects[name].turn, is_collision),
            )
            touch_count = self.touch_count + 1 * is_collision
            turn = touch_count % 2
            return self._replace(turn=turn, touch_count=touch_count)
        return self


class StaticBox(namedtuple("StaticBox", ["rigidbody", "collider"])):
    rigidbody: RigidBody
    collider: BoxCollider

    def update(self, config):
        return self


class GoalLine(namedtuple("GoalLine", ["rigidbody", "collider", "score", "team"])):
    rigidbody: RigidBody
    collider: BoxCollider
    score: jnp.array
    team: jnp.array

    def update(self, config):
        return self

    def on_collision(self, objects, is_collision, name):
        if "ball" in name:
            notify(objects, "goal", (self.team, is_collision))
            return self._replace(score=self.score + 1 * is_collision)
        else:
            return self


class GameManager(namedtuple("GameManager", ["reward", "done", "penalty", "timestep"])):
    reward: jnp.array
    done: jnp.array
    penalty: jnp.array
    timestep: jnp.array

    def update(self, config):
        return self._replace(
            reward={
                "agent_0": jnp.array(0.0),
                "agent_1": jnp.array(0.0),
                "agent_2": jnp.array(0.0),
                "agent_3": jnp.array(0.0),
            },
            done=jnp.array(False),
            timestep=self.timestep + 1,
        )

    def on_ball_touch_paddle(self, objects, info):
        """
        Teams:
        - Left team: agent_0 (left paddle) and agent_1 (right paddle)
        - Right team: agent_2 (left paddle) and agent_3 (right paddle)
        --------------------------------
        |agent |  0  |  1  |  2  |  3  |
        |-----------------------------
        | turn |  0  |  1  |  1  |  0  |
        |-----------------------------
        | team |  0  |  0  |  1  |  1  |
        ---------------------------------
        """

        valid, team, agent_turn, is_collision = info
        reward0 = (-1.0 * (~valid) * (team == 0)) * is_collision * (agent_turn == 0) * self.penalty
        reward1 = (-1.0 * (~valid) * (team == 0)) * is_collision * (agent_turn == 1) * self.penalty
        reward2 = (-1.0 * (~valid) * (team == 1)) * is_collision * (agent_turn == 1) * self.penalty
        reward3 = (-1.0 * (~valid) * (team == 1)) * is_collision * (agent_turn == 0) * self.penalty

        reward = jax.tree.map(
            lambda x, y: x + y,
            self.reward,
            {"agent_0": reward0, "agent_1": reward1, "agent_2": reward2, "agent_3": reward3},
        )
        return self._replace(reward=reward)

    def on_goal(self, objects, info):
        team, is_collision = info
        left_team_reward = (team == 1) * is_collision
        right_team_reward = (team == 0) * is_collision
        done = is_collision

        reward = jax.tree.map(
            lambda x, y: x + y,
            self.reward,
            {
                "agent_0": left_team_reward,
                "agent_1": left_team_reward,
                "agent_2": right_team_reward,
                "agent_3": right_team_reward,
            },
        )
        return self._replace(reward=reward, done=done | self.done)


class Pong:
    """
    Teams:
    - Left team: agent_0 (left paddle) and agent_1 (right paddle)
    - Right team: agent_2 (left paddle) and agent_3 (right paddle)
    """

    def __init__(self, config):
        self.config = config

    def get_obs(self, state):
        """
        --------------------------------
        |agent |  0  |  1  |  2  |  3  |
        |-----------------------------
        | turn |  0  |  1  |  1  |  0  |
        |-----------------------------
        | team |  0  |  0  |  1  |  1  |
        ---------------------------------
        """
        left_team_obs = jnp.stack(
            [
                state["ball"].rigidbody.position,
                state["ball"].rigidbody.velocity,
                state["agent_0"].rigidbody.position,
                state["agent_1"].rigidbody.position,
                state["agent_2"].rigidbody.position,
                state["agent_3"].rigidbody.position,
            ]
        )

        symmetry_vector = jnp.array([[-1.0, 1.0]])

        right_team_obs = (
            jnp.stack(
                [
                    state["ball"].rigidbody.position,
                    state["ball"].rigidbody.velocity,
                    state["agent_3"].rigidbody.position,
                    state["agent_2"].rigidbody.position,
                    state["agent_1"].rigidbody.position,
                    state["agent_0"].rigidbody.position,
                ]
            )
            * symmetry_vector
        )

        return {
            "agent_0": left_team_obs,
            "agent_1": left_team_obs,
            "agent_2": right_team_obs,
            "agent_3": right_team_obs,
        }

    def reset(self, key, env_params=None):
        ball_init_velocity = jax.random.normal(key, (2,))
        ball_init_velocity /= jnp.sqrt(jnp.square(ball_init_velocity).sum())
        ball = Ball(
            rigidbody=RigidBody(
                position=jnp.array([0.0, 0.0]),
                velocity=ball_init_velocity,
                mass=jnp.array(1.0),
                acceleration=jnp.array([0.0, 0.0]),
            ),
            collider=CircleCollider(radius=0.1),
            turn=jax.random.randint(key, (1,), 0, 2)[0],
            touch_count=jnp.array(0),
            v_max=jnp.array(1.5),
        )

        static_mass = jnp.array(1e10)

        top_wall = StaticBox(
            rigidbody=RigidBody(
                position=jnp.array([0.0, 2.5]),
                velocity=jnp.array([0.0, 0.0]),
                mass=static_mass,
                acceleration=jnp.array([0.0, 0.0]),
            ),
            collider=BoxCollider(width=10.0, height=1.0),
        )
        bottom_wall = StaticBox(
            rigidbody=RigidBody(
                position=jnp.array([0.0, -2.5]),
                velocity=jnp.array([0.0, 0.0]),
                mass=static_mass,
                acceleration=jnp.array([0.0, 0.0]),
            ),
            collider=BoxCollider(width=10.0, height=1.0),
        )

        left_wall = GoalLine(
            rigidbody=RigidBody(
                position=jnp.array([-5.0, 0.0]),
                velocity=jnp.array([0.0, 0.0]),
                mass=static_mass,
                acceleration=jnp.array([0.0, 0.0]),
            ),
            collider=BoxCollider(width=1.0, height=4.0),
            score=jnp.array(0),
            team=jnp.array(0),
        )

        right_wall = GoalLine(
            rigidbody=RigidBody(
                position=jnp.array([5.0, 0.0]),
                velocity=jnp.array([0.0, 0.0]),
                mass=static_mass,
                acceleration=jnp.array([0.0, 0.0]),
            ),
            collider=BoxCollider(width=1.0, height=4.0),
            score=jnp.array(0),
            team=jnp.array(1),
        )
        left_paddle0 = Paddle(
            rigidbody=RigidBody(
                position=jnp.array([-2.5, 0.0]),
                velocity=jnp.array([0.0, 0.0]),
                mass=static_mass,
                acceleration=jnp.array([0.0, 0.0]),
            ),
            collider=BoxCollider(width=0.1, height=0.5),
            action_idx=jnp.array(0),
            pos_max=jnp.array([-2.5, 1.75]),
            pos_min=jnp.array([-2.5, -1.75]),
            v_max=jnp.array(1.0),
            team=jnp.array(0),
            turn=jnp.array(0),
        )

        left_paddle1 = Paddle(
            rigidbody=RigidBody(
                position=jnp.array([-3.0, 0.0]),
                velocity=jnp.array([0.0, 0.0]),
                mass=static_mass,
                acceleration=jnp.array([0.0, 0.0]),
            ),
            collider=BoxCollider(width=0.1, height=0.5),
            action_idx=jnp.array(1),
            pos_max=jnp.array([-3.0, 1.75]),
            pos_min=jnp.array([-3.0, -1.75]),
            v_max=jnp.array(1.0),
            team=jnp.array(0),
            turn=jnp.array(1),
        )

        right_paddle0 = Paddle(
            rigidbody=RigidBody(
                position=jnp.array([2.5, 0.0]),
                velocity=jnp.array([0.0, 0.0]),
                mass=static_mass,
                acceleration=jnp.array([0.0, 0.0]),
            ),
            collider=BoxCollider(width=0.1, height=0.5),
            action_idx=jnp.array(2),
            pos_max=jnp.array([2.5, 1.75]),
            pos_min=jnp.array([2.5, -1.75]),
            v_max=jnp.array(1.0),
            team=jnp.array(1),
            turn=jnp.array(0),
        )

        right_paddle1 = Paddle(
            rigidbody=RigidBody(
                position=jnp.array([3.0, 0.0]),
                velocity=jnp.array([0.0, 0.0]),
                mass=static_mass,
                acceleration=jnp.array([0.0, 0.0]),
            ),
            collider=BoxCollider(width=0.1, height=0.5),
            action_idx=jnp.array(3),
            pos_max=jnp.array([3.0, 1.75]),
            pos_min=jnp.array([3.0, -1.75]),
            v_max=jnp.array(1.0),
            team=jnp.array(1),
            turn=jnp.array(1),
        )

        game_manager = GameManager(
            reward={
                "agent_0": jnp.array(0.0),
                "agent_1": jnp.array(0.0),
                "agent_2": jnp.array(0.0),
                "agent_3": jnp.array(0.0),
            },
            done=jnp.array(False),
            penalty=jnp.array(0.5),
            timestep=jnp.array(0),
        )
        env_state = {
            "ball": ball,
            "top_wall": top_wall,
            "bottom_wall": bottom_wall,
            "agent_0": left_paddle0,
            "agent_1": left_paddle1,
            "agent_2": right_paddle0,
            "agent_3": right_paddle1,
            "left_wall": left_wall,
            "right_wall": right_wall,
            "game_manager": game_manager,
        }
        return self.get_obs(env_state), env_state

    def step(self, key_step, state, action, env_params):
        state = {sprite: state[sprite].update(self.config) for sprite in state.keys()}
        state = physics_step(self.config, state, list(state.keys()))

        action_vector = jnp.concatenate(
            [action["agent_0"], action["agent_1"], action["agent_2"], action["agent_3"]]
        )
        state = notify(state, "action", action_vector)
        reward = state["game_manager"].reward
        done = state["game_manager"].done
        timestep = state["game_manager"].timestep
        return self.get_obs(state), state, reward, done, {"timestep": timestep}
