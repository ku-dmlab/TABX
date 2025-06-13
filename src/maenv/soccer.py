import os

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
import jax
import jax.numpy as jnp
import numpy as np
from collections import namedtuple
from src.maenv.physics import physics_update, physics_step
from src.maenv.util import notify
from src.maenv.render import Texture, Transform, RigidBody, BoxCollider, CircleCollider
import tensorflow_probability.substrates.jax as tfp


tfd = tfp.distributions
tfb = tfp.bijectors


class Action:
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3
    RIGHT_UP = 4
    RIGHT_DOWN = 5
    LEFT_UP = 6
    LEFT_DOWN = 7
    NONE = 8
    KICK = 9


action_table = (
    jnp.array(
        [
            [0, 1.0],
            [0, -1.0],
            [-1.0, 0],
            [1.0, 0],
            [1.0 / jnp.sqrt(2), 1.0 / jnp.sqrt(2)],
            [1.0 / jnp.sqrt(2), -1.0 / jnp.sqrt(2)],
            [-1.0 / jnp.sqrt(2), 1.0 / jnp.sqrt(2)],
            [-1.0 / jnp.sqrt(2), -1.0 / jnp.sqrt(2)],
            [0, 0],
            [0, 0],
        ]
    )
    * 0.2
)


class Player(
    namedtuple(
        "Player",
        [
            "transform",
            "rigidbody",
            "collider",
            "texture",
            "kick",
            "team",
            "is_dribbble",
            "init_position",
            "init_color",
            "pos_max",
            "pos_min",
        ],
    )
):
    transform: Transform
    rigidbody: RigidBody
    collider: BoxCollider
    texture: Texture
    kick: jnp.array
    team: jnp.array
    is_dribbble: jnp.array
    init_position: jnp.array
    init_color: jnp.array
    x_max: jnp.array
    x_min: jnp.array
    y_max: jnp.array
    y_min: jnp.array

    def __new__(
        cls,
        transform,
        rigidbody,
        collider,
        texture=Texture(),
        kick=jnp.array([0.0]),
        team=jnp.array([0]),
        is_dribbble=jnp.array([0.0]),
        init_position=jnp.array([0.0, 0.0]),
        init_color=jnp.array([1.0, 1.0, 1.0]),
        pos_max=jnp.array([7.0, 4.0]),
        pos_min=jnp.array([-7.0, -4.0]),
    ):
        return super().__new__(
            cls,
            transform,
            rigidbody,
            collider,
            texture,
            kick,
            team,
            is_dribbble,
            init_position,
            init_color,
            pos_max,
            pos_min,
        )

    def update(self, config):
        updated_object = physics_update(config, self).color_update()
        updated_object = updated_object._replace(
            transform=updated_object.transform._replace(
                position=jnp.clip(updated_object.transform.position, self.pos_min, self.pos_max)
            )
        )
        return updated_object

    def color_update(self):
        color = self.init_color + jnp.array((0.0, 1.0, 0.0)) * self.kick
        return self._replace(texture=self.texture.update(color, self.texture.alpha))

    def act(self, action):
        """
        0 : up
        1 : down
        2 : left
        3 : right
        4 : right up
        5 : right down
        6 : left up
        7 : left down
        8 : none
        9 : kick
        """

        next_velocity = action_table[action[0]]

        latest_is_kick = self.kick

        is_kick = (action == Action.KICK) * 1.0

        # dribble penalty
        next_velocity = next_velocity * 0.3 * (1 - self.is_dribbble) + 0.7 * next_velocity

        next_velocity += self.rigidbody.velocity * is_kick * 1.0 * (1 - latest_is_kick)

        return self._replace(
            rigidbody=self.rigidbody._replace(velocity=next_velocity), kick=is_kick
        )

    def on_collision(self, objects, is_collision, name):
        if "ball" in name:
            return self._replace(is_dribbble=1.0 * is_collision)

        return self

    def on_goal(self, objects, info):
        team, is_collision = info

        init_transform = Transform(position=self.init_position, rotation=jnp.array([0.0, 0.0]))

        next_transform = jax.tree.map(
            lambda x, y: jnp.where(is_collision, y, x), self.transform, init_transform
        )

        return self._replace(transform=next_transform)


class GoalPost(namedtuple("GoalPost", ["transform", "collider", "texture", "score", "team"])):
    transform: Transform
    collider: BoxCollider
    texture: Texture
    score: jnp.array
    team: jnp.array

    def __new__(
        cls,
        transform,
        collider,
        texture=Texture(color=jnp.array((0.1, 0.1, 0.1))),
        score=jnp.array([0]),
        team=jnp.array([0]),
    ):
        return super().__new__(cls, transform, collider, texture, score, team)

    def on_collision(self, objects, is_collision, name):
        if "ball" in name:
            objects = notify(objects, "goal", (self.team, is_collision))
            return self._replace(score=self.score + 1 * is_collision)
        else:
            return self


class StaticBox(namedtuple("StaticBox", ["transform", "rigidbody", "collider", "texture"])):
    transform: Transform
    rigidbody: RigidBody
    collider: BoxCollider
    texture: Texture

    def __new__(cls, transform, rigidbody, collider, texture=Texture()):
        return super().__new__(cls, transform, rigidbody, collider, texture)


class StaticCircle(namedtuple("StaticCircle", ["transform", "rigidbody", "collider", "texture"])):
    transform: Transform
    rigidbody: RigidBody
    collider: CircleCollider
    texture: Texture

    def __new__(cls, transform, rigidbody, collider, texture=Texture()):
        return super().__new__(cls, transform, rigidbody, collider, texture)

    def update(self, config):
        return self._replace(rigidbody=self.rigidbody._replace(velocity=jnp.array([0.0, 0.0])))


class Ball(
    namedtuple(
        "Ball",
        ["transform", "rigidbody", "collider", "texture", "v_max", "init_position"],
    )
):
    transform: Transform
    rigidbody: RigidBody
    collider: CircleCollider
    texture: Texture
    v_max: jnp.array
    init_position: jnp.array

    def __new__(
        cls,
        transform,
        rigidbody,
        collider,
        texture=Texture(),
        v_max=jnp.array([1.0]),
        init_position=jnp.array([0.0, 0.0]),
    ):
        return super().__new__(cls, transform, rigidbody, collider, texture, v_max, init_position)

    def update(self, config):
        updated_object = physics_update(config, self)
        velocity = updated_object.rigidbody.velocity
        # friction
        velocity = velocity * 0.99
        updated_object = updated_object._replace(
            rigidbody=updated_object.rigidbody._replace(
                velocity=jnp.clip(velocity, -self.v_max, self.v_max)
            )
        )

        return updated_object

    def on_collision(self, objects, is_collision, name):
        if "agent" in name:
            velocity = self.rigidbody.velocity
            kick = objects[name].kick
            distnace = self.transform.position - objects[name].transform.position
            notify(objects, "ball_kick", (name, kick, is_collision))
            return self._replace(
                rigidbody=self.rigidbody._replace(
                    velocity=jnp.clip(
                        velocity + 1.0 * (distnace) * is_collision * kick,
                        -self.v_max,
                        self.v_max,
                    )
                )
            )
        if "goal_post" in name:
            init_transform = Transform(
                position=jnp.array([0.0, 0.0]), rotation=jnp.array([0.0, 0.0])
            )
            zero_velocity = self.rigidbody._replace(
                velocity=jnp.array([0.0, 0.0]), acceleration=jnp.array([0.0, 0.0])
            )
            next_transform = jax.tree.map(
                lambda x, y: jnp.where(is_collision, y, x),
                self.transform,
                init_transform,
            )
            next_velocity = jax.tree.map(
                lambda x, y: jnp.where(is_collision, y, x),
                self.rigidbody,
                zero_velocity,
            )
            return self._replace(transform=next_transform, rigidbody=next_velocity)

        return self


class GameManager(namedtuple("GameManager", ["reward", "done", "timestep"])):
    reward: jnp.array
    done: jnp.array
    timestep: jnp.array

    def update(self, config):
        return self._replace(
            reward={
                "agent_0": jnp.array([0.0]),
                "agent_1": jnp.array([0.0]),
                "agent_2": jnp.array([0.0]),
                "agent_3": jnp.array([0.0]),
            },
            done=jnp.array([False]),
            timestep=self.timestep + 1,
        )

    def on_ball_kick(self, objects, info):
        name, kick, is_collision = info
        reward = {
            "agent_0": jnp.array([0.0]),
            "agent_1": jnp.array([0.0]),
            "agent_2": jnp.array([0.0]),
            "agent_3": jnp.array([0.0]),
        }
        reward[name] = kick * 0.1 * is_collision * 1.0

        next_reward = jax.tree.map(lambda x, y: x + y, self.reward, reward)
        return self._replace(reward=next_reward)

    def on_goal(self, objects, info):
        team, is_collision = info
        left_win = ((team == 1) * is_collision) * 10
        right_win = ((team == 0) * is_collision) * 10

        left_team_reward = left_win - right_win
        right_team_reward = right_win - left_win
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
        next_reward = jax.tree.map(lambda x, y: x + y, self.reward, reward)
        return self._replace(reward=next_reward, done=done | self.done)


class Soccer:
    def __init__(self, config):
        self.config = config

    def get_obs(self, state):
        """
        Team 0   | Team 1
        Agent_0 | | Agent_2
        Agent_1 | | Agent_3
        """

        ball_position = state["ball"].transform.position
        ball_velocity = state["ball"].rigidbody.velocity
        agent_0_position = state["agent_0"].transform.position
        agent_1_position = state["agent_1"].transform.position
        agent_2_position = state["agent_2"].transform.position
        agent_3_position = state["agent_3"].transform.position

        symmetry_vector = jnp.array([-1.0, 1.0])
        agent_0_obs = jnp.concatenate(
            [
                ball_position,
                ball_velocity,
                agent_0_position,
                agent_1_position,
                agent_2_position,
                agent_3_position,
            ]
        )
        agent_1_obs = jnp.concatenate(
            [
                ball_position,
                ball_velocity,
                agent_1_position,
                agent_0_position,
                agent_3_position,
                agent_2_position,
            ]
        )
        agent_2_obs = jnp.concatenate(
            [
                ball_position * symmetry_vector,
                ball_velocity * symmetry_vector,
                agent_2_position * symmetry_vector,
                agent_3_position * symmetry_vector,
                agent_0_position * symmetry_vector,
                agent_1_position * symmetry_vector,
            ]
        )
        agent_3_obs = jnp.concatenate(
            [
                ball_position * symmetry_vector,
                ball_velocity * symmetry_vector,
                agent_3_position * symmetry_vector,
                agent_2_position * symmetry_vector,
                agent_1_position * symmetry_vector,
                agent_0_position * symmetry_vector,
            ]
        )

        return {
            "agent_0": agent_0_obs,
            "agent_1": agent_1_obs,
            "agent_2": agent_2_obs,
            "agent_3": agent_3_obs,
        }

    def reset(self, key, env_params=None):
        ball_init_distribution = tfd.Uniform(
            low=jnp.array([-6.5, -3.5]), high=jnp.array([6.5, 3.5])
        )
        ball_init_position = ball_init_distribution.sample(seed=key)
        ball = Ball(
            transform=Transform(position=ball_init_position, rotation=jnp.array([0.0, 0.0])),
            rigidbody=RigidBody(
                mass=jnp.array([10.0]),
                is_kinematic=jnp.array([False]),
            ),
            collider=CircleCollider(radius=jnp.array([0.15])),
            texture=Texture(),
            init_position=jnp.array([-0.5, 0.4]),
        )

        agent_0 = Player(
            transform=Transform(position=jnp.array([-2.0, 2.0]), rotation=jnp.array([0.0, 0.0])),
            init_position=jnp.array([-2.0, 2.0]),
            rigidbody=RigidBody(
                mass=jnp.array([1.0]),
            ),
            collider=CircleCollider(radius=jnp.array([0.2])),
            texture=Texture(),
            team=jnp.array([0]),
            init_color=jnp.array([1.0, 0.0, 0.0]),
        )

        agent_1 = Player(
            transform=Transform(position=jnp.array([-2.0, -2.0]), rotation=jnp.array([0.0, 0.0])),
            init_position=jnp.array([-2.0, -2.0]),
            rigidbody=RigidBody(
                mass=jnp.array([1.0]),
            ),
            collider=CircleCollider(radius=jnp.array([0.2])),
            texture=Texture(),
            team=jnp.array([0]),
            init_color=jnp.array([1.0, 0.0, 0.0]),
        )

        agent_2 = Player(
            transform=Transform(position=jnp.array([2.0, 2.0]), rotation=jnp.array([0.0, 0.0])),
            init_position=jnp.array([2.0, 2.0]),
            rigidbody=RigidBody(
                mass=jnp.array([1.0]),
            ),
            collider=CircleCollider(radius=jnp.array([0.2])),
            texture=Texture(color=jnp.array([0.0, 0.0, 1.0]), alpha=jnp.array([1.0])),
            team=jnp.array([1]),
            init_color=jnp.array([0.0, 0.0, 1.0]),
        )

        agent_3 = Player(
            transform=Transform(position=jnp.array([2.0, -2.0]), rotation=jnp.array([0.0, 0.0])),
            init_position=jnp.array([2.0, -2.0]),
            rigidbody=RigidBody(
                mass=jnp.array([1.0]),
            ),
            collider=CircleCollider(radius=jnp.array([0.2])),
            texture=Texture(color=jnp.array([0.0, 0.0, 1.0]), alpha=jnp.array([1.0])),
            team=jnp.array([1]),
            init_color=jnp.array([0.0, 0.0, 1.0]),
        )

        static_mass = jnp.array([1000000.0])
        top_wall = StaticBox(
            transform=Transform(position=jnp.array([0.0, 6.0]), rotation=jnp.array([0.0, 0.0])),
            rigidbody=RigidBody(
                is_kinematic=jnp.array([True]),
                mass=static_mass,
            ),
            collider=BoxCollider(width=jnp.array([30.0]), height=jnp.array([4.0])),
        )
        bottom_wall = StaticBox(
            transform=Transform(position=jnp.array([0.0, -6.0]), rotation=jnp.array([0.0, 0.0])),
            rigidbody=RigidBody(
                is_kinematic=jnp.array([True]),
                mass=static_mass,
            ),
            collider=BoxCollider(width=jnp.array([30.0]), height=jnp.array([4.0])),
        )

        left_top_wall = StaticBox(
            transform=Transform(position=jnp.array([-8.0, 6.0]), rotation=jnp.array([0.0, 0.0])),
            rigidbody=RigidBody(
                is_kinematic=jnp.array([True]),
                mass=static_mass,
            ),
            collider=BoxCollider(width=jnp.array([2.0]), height=jnp.array([9.0])),
        )

        left_bottom_wall = StaticBox(
            transform=Transform(position=jnp.array([-8.0, -6.0]), rotation=jnp.array([0.0, 0.0])),
            rigidbody=RigidBody(
                is_kinematic=jnp.array([True]),
                mass=static_mass,
            ),
            collider=BoxCollider(width=jnp.array([2.0]), height=jnp.array([9.0])),
        )

        right_top_wall = StaticBox(
            transform=Transform(position=jnp.array([8.0, 6.0]), rotation=jnp.array([0.0, 0.0])),
            rigidbody=RigidBody(
                is_kinematic=jnp.array([True]),
                mass=static_mass,
            ),
            collider=BoxCollider(width=jnp.array([2.0]), height=jnp.array([9.0])),
        )

        right_bottom_wall = StaticBox(
            transform=Transform(
                position=jnp.array([8.0, -6.0]),
                rotation=jnp.array([0.0, 0.0]),
            ),
            rigidbody=RigidBody(
                is_kinematic=jnp.array([True]),
                mass=static_mass,
            ),
            collider=BoxCollider(width=jnp.array([2.0]), height=jnp.array([9.0])),
        )

        left_goal_post = GoalPost(
            transform=Transform(position=jnp.array([-8.0, 0.0]), rotation=jnp.array([0.0, 0.0])),
            collider=BoxCollider(width=jnp.array([2.0]), height=jnp.array([3.0])),
            texture=Texture(jnp.array((1.0, 0.5, 0.5)), jnp.array([1.0])),
        )

        right_goal_post = GoalPost(
            transform=Transform(position=jnp.array([8.0, 0.0]), rotation=jnp.array([0.0, 0.0])),
            collider=BoxCollider(width=jnp.array([2.0]), height=jnp.array([3.0])),
            texture=Texture(jnp.array((0.5, 0.5, 1.0)), jnp.array([1.0])),
            team=jnp.array([1]),
        )

        left_top_circle = StaticCircle(
            transform=Transform(position=jnp.array([-7.0, 1.5]), rotation=jnp.array([0.0, 0.0])),
            collider=CircleCollider(radius=jnp.array([0.5])),
            texture=Texture(jnp.array((1.0, 0.5, 0.5)), jnp.array([1.0])),
            rigidbody=RigidBody(
                is_kinematic=jnp.array([True]),
                mass=static_mass,
            ),
        )

        left_bottom_circle = StaticCircle(
            transform=Transform(position=jnp.array([-7.0, -1.5]), rotation=jnp.array([0.0, 0.0])),
            collider=CircleCollider(radius=jnp.array([0.5])),
            texture=Texture(jnp.array((1.0, 0.5, 0.5)), jnp.array([1.0])),
            rigidbody=RigidBody(
                is_kinematic=jnp.array([True]),
                mass=static_mass,
            ),
        )

        right_top_circle = StaticCircle(
            transform=Transform(position=jnp.array([7.0, 1.5]), rotation=jnp.array([0.0, 0.0])),
            collider=CircleCollider(radius=jnp.array([0.5])),
            texture=Texture(jnp.array((0.5, 0.5, 1.0)), jnp.array([1.0])),
            rigidbody=RigidBody(
                is_kinematic=jnp.array([True]),
                mass=static_mass,
            ),
        )

        right_bottom_circle = StaticCircle(
            transform=Transform(position=jnp.array([7.0, -1.5]), rotation=jnp.array([0.0, 0.0])),
            collider=CircleCollider(radius=jnp.array([0.5])),
            texture=Texture(jnp.array((0.5, 0.5, 1.0)), jnp.array([1.0])),
            rigidbody=RigidBody(
                is_kinematic=jnp.array([True]),
                mass=static_mass,
            ),
        )

        gmae_manager = GameManager(
            reward={
                "agent_0": jnp.array([0.0]),
                "agent_1": jnp.array([0.0]),
                "agent_2": jnp.array([0.0]),
                "agent_3": jnp.array([0.0]),
            },
            done=jnp.array([False]),
            timestep=jnp.array([0]),
        )
        state = {
            "ball": ball,
            "agent_0": agent_0,
            "agent_1": agent_1,
            "agent_2": agent_2,
            "agent_3": agent_3,
            "top_wall": top_wall,
            "bottom_wall": bottom_wall,
            "left_goal_post": left_goal_post,
            "right_goal_post": right_goal_post,
            "game_manager": gmae_manager,
            "left_top_wall": left_top_wall,
            "left_bottom_wall": left_bottom_wall,
            "right_top_wall": right_top_wall,
            "right_bottom_wall": right_bottom_wall,
            "left_top_circle": left_top_circle,
            "left_bottom_circle": left_bottom_circle,
            "right_top_circle": right_top_circle,
            "right_bottom_circle": right_bottom_circle,
        }
        return self.get_obs(state), state

    def step(self, key, state, action):
        for sprite in state.keys():
            if hasattr(state[sprite], "update"):
                state[sprite] = state[sprite].update(self.config)

        collider_filter = {
            "ball": [
                "agent_0",
                "agent_2",
                "agent_3",
                "agent_1",
                "top_wall",
                "bottom_wall",
                "left_top_circle",
                "left_bottom_circle",
                "right_top_circle",
                "right_bottom_circle",
                "left_goal_post",
                "right_goal_post",
                "left_top_wall",
                "left_bottom_wall",
                "right_top_wall",
                "right_bottom_wall",
            ],
            "left_top_circle": ["agent_0", "agent_2", "agent_3", "agent_1", "ball"],
            "left_bottom_circle": ["agent_0", "agent_2", "agent_3", "agent_1", "ball"],
            "right_top_circle": ["agent_0", "agent_2", "agent_3", "agent_1", "ball"],
            "right_bottom_circle": ["agent_0", "agent_2", "agent_3", "agent_1", "ball"],
            "agent_0": [
                "agent_1",
                "agent_2",
                "agent_3",
                "ball",
                "left_top_circle",
                "left_bottom_circle",
                "right_top_circle",
                "right_bottom_circle",
            ],
            "agent_1": [
                "agent_0",
                "agent_2",
                "agent_3",
                "ball",
                "left_top_circle",
                "left_bottom_circle",
                "right_top_circle",
                "right_bottom_circle",
            ],
            "agent_2": [
                "agent_0",
                "agent_1",
                "agent_3",
                "ball",
                "left_top_circle",
                "left_bottom_circle",
                "right_top_circle",
                "right_bottom_circle",
            ],
            "agent_3": [
                "agent_0",
                "agent_1",
                "agent_2",
                "ball",
                "left_top_circle",
                "left_bottom_circle",
                "right_top_circle",
                "right_bottom_circle",
            ],
        }

        state = physics_step(self.config, state, list(state.keys()), collider_filter)
        for sprite in ["agent_0", "agent_1", "agent_2", "agent_3"]:
            state[sprite] = state[sprite].act(action[sprite])
        reward = state["game_manager"].reward
        agent_0_ball_distance = jnp.sqrt(
            jnp.square(state["agent_0"].transform.position - state["ball"].transform.position).sum()
        )
        agent_1_ball_distance = jnp.sqrt(
            jnp.square(state["agent_1"].transform.position - state["ball"].transform.position).sum()
        )
        agent_2_ball_distance = jnp.sqrt(
            jnp.square(state["agent_2"].transform.position - state["ball"].transform.position).sum()
        )
        agent_3_ball_distance = jnp.sqrt(
            jnp.square(state["agent_3"].transform.position - state["ball"].transform.position).sum()
        )

        left_team_distance_reward = -jnp.minimum(agent_0_ball_distance, agent_1_ball_distance)
        right_team_distance_reward = -jnp.minimum(agent_2_ball_distance, agent_3_ball_distance)

        agent_distance = {
            "agent_0": left_team_distance_reward,
            "agent_1": left_team_distance_reward,
            "agent_2": right_team_distance_reward,
            "agent_3": right_team_distance_reward,
        }

        reward = jax.tree.map(lambda x, y: x + y * 0.05, reward, agent_distance)
        done = state["game_manager"].done
        timestep = state["game_manager"].timestep

        return self.get_obs(state), state, reward, done, {"timestep": timestep}
