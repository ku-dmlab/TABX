from collections import namedtuple
import jax
import jax.numpy as jnp

from src.maenv.soccer.actions import action_table, Action
from src.maenv.physics import Transform, RigidBody, BoxCollider, CircleCollider, physics_update
from src.maenv.util import notify
from src.maenv.render import Texture


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
