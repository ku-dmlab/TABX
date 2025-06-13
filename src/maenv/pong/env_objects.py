from collections import namedtuple
import jax.numpy as jnp

from src.maenv.physics import RigidBody, BoxCollider, CircleCollider
from src.maenv.util import notify


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
