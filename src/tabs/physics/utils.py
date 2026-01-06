import json
from collections import namedtuple
from pathlib import Path
from typing import List

import jax
import jax.numpy as jnp

from src.tabs.physics.components import BoxCollider, CircleCollider
from src.tabs.physics.constants import PHYSICS_PARAMS
from src.tabs.physics.params import PhysicsParams


def circle_circle_normal(circle_a, circle_b):
    center_a = circle_a.transform.position
    center_b = circle_b.transform.position
    delta = center_b - center_a
    distance_squared = jnp.sum(delta**2)
    distance = jnp.sqrt(distance_squared)
    combined_radius = circle_a.collider.radius + circle_b.collider.radius

    depth = combined_radius - distance
    normal = jnp.where(distance < 1e-8, jnp.array([1.0, 0.0]), delta / distance)

    return normal, depth, distance


def circle_box_normal(circle_object, box_object):
    half_width = box_object.collider.width / 2
    half_height = box_object.collider.height / 2
    box_center = box_object.transform.position
    circle_center = circle_object.transform.position

    closest_point = jnp.clip(
        circle_center,
        box_center - jnp.concatenate([half_width, half_height]),
        box_center + jnp.concatenate([half_width, half_height]),
    )

    delta = closest_point - circle_center

    distance_squared = jnp.sum(delta**2)
    radius = circle_object.collider.radius

    distance = jnp.sqrt(distance_squared)
    normal = jnp.where(distance < 1e-20, jnp.array([1.0, 0.0]), delta / distance)
    depth = radius - distance

    return normal, depth, distance


def get_normal(object_a, object_b):
    if isinstance(object_a.collider, CircleCollider) and isinstance(
        object_b.collider, CircleCollider
    ):
        return circle_circle_normal(object_a, object_b)
    elif isinstance(object_a.collider, CircleCollider) and isinstance(
        object_b.collider, BoxCollider
    ):
        return circle_box_normal(object_a, object_b)
    elif isinstance(object_a.collider, BoxCollider) and isinstance(
        object_b.collider, CircleCollider
    ):
        normal, depth, distance = circle_box_normal(object_b, object_a)
        return normal, depth, distance
    else:
        raise ValueError(
            f"Unsupported collider types: {type(object_a.collider)} and {type(object_b.collider)}"
        )


def resolve_collision_pure_concise(config, object_a, object_b):
    try:
        normal, depth, _ = get_normal(object_a, object_b)
    except ValueError:
        Warning(f"Collision between {object_a} and {object_b} failed")
        return object_a, object_b, jnp.array(False)

    if not (hasattr(object_a, "rigidbody") and hasattr(object_b, "rigidbody")):
        return object_a, object_b, depth > 0

    rb_a = object_a.rigidbody
    rb_b = object_b.rigidbody
    v_a = object_a.rigidbody.velocity
    v_b = object_b.rigidbody.velocity
    p_a = object_a.transform.position
    p_b = object_b.transform.position

    rel_velocity = v_b - v_a
    vel_along_normal = jnp.dot(rel_velocity, normal)

    percent = config.percent  # 0.5
    slop = config.slop  # 0.01
    restitution = config.restitution  # 0.8

    total_inverse_mass = 1.0 / rb_a.mass + 1.0 / rb_b.mass
    j = -(1.0 + restitution) * vel_along_normal / total_inverse_mass
    impulse = j * normal

    def compute_rigidbodies(is_collision, is_separating):
        correction_base = normal * depth * percent
        if is_collision and not is_separating:
            correction = jnp.maximum(depth - slop, 0.0) * percent * normal
        else:
            correction = correction_base if is_collision else jnp.zeros_like(correction_base)

        correction_a = correction * (1.0 / rb_a.mass) / total_inverse_mass
        correction_b = correction * (1.0 / rb_b.mass) / total_inverse_mass

        new_vel_a = v_a - (impulse / rb_a.mass if is_collision and not is_separating else 0)
        new_vel_b = v_b + (impulse / rb_b.mass if is_collision and not is_separating else 0)
        new_transform_a = object_a.transform._replace(position=p_a - correction_a)
        new_rb_a = rb_a._replace(velocity=new_vel_a, mass=rb_a.mass, acceleration=rb_a.acceleration)

        new_transform_b = object_b.transform._replace(position=p_b + correction_b)
        new_rb_b = rb_b._replace(velocity=new_vel_b, mass=rb_b.mass, acceleration=rb_b.acceleration)

        return object_a.replace(transform=new_transform_a, rigidbody=new_rb_a), object_b.replace(
            transform=new_transform_b, rigidbody=new_rb_b
        )

    object_a_no_collision, object_b_no_collision = compute_rigidbodies(False, False)
    object_a_separating, object_b_separating = compute_rigidbodies(True, True)
    object_a_collision, object_b_collision = compute_rigidbodies(True, False)

    is_collision = depth > 0
    is_separating = vel_along_normal > 0

    def select_rb(no_coll, sep, coll):
        result = jnp.where(is_separating, sep, coll)
        return jnp.where(is_collision, result, no_coll)

    final_object_a = jax.tree.map(
        select_rb, object_a_no_collision, object_a_separating, object_a_collision
    )
    final_object_b = jax.tree.map(
        select_rb, object_b_no_collision, object_b_separating, object_b_collision
    )

    final_object_a = jax.tree.map(
        lambda x, y: jnp.where(object_a.rigidbody.is_kinematic, y, x), final_object_a, object_a
    )
    final_object_b = jax.tree.map(
        lambda x, y: jnp.where(object_b.rigidbody.is_kinematic, y, x), final_object_b, object_b
    )
    return final_object_a, final_object_b, is_collision


def physics_step(config, objects, physics_sprites_target, collider_filter):
    for i in range(len(physics_sprites_target)):
        for j in range(i + 1, len(physics_sprites_target)):
            if physics_sprites_target[i] in collider_filter:
                if physics_sprites_target[j] not in collider_filter[physics_sprites_target[i]]:
                    continue
            elif physics_sprites_target[j] in collider_filter:
                if physics_sprites_target[i] not in collider_filter[physics_sprites_target[j]]:
                    continue
            else:
                continue

            disabled = False

            obj_a = objects[physics_sprites_target[i]]
            obj_b = objects[physics_sprites_target[j]]

            if hasattr(obj_a, "status") and hasattr(obj_b, "status"):
                disabled = obj_a.status.is_disabled | obj_b.status.is_disabled

            new_a, new_b, is_collision = resolve_collision_pure_concise(config, obj_a, obj_b)
            if hasattr(new_a, "on_collision"):
                new_a = new_a.on_collision(objects, is_collision, physics_sprites_target[j])
            if hasattr(new_b, "on_collision"):
                new_b = new_b.on_collision(objects, is_collision, physics_sprites_target[i])

            objects[physics_sprites_target[i]] = jax.tree.map(
                lambda x, y: jnp.where(disabled, y, x), new_a, obj_a
            )
            objects[physics_sprites_target[j]] = jax.tree.map(
                lambda x, y: jnp.where(disabled, y, x), new_b, obj_b
            )

    return objects


def physics_update(config, object):
    rigidbody: namedtuple = object.rigidbody
    rigidbody = rigidbody._replace(
        velocity=(rigidbody.velocity + rigidbody.acceleration * config.dt)
        * (1 - rigidbody.is_kinematic)
    )
    transform: namedtuple = object.transform
    transform = transform._replace(position=transform.position + rigidbody.velocity * config.dt)
    return object.replace(transform=transform, rigidbody=rigidbody)


def load_json_to_jnp(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        loaded = json.load(f)

    def to_jnp_array(obj):
        if isinstance(obj, dict):
            return {k: to_jnp_array(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return jnp.array(obj)
        else:
            return jnp.array([obj]).reshape(-1, 1)

    physics_params = to_jnp_array(loaded)

    return PhysicsParams(**physics_params)


def load_physics_params_from_json(physics_param_name: str = "default"):
    base_path = Path(__file__).resolve().parent
    if physics_param_name not in PHYSICS_PARAMS:
        raise ValueError(f"Physics param name {physics_param_name} not found in {PHYSICS_PARAMS}")
    return load_json_to_jnp(str(base_path / "parameters" / f"{physics_param_name}.json"))


def build_batched_physics_params(
    physics_param_names: List[str] | str = "default", n_repeat: int = 1
):
    if isinstance(physics_param_names, str):
        physics_param_names = [physics_param_names]
    physics_params = [
        load_physics_params_from_json(physics_param_name)
        for physics_param_name in physics_param_names
    ]
    stacked_physics_params = jax.tree.map(
        lambda *args: jnp.repeat(jnp.stack(args), axis=0, repeats=n_repeat),
        *physics_params,
    )
    return stacked_physics_params
