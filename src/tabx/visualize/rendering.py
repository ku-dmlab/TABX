import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pygame

from src.tabx.constants import ALL_UNIT_NAMES
from src.tabx.tabx import DefaultUnit, Zone

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GRAY = (135, 135, 135)
DARK_GRAY = (50, 50, 50)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)
ORANGE = (255, 137, 34)

ALPHA = (125,)
ZONE_ALPHA = (75,)

BG_MAIN = DARK_GRAY
COLOR_TEXT = BLACK
COLOR_BAR = GRAY
COLOR_HP = RED
COLOR_COOLDOWN = GREEN
COLOR_ATTACK = ORANGE + ALPHA
COLIR_ATTACK_BORDER = ORANGE
COLOR_SIGHT = GRAY + ALPHA
COLOR_SIGHT_BORDER = GRAY

COLOR_LAVA = RED
COLOR_BUSH = GREEN
COLOR_SWAMP = BLUE
COLOR_ZONES = {1: COLOR_LAVA, 2: COLOR_BUSH, 3: COLOR_SWAMP}

COLOR_PIX_ALLY = (255, 120, 120)
COLOR_PIX_ALLY_DEAD = (76, 0, 0)
COLOR_PIX_ENEMY = (109, 233, 150)
COLOR_PIX_ENEMY_DEAD = (0, 76, 0)

WIDTH = 1920
HEIGHT = 1080

PIX_UNIT_SIZE = 17

ASSET_PATH = Path(__file__).resolve().parent.joinpath("assets", "units")

pygame.font.init()


def world_to_screen(pos: Tuple, width: int = WIDTH, height: int = HEIGHT):
    """The origin (0,0) is the center of the screen"""
    x, y = pos
    return x + width // 2, height // 2 - y


def draw_fan_sight_range(
    canvas: pygame.Surface,
    width: int,
    height: int,
    pos: Tuple,
    rotation: float,
    sight_angle: float,
    x: int = 0,
    y: int = 0,
):
    _canvas = pygame.Surface((width, height), pygame.SRCALPHA)

    # Convert sight_angle into radian
    if sight_angle > 6.28:  # Degree
        sight_angle_rad = math.radians(sight_angle)
    else:  # Radian
        sight_angle_rad = sight_angle

    if sight_angle_rad < 0.01:  # About 0.6 degree
        return

    # The maximum length of battle field
    max_length = math.sqrt(width**2 + height**2)

    # Normalize rotation angle from -π to π
    normalized_rotation = math.atan2(math.sin(rotation), math.cos(rotation))

    # Start angle and end angle of fan
    start_angle = normalized_rotation - sight_angle_rad / 2
    end_angle = normalized_rotation + sight_angle_rad / 2

    start_line_end_pos = (
        pos[0] + math.cos(start_angle) * max_length,
        pos[1] - math.sin(start_angle) * max_length,
    )
    end_line_end_pos = (
        pos[0] + math.cos(end_angle) * max_length,
        pos[1] - math.sin(end_angle) * max_length,
    )

    points = [pos, start_line_end_pos, end_line_end_pos]
    pygame.draw.polygon(_canvas, color=COLOR_SIGHT, points=points)
    pygame.draw.line(
        _canvas, color=COLOR_SIGHT_BORDER, start_pos=pos, end_pos=start_line_end_pos, width=2
    )
    pygame.draw.line(
        _canvas, color=COLOR_SIGHT_BORDER, start_pos=pos, end_pos=end_line_end_pos, width=2
    )

    # Draw on canvas
    canvas.blit(_canvas, (x, y))


def draw_rectangular_attack_range(
    canvas: pygame.Surface,
    width: int,
    height: int,
    pix_unit_size: int,
    pos: Tuple,
    rotation: float,
    attack_range: float,
    radius: float,
    x: int = 0,
    y: int = 0,
    attack_angle: float = math.pi / 4,
):
    _canvas = pygame.Surface((width, height), pygame.SRCALPHA)

    cos_half_angle = math.cos(attack_angle / 2)
    sin_half_angle = math.sin(attack_angle / 2)

    # Attack field
    # p1 = [cos, -sin] * r, p2 = [cos, sin] * r
    rx = cos_half_angle * radius
    ry = sin_half_angle * radius
    rect_points = [
        (rx, -ry),
        (rx + attack_range * pix_unit_size, -ry),
        (rx + attack_range * pix_unit_size, ry),
        (rx, ry),
    ]

    cos_rot = math.cos(rotation)
    sin_rot = math.sin(rotation)

    points = []
    for _x, _y in rect_points:
        # Rotation
        new_x = _x * cos_rot - _y * sin_rot
        new_y = _x * sin_rot + _y * cos_rot

        _x = pos[0] + new_x
        _y = pos[1] - new_y

        points.append((_x, _y))

    # Draw attack field
    pygame.draw.polygon(_canvas, COLOR_ATTACK, points)
    pygame.draw.polygon(_canvas, COLIR_ATTACK_BORDER, points, 2)

    # Draw on canvas
    canvas.blit(_canvas, (x, y))


def draw_stats_bar(
    canvas: pygame.Surface,
    pos: Tuple,
    val: float,
    max_val: float,
    fg_color: Tuple,
    radius: float,
    bar_y_offset: float,
):
    bar_width = int(2 * radius)
    bar_height = 8

    ratio = max(0, min(1, val / max_val))

    _canvas = pygame.Surface((bar_width, bar_height), pygame.SRCALPHA)
    pygame.draw.rect(_canvas, COLOR_BAR, (0, 0, bar_width, bar_height))  # Background
    pygame.draw.rect(_canvas, fg_color, (0, 0, int(bar_width * ratio), bar_height))  # Foreground

    # Draw on canvas
    canvas.blit(_canvas, (pos[0] - bar_width // 2, pos[1] - radius - bar_height - bar_y_offset))


def draw_unit(
    canvas: pygame.Surface,
    unit: DefaultUnit,
    width: int,
    height: int,
    pix_unit_size: int,
    show_attack: bool = True,
    show_stats_bar: bool = True,
    show_sight: bool = False,
    pix_unit: bool = False,
    offset: int = 10,
):
    unit_id = int(np.array(unit.status.unit_id).item())
    pos = np.array(unit.transform.position)
    pos = (pos[0].item() * pix_unit_size, pos[1].item() * pix_unit_size)
    rotation = np.array(unit.transform.rotation).item()  # radian
    radius = np.array(unit.collider.radius).item() * pix_unit_size
    team = np.array(unit.team).item()
    health = np.array(unit.status.health).item()
    max_health = np.array(unit.status.max_health).item()
    cooldown = np.array(unit.status.cooldown).item()
    max_cooldown = np.array(unit.status.attack_cooldown).item()
    attack_range = np.array(unit.status.attack_range).item()
    sight_angle = np.array(unit.status.sight_angle).item()
    is_alive = np.array(unit.status.is_alive).item()

    screen_pos = world_to_screen(pos, width, height)
    rotation_degree = rotation * 180 / math.pi

    # Draw attack
    if show_attack and unit.is_attacking:
        draw_rectangular_attack_range(
            canvas,
            width=width,
            height=height,
            pix_unit_size=pix_unit_size,
            pos=screen_pos,
            rotation=rotation,
            attack_range=attack_range,
            radius=radius,
            attack_angle=sight_angle,
        )

    portrait_surface = pygame.Surface((radius * 2 + offset, radius * 2 + offset), pygame.SRCALPHA)
    center = (radius + offset // 2, radius + offset // 2)

    # Draw units
    if team == 0:
        if is_alive:
            if pix_unit:
                pygame.draw.circle(portrait_surface, COLOR_PIX_ALLY, center, max(1, radius))
            else:
                portrait = pygame.image.load(
                    ASSET_PATH.joinpath(f"{ALL_UNIT_NAMES[unit_id]}_ally.png")
                )
        else:
            if pix_unit:
                pygame.draw.circle(portrait_surface, COLOR_PIX_ALLY_DEAD, center, max(1, radius))
            else:
                portrait = pygame.image.load(
                    ASSET_PATH.joinpath(f"{ALL_UNIT_NAMES[unit_id]}_ally_dead.png")
                )
    else:
        if is_alive:
            if pix_unit:
                pygame.draw.circle(portrait_surface, COLOR_PIX_ENEMY, center, max(1, radius))
            else:
                portrait = pygame.image.load(
                    ASSET_PATH.joinpath(f"{ALL_UNIT_NAMES[unit_id]}_enemy.png")
                )
        else:
            if pix_unit:
                pygame.draw.circle(portrait_surface, COLOR_PIX_ENEMY_DEAD, center, max(1, radius))
            else:
                portrait = pygame.image.load(
                    ASSET_PATH.joinpath(f"{ALL_UNIT_NAMES[unit_id]}_enemy_dead.png")
                )

    if not pix_unit:
        portrait = pygame.transform.scale(portrait, (2 * radius, 2 * radius))
        portrait = pygame.transform.rotate(portrait, 270 + rotation_degree)
        new_rect = portrait.get_rect(center=center)
        portrait_surface.blit(portrait, new_rect.topleft)

    # Draw on the canvas
    canvas.blit(
        portrait_surface,
        (screen_pos[0] - radius - offset // 2, screen_pos[1] - radius - offset // 2),
    )

    # Draw sight
    if show_sight:
        draw_fan_sight_range(
            canvas,
            width=width,
            height=height,
            pos=screen_pos,
            rotation=rotation,
            sight_angle=sight_angle,
        )

    # Draw health and cooldown bar
    if show_stats_bar:
        draw_stats_bar(
            canvas,
            pos=screen_pos,
            val=health,
            max_val=max_health,
            fg_color=COLOR_HP,
            radius=radius,
            bar_y_offset=12,
        )
        draw_stats_bar(
            canvas,
            pos=screen_pos,
            val=cooldown,
            max_val=max_cooldown,
            fg_color=COLOR_COOLDOWN,
            radius=radius,
            bar_y_offset=4,
        )


def draw_zone(canvas: pygame.Surface, zone: Zone, width: int, height: int, pix_unit_size: int):
    pos = np.array(zone.ellipse.position)
    pos = (pos[0].item() * pix_unit_size, pos[1].item() * pix_unit_size)
    screen_pos = world_to_screen(pos, width, height)

    axes = np.array(zone.ellipse.axes)
    axes = (axes[0].item() * 2 * pix_unit_size, axes[1].item() * 2 * pix_unit_size)

    _canvas = pygame.Surface(axes, pygame.SRCALPHA)

    pygame.draw.ellipse(
        _canvas,
        COLOR_ZONES[int(np.array(zone.zone_type).item())] + ZONE_ALPHA,
        (0, 0, axes[0], axes[1]),
    )
    pygame.draw.ellipse(
        _canvas, COLOR_ZONES[int(np.array(zone.zone_type).item())], (0, 0, axes[0], axes[1]), 2
    )

    # Draw on the canvas
    canvas.blit(_canvas, (screen_pos[0] - axes[0] // 2, screen_pos[1] - axes[1] // 2))


def get_tabx_render(
    state: Dict,
    unit_keys: List,
    zone_keys: List,
    pix_unit_size: int = PIX_UNIT_SIZE,
    show_attack: bool = True,
    show_stats_bar: bool = True,
    show_sight: bool = False,
    pix_unit: bool = False,
):
    """Return the step frame of TABX formatted as a NumPy array"""

    canvas = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
    canvas.fill(BG_MAIN)

    # Draw zones
    for zone_key in zone_keys:
        _zone = state[zone_key]
        if _zone.zone_type == 0:
            continue
        draw_zone(canvas, zone=_zone, width=WIDTH, height=HEIGHT, pix_unit_size=pix_unit_size)

    # Draw units
    for unit_key in unit_keys:
        _unit = state[unit_key]
        if _unit.status.is_disabled:
            continue
        draw_unit(
            canvas,
            unit=_unit,
            width=WIDTH,
            height=HEIGHT,
            pix_unit_size=pix_unit_size,
            show_attack=show_attack,
            show_stats_bar=show_stats_bar,
            show_sight=show_sight,
            pix_unit=pix_unit,
        )

    array = np.transpose(np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2))  # Rotate
    array = np.flip(array, axis=0)  # Flip because of matplotlib axis

    return array
