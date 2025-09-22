from typing import List, Tuple, Dict
import math
import pygame
import numpy as np

from src.tabs.constants import ALL_UNIT_NAMES
from src.tabs.tabs_unit_comb.tabs_unit_comb import State as CombState
from src.tabs.tabs_unit_deploy.tabs_unit_deploy import State as DeployState
from src.tabs.tabs_battle_simulator.tabs_battle_simulator import DefaultUnit
from src.tabs.units import get_all_unit_spec

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GRAY = (135, 135, 135)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)
ORANGE = (255, 137, 34)

ALPHA = (125,)

BG_MAIN = WHITE
COLOR_TEXT = BLACK
COLOR_BAR = GRAY
COLOR_HP = RED
COLOR_COOLDOWN = GREEN
COLOR_ATTACK = ORANGE + ALPHA
COLIR_ATTACK_BORDER = ORANGE
COLOR_SIGHT = GRAY + ALPHA
COLOR_SIGHT_BORDER = GRAY

COLOR_PIX_ALLY = (255, 120, 120)
COLOR_PIX_ALLY_DEAD = (76, 0, 0)
COLOR_PIX_ENEMY = (109, 233, 150)
COLOR_PIX_ENEMY_DEAD = (0, 76, 0)

WIDTH = 640
HEIGHT = 480

PIX_UNIT_SIZE = 11

pygame.font.init()

all_spec = get_all_unit_spec()


def draw_text(
    canvas: pygame.Surface,
    text: str,
    width: int,
    height: int,
    x: int,
    y: int,
    size: int = 20,
    align_center: bool = True,
):
    _canvas = pygame.Surface((width, height), pygame.SRCALPHA)

    font = pygame.font.SysFont(name=None, size=size, bold=False)
    text_width, text_height = font.size(text)
    _text = font.render(text, True, COLOR_TEXT, None)
    if align_center:
        # Align center
        _canvas.blit(_text, (width // 2 - text_width // 2, height // 2 - text_height // 2 + 1))
    else:
        # Align right
        _canvas.blit(_text, (width - text_width - 10, height // 2 - text_height // 2 + 1))

    canvas.blit(_canvas, (x, y))


def draw_comb_unit_list(
    canvas: pygame.Surface,
    unit_list: List,
    price: List,
    width: int,
    height: int,
    x: int,
    y: int,
    size: int = 20,
):
    _canvas = pygame.Surface((width, height), pygame.SRCALPHA)

    font = pygame.font.SysFont(name=None, size=size, bold=False)

    # total price
    total_price = str(int(sum(unit_list * price)))
    text_width, text_height = font.size(total_price)
    _text = font.render(total_price, True, COLOR_TEXT, None)
    # Align right
    _canvas.blit(_text, (width - text_width - 10, text_height // 2 + 1))

    _w_offset, _h_offset = 4, 54
    _w, _h_name, _h = 86, 21, 31
    for idx, n in enumerate(unit_list):
        _text = font.render(str(n), True, COLOR_TEXT, None)
        # Align center
        _x = _w_offset + _w // 2 - text_width // 2 + _w * (idx % 3) + 10
        _y = _h_offset + _h_name * (idx // 3 + 1) + _h // 2 + _h * (idx // 3 - 1) - (idx // 3 - 1)
        _canvas.blit(_text, (_x, _y))

    canvas.blit(_canvas, (x, y))


def draw_catalog(
    canvas: pygame.Surface, all_spec: List, width: int, height: int, x: int, y: int, size: int = 20
):
    _canvas = pygame.Surface((width, height), pygame.SRCALPHA)

    font = pygame.font.SysFont(name=None, size=size, bold=False)
    stat_font = pygame.font.SysFont(name=None, size=14, bold=False)

    n_units = len(all_spec["prices"])
    prices = all_spec["prices"]
    stats_list = [
        "healths",
        "body_radiuses",
        "body_weights",
        "speeds",
        "attack_damages",
        "attack_ranges",
        "attack_cooldown",
    ]

    _w_pad, _h_pad = 3, 4
    _width, _height = 117, 152  # unit spec canvas size
    _x_portrait, _y_portrait = 33, 6  # (x, y) in unit spec canvas window
    _stats_font_size = 8
    for idx in range(n_units):
        _unit = pygame.Surface((_width, _height), pygame.SRCALPHA)
        # Draw portrait
        portrait = pygame.image.load(f"./assets/units/{ALL_UNIT_NAMES[idx]}.png")
        portrait = pygame.transform.scale(portrait, (46, 46))
        _unit.blit(portrait, (_x_portrait, _y_portrait))

        # Draw price
        price = str(prices[idx])
        text_width, text_height = font.size(price)
        _text = font.render(price, True, COLOR_TEXT, None)
        _unit.blit(_text, (60 - text_width // 2, 63 - text_height // 2))
        coin = pygame.image.load("./assets/coin_small.png")
        _unit.blit(coin, (32, 65 - text_height // 2))

        # Draw stats
        for jdx, stats in enumerate(stats_list):
            value = str(all_spec[stats][idx])
            text_width, _ = font.size(value)
            _text = stat_font.render(value, True, COLOR_TEXT, None)
            if jdx < 4:
                # Physical stats
                _y = 77 + (_stats_font_size + 2) * jdx
            else:
                # Attack stats
                _y = 118 + (_stats_font_size + 2) * (jdx - 4)
            _unit.blit(_text, (94 - text_width // 2, _y))

        # Draw unit spec
        _x = _width * (idx % 3) + _w_pad * (idx % 3 + 1)
        _y = _height * (idx // 3) + _h_pad * (idx // 3) - 1
        _canvas.blit(_unit, (_x, _y))

    canvas.blit(_canvas, (x, y))


def draw_remaining_units(
    canvas: pygame.Surface,
    remaining_units: List,
    width: int,
    height: int,
    x: int,
    y: int,
    size: int = 20,
):
    _canvas = pygame.Surface((width, height), pygame.SRCALPHA)

    font = pygame.font.SysFont(name=None, size=size, bold=False)

    _w, _h = 54, 24  # width and height of text box
    for idx, n in enumerate(remaining_units):
        text_width, text_height = font.size(str(n))
        _text = font.render(str(n), True, COLOR_TEXT, None)
        # Align center
        _x = _w // 2 - text_width // 2 + _w * idx
        _y = _h // 2 - text_height // 2 + 6
        _canvas.blit(_text, (_x, _y))

    canvas.blit(_canvas, (x, y))


def draw_deployment(
    canvas: pygame.Surface,
    battle_field: List,
    body_radiuses: List,
    space_occupied: List,
    is_ally: bool,
    width: int,
    height: int,
    x: int,
    y: int,
):
    _canvas = pygame.Surface((height, width), pygame.SRCALPHA)  # Because of rotation

    n_row, n_col = battle_field.shape
    _w, _h = 69, 69  # width and height of unit space
    _portrait_pad = 37
    for i in range(n_row):
        for j in range(n_col):
            unit = int(battle_field[i, j])  # unit id
            if unit == 0:
                continue

            w = int(_w * np.sqrt(space_occupied[unit - 1]))
            h = int(_h * np.sqrt(space_occupied[unit - 1]))
            _unit = pygame.Surface((w, h), pygame.SRCALPHA)

            # Rotate battle_field facing each other
            if is_ally:
                portrait = pygame.image.load(f"./assets/units/{ALL_UNIT_NAMES[unit - 1]}_ally.png")
            else:
                portrait = pygame.image.load(f"./assets/units/{ALL_UNIT_NAMES[unit - 1]}_enemy.png")

            # Scale by body radius
            size = (
                int((_w - _portrait_pad) * body_radiuses[unit - 1]),
                int((_h - _portrait_pad) * body_radiuses[unit - 1]),
            )
            portrait = pygame.transform.scale(portrait, size)
            _unit.blit(portrait, ((w - size[0]) // 2, (h - size[1]) // 2))

            # Draw unit spec
            _x = _w * j
            _y = _h * i
            _canvas.blit(_unit, (_x, _y))

    # Rotate battle_field facing each other
    if is_ally:
        _canvas = pygame.transform.rotate(_canvas, 270)
    else:
        _canvas = pygame.transform.rotate(_canvas, 90)

    canvas.blit(_canvas, (x, y))


def world_to_screen(pos: Tuple, width: int = WIDTH, height: int = HEIGHT):
    """The origin (0,0) is the center of the screen"""
    x, y = pos
    return x + width // 2, height // 2 - y


def draw_fan_sight_range(
    canvas: pygame.Surface,
    width: int,
    height: int,
    x: int,
    y: int,
    pos: Tuple,
    rotation: float,
    sight_angle: float,
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
        max_length * math.cos(start_angle) + pos[0],
        max_length * math.sin(start_angle) + pos[1],
    )
    end_line_end_pos = (
        max_length * math.cos(end_angle) + pos[0],
        max_length * math.sin(end_angle) + pos[1],
    )

    points = [pos, start_line_end_pos, end_line_end_pos]
    pygame.draw.polygon(_canvas, color=COLOR_SIGHT, points=points)
    pygame.draw.line(
        _canvas, color=COLOR_SIGHT_BORDER, start_pos=pos, end_pos=end_line_end_pos, width=2
    )

    # Draw on canvas
    canvas.blit(_canvas, (x, y))


def draw_rectangular_attack_range(
    canvas: pygame.Surface,
    width: int,
    height: int,
    x: int,
    y: int,
    pos: Tuple,
    rotation: float,
    attack_range: float,
    radius: float,
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
        (rx + attack_range * PIX_UNIT_SIZE, -ry),
        (rx + attack_range * PIX_UNIT_SIZE, ry),
        (rx, ry),
    ]

    cos_rot = math.cos(rotation)
    sin_rot = math.sin(rotation)

    points = []
    for _x, _y in rect_points:
        # Rotation
        new_x = _x * cos_rot - _y * sin_rot
        new_y = _x * sin_rot + _y * cos_rot

        _x = new_x + pos[0]
        _y = new_y + pos[1]

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
    bar_width = int(20 * radius / PIX_UNIT_SIZE)
    bar_height = 5

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
    x: int,
    y: int,
    show_attack: bool = True,
    show_stats_bar: bool = True,
    show_sight: bool = False,
    pix_unit: bool = False,
):
    _canvas = pygame.Surface((width, height), pygame.SRCALPHA)

    unit_id = int(np.array(unit.status.unit_id).item())
    pos = np.array(unit.transform.position)
    pos = (pos[0].item() * PIX_UNIT_SIZE, pos[1].item() * PIX_UNIT_SIZE)
    rotation = np.array(unit.transform.rotation).item()  # radian
    radius = np.array(unit.collider.radius).item() * PIX_UNIT_SIZE
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
            _canvas,
            width=width,
            height=height,
            x=0,
            y=0,
            pos=screen_pos,
            rotation=rotation,
            attack_range=attack_range,
            radius=radius,
            attack_angle=sight_angle,
        )

    # Draw units
    if team == 0:
        if is_alive:
            if pix_unit:
                pygame.draw.circle(_canvas, COLOR_PIX_ALLY, screen_pos, radius)
            else:
                portrait = pygame.image.load(f"./assets/units/{ALL_UNIT_NAMES[unit_id]}_ally.png")
        else:
            if pix_unit:
                pygame.draw.circle(_canvas, COLOR_PIX_ALLY_DEAD, screen_pos, radius)
            else:
                portrait = pygame.image.load(
                    f"./assets/units/{ALL_UNIT_NAMES[unit_id]}_ally_dead.png"
                )
    else:
        if is_alive:
            if pix_unit:
                pygame.draw.circle(_canvas, COLOR_PIX_ENEMY, screen_pos, radius)
            else:
                portrait = pygame.image.load(f"./assets/units/{ALL_UNIT_NAMES[unit_id]}_enemy.png")
        else:
            if pix_unit:
                pygame.draw.circle(_canvas, COLOR_PIX_ENEMY_DEAD, screen_pos, radius)
            else:
                portrait = pygame.image.load(
                    f"./assets/units/{ALL_UNIT_NAMES[unit_id]}_enemy_dead.png"
                )

    if not pix_unit:
        portrait = pygame.transform.scale(portrait, (2 * radius, 2 * radius))
        portrait = pygame.transform.rotate(portrait, 270 - rotation_degree)
        _canvas.blit(portrait, (screen_pos[0] - radius, screen_pos[1] - radius))

    # Draw sight
    if show_sight:
        draw_fan_sight_range(
            _canvas,
            width=width,
            height=height,
            x=0,
            y=0,
            pos=screen_pos,
            rotation=rotation,
            sight_angle=sight_angle,
        )

    # Draw health and cooldown bar
    if show_stats_bar:
        draw_stats_bar(
            _canvas,
            pos=screen_pos,
            val=health,
            max_val=max_health,
            fg_color=COLOR_HP,
            radius=radius,
            bar_y_offset=8,
        )
        draw_stats_bar(
            _canvas,
            pos=screen_pos,
            val=cooldown,
            max_val=max_cooldown,
            fg_color=COLOR_COOLDOWN,
            radius=radius,
            bar_y_offset=4,
        )

    # Draw on the canvas
    canvas.blit(_canvas, (x, y))


def get_comb_render(scenario_name: str, state: CombState):
    """Return the step frame of TABSUnitComb formatted as a NumPy array"""
    timestep = str(state.timestep.item())
    remaining_budget = str(state.budget.item())
    unit_list = state.current_unit_list
    enemy_unit_list = state.enemy_unit_comp
    prices = state.all_price

    canvas = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
    canvas.fill(BG_MAIN)

    bg = pygame.image.load("./assets/TABSUnitComb.png")
    canvas.blit(bg, (0, 0))

    # NOTE: width, height, x, y are contants fitting to the TABSUnitComb.png

    # Draw scenario name
    draw_text(canvas, text=scenario_name, width=256, height=30, x=377, y=8, size=28)
    # Draw timestep
    draw_text(canvas, text=timestep, width=38, height=29, x=430, y=49, align_center=False)
    # Draw remaining budget
    draw_text(canvas, text=remaining_budget, width=66, height=29, x=567, y=49, align_center=False)
    # Draw purchased ally unit list
    draw_comb_unit_list(
        canvas, unit_list=unit_list, price=prices, width=256, height=184, x=377, y=91
    )
    # Draw enemy composition
    draw_comb_unit_list(
        canvas, unit_list=enemy_unit_list, price=prices, width=256, height=184, x=377, y=287
    )
    # Draw all unit specification
    draw_catalog(canvas, all_spec=all_spec, width=357, height=463, x=8, y=8)

    array = np.transpose(np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2))  # Rotate
    array = np.flip(array, axis=0)  # Flip because of matplotlib axis

    return array


def get_deploy_render(scenario_name: str, state: DeployState):
    """Return the step frame of TABSUnitDeploy formatted as a NumPy array"""
    timestep = str(state.timestep.item())
    remaining_units = state.remaining_units
    if remaining_units.sum() == 0:
        next_unit = "-"
    else:
        next_unit = ALL_UNIT_NAMES[state.next_unit.item() - 1].title()
    space_occupied = state.space_occupied_spec
    body_radiuses = all_spec["body_radiuses"]

    canvas = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
    canvas.fill(BG_MAIN)

    bg = pygame.image.load("./assets/TABSUnitDeploy.png")
    canvas.blit(bg, (0, 0))

    # NOTE: width, height, x, y are contants fitting to the TABSUnitDeploy.png

    # Draw scenario name
    draw_text(canvas, text=scenario_name, width=433, height=28, x=199, y=10, size=28)
    # Draw timestep
    draw_text(canvas, text=timestep, width=55, height=28, x=140, y=10, align_center=False)
    # Draw next unit
    draw_text(canvas, text=next_unit, width=60, height=45, x=10, y=37, size=16)
    # Draw remaining unit list
    draw_remaining_units(
        canvas, remaining_units=remaining_units, width=486, height=25, x=145, y=51, size=20
    )
    # Draw deployment
    draw_deployment(
        canvas,
        battle_field=state.battle_field,
        body_radiuses=body_radiuses,
        space_occupied=space_occupied,
        is_ally=True,
        width=277,
        height=345,
        x=10,
        y=127,
    )  # ally
    draw_deployment(
        canvas,
        battle_field=state.enemy_battle_field,
        body_radiuses=body_radiuses,
        space_occupied=space_occupied,
        is_ally=False,
        width=277,
        height=345,
        x=356,
        y=127,
    )  # enemy

    array = np.transpose(np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2))  # Rotate
    array = np.flip(array, axis=0)  # Flip because of matplotlib axis

    return array


def get_battle_simulator_render(
    scenario_name: str,
    state: Dict,
    unit_keys: List,
    show_attack: bool = True,
    show_stats_bar: bool = True,
    show_sight: bool = False,
    pix_unit: bool = False,
):
    """Return the step frame of TABSBattleSimulator formatted as a NumPy array"""
    timestep = str(state["game_manager"].timestep.item())

    canvas = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
    canvas.fill(BG_MAIN)

    bg = pygame.image.load("./assets/TABSBattleSimulator.png")
    canvas.blit(bg, (0, 0))

    # NOTE: width, height, x, y are contants fitting to the TABSBattleSimulator.png

    # Draw scenario name
    draw_text(canvas, text=scenario_name, width=190, height=28, x=173, y=6, size=28)
    # Draw timestep
    draw_text(canvas, text=timestep, width=46, height=28, x=421, y=6, align_center=False)
    # Draw units
    for idx, unit_key in enumerate(unit_keys):
        _unit = state[unit_key]
        if _unit.status.is_disabled:
            continue
        draw_unit(
            canvas,
            unit=_unit,
            width=560,
            height=400,
            x=40,
            y=40,
            show_attack=show_attack,
            show_stats_bar=show_stats_bar,
            show_sight=show_sight,
            pix_unit=pix_unit,
        )

    array = np.transpose(np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2))  # Rotate
    array = np.flip(array, axis=0)  # Flip because of matplotlib axis

    return array
