from typing import List
import pygame
import numpy as np

from src.tabs.constants import ALL_UNIT_NAMES
from src.tabs.tabs_unit_comb.tabs_unit_comb import State as CombState
from src.tabs.tabs_unit_deploy.tabs_unit_deploy import State as DeployState
from src.tabs.units import get_all_unit_spec

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)

BG_MAIN = WHITE
TEXT_COLOR = BLACK

WIDTH = 640
HEIGHT = 480

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
    _text = font.render(text, True, TEXT_COLOR, None)
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
    _text = font.render(total_price, True, TEXT_COLOR, None)
    # Align right
    _canvas.blit(_text, (width - text_width - 10, text_height // 2 + 1))

    _w_offset, _h_offset = 4, 54
    _w, _h_name, _h = 86, 21, 31
    for idx, n in enumerate(unit_list):
        _text = font.render(str(n), True, TEXT_COLOR, None)
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
        _text = font.render(price, True, TEXT_COLOR, None)
        _unit.blit(_text, (60 - text_width // 2, 63 - text_height // 2))
        coin = pygame.image.load("./assets/coin_small.png")
        _unit.blit(coin, (32, 65 - text_height // 2))

        # Draw stats
        for jdx, stats in enumerate(stats_list):
            value = str(all_spec[stats][idx])
            text_width, _ = font.size(value)
            _text = stat_font.render(value, True, TEXT_COLOR, None)
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
        _text = font.render(str(n), True, TEXT_COLOR, None)
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


def get_comb_render(scenario_name: str, state: CombState):
    """Return the step frame of TABSUnitComb formatted as a NumPy array"""
    timestep = str(state.timestep.item())
    remaining_budget = str(state.budget.item())
    unit_list = state.current_unit_list
    enemy_unit_list = state.enemy_unit_comp
    prices = state.all_price

    canvas = pygame.Surface((WIDTH, HEIGHT))
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

    canvas = pygame.Surface((WIDTH, HEIGHT))
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
