from typing import List
import pygame
import numpy as np

from src.tabs.constants import ALL_UNIT_NAMES
from src.tabs.tabs_unit_comb.tabs_unit_comb import State as CombState
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
    return canvas


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
    return canvas


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
    return canvas


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
    canvas = draw_text(canvas, text=scenario_name, width=256, height=30, x=377, y=8, size=28)
    # Draw timestep
    canvas = draw_text(canvas, text=timestep, width=38, height=29, x=430, y=49, align_center=False)
    # Draw remaining budget
    canvas = draw_text(
        canvas, text=remaining_budget, width=66, height=29, x=567, y=49, align_center=False
    )
    # Draw purchased ally unit list
    canvas = draw_comb_unit_list(
        canvas, unit_list=unit_list, price=prices, width=256, height=184, x=377, y=91
    )
    # Draw enemy composition
    canvas = draw_comb_unit_list(
        canvas, unit_list=enemy_unit_list, price=prices, width=256, height=184, x=377, y=287
    )
    # Draw all unit specification
    canvas = draw_catalog(canvas, all_spec=all_spec, width=357, height=463, x=8, y=8)

    array = np.transpose(np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2))  # Rotate
    array = np.flip(array, axis=0)  # Flip because of matplotlib axis

    return array
