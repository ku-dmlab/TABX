import json
import math
import os
from datetime import datetime

import jax.numpy as jnp
import numpy as np
import pygame

from src.tabs.constants import ALL_UNIT_NAMES, SIGHT_ANGLE, UNITID2CHAR, UnitID
from src.tabs.units import get_all_unit_spec

# Screen settings
SCREEN_WIDTH = 1400
SCREEN_HEIGHT = 900
GRID_SIZE = 70  # Size of each cell

# Color definitions
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (200, 200, 200)
LIGHT_GRAY = (220, 220, 220)
DARK_GRAY = (100, 100, 100)
BLUE = (100, 150, 255)
RED = (255, 100, 100)
GREEN = (100, 255, 100)
YELLOW = (255, 255, 100)
ORANGE = (255, 165, 0)
PURPLE = (200, 100, 255)
PINK = (255, 150, 180)  # For ALLY button
LIGHT_GREEN = (150, 255, 150)  # For ENEMY button
CORAL = (255, 160, 122)  # For LAVA button
LAVA_COLOR = (255, 50, 50, 80)  # Semi-transparent red (Type 1)
BUSH_COLOR = (50, 255, 50, 80)  # Semi-transparent green (Type 2)
SWAMP_COLOR = (100, 150, 200, 80)  # Semi-transparent blue (Type 3)

# Field parameters (from scenarios.py)
UNIT_SPACING = 4.25  # Fixed spacing between units
SIDE_GAP = 16.0
FIELD_MARGIN_WIDTH = 0.0
FIELD_MARGIN_HEIGHT = 0.0

# Mammoth marker: marks the 3 cells occupied by Mammoth (excluding top-left)
MAMMOTH_OCCUPIED = -999


class MapEditor:
    def __init__(self, max_field_height=78.0, max_field_width=121.0):
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("TABS Scenario Generator")

        # Field size settings
        self.max_field_height = max_field_height
        self.max_field_width = max_field_width
        self.scenario_name = "new_scenario"  # Current scenario name

        # Fixed unit spacing (configurable)
        self.unit_spacing = UNIT_SPACING  # 4.25

        # Calculate grid size based on fixed spacing
        self.grid_height = int(max_field_height / self.unit_spacing)
        self.grid_width = int(max_field_width / self.unit_spacing)

        # Actual unit spacing uses fixed value
        self.actual_unit_spacing_x = self.unit_spacing
        self.actual_unit_spacing_y = self.unit_spacing

        # Grid setup - single grid without ally/enemy distinction
        self.grid = np.zeros(
            (self.grid_height, self.grid_width), dtype=int
        )  # 0 = empty, positive = unit_id
        self.team_grid = np.zeros(
            (self.grid_height, self.grid_width), dtype=int
        )  # 0 = ally, 1 = enemy
        self.rotation_grid = np.zeros(
            (self.grid_height, self.grid_width), dtype=float
        )  # rotation in radians

        # Currently selected unit
        self.selected_unit_id = 0  # 0 = erase, 1-9 = units
        self.selected_team = 0  # 0 = ally, 1 = enemy

        # Edit mode: 'unit' or 'zone'
        self.edit_mode = "unit"

        # UI visibility - panels always visible
        self.unit_panel_visible = True
        self.zone_panel_visible = True

        # Zone scenario settings
        self.zones = []  # List of {type: 1/2/3, position: [x, y], axes: [w, h], effect_value: float}
        self.selected_zone_type = 1  # 1 = lava, 2 = bush, 3 = swamp
        self.zone_effect_value = 10.0  # damage for lava, slow factor for swamp
        self.dragging_zone = False
        self.zone_start_pos = None
        self.current_zone_rect = None
        self.current_zone_center = None  # For ellipse preview
        self.editing_zone_index = None  # For editing existing zone effect_value

        # Load unit specs
        self.all_spec = get_all_unit_spec()
        self.unit_names = ALL_UNIT_NAMES

        # UI area - draggable panels
        self.battlefield_center_x = 400
        self.battlefield_center_y = 450

        # Fixed x position for right panels (20px from right edge of screen)
        right_panel_x = SCREEN_WIDTH - 370 - 20  # 1400 - 370 - 20 = 1010

        # Toolbar (TABS SCENARIO GENERATOR) - top
        self.toolbar_x = right_panel_x
        self.toolbar_y = 20
        self.toolbar_width = 370
        self.toolbar_height = 90
        self.toolbar_dragging = False
        self.toolbar_drag_offset = (0, 0)

        # Unit palette window - below Toolbar
        self.unit_panel_x = right_panel_x
        self.unit_panel_y = self.toolbar_y + self.toolbar_height + 10  # 120
        self.unit_panel_width = 370
        self.unit_panel_height = 460
        self.unit_panel_dragging = False
        self.unit_panel_drag_offset = (0, 0)

        # Zone palette window - starts at same position as Unit
        self.zone_panel_x = right_panel_x
        self.zone_panel_y = self.toolbar_y + self.toolbar_height + 10  # 120
        self.zone_panel_width = 370
<<<<<<< HEAD
        self.zone_panel_height = 240  # Increased to fit description
=======
        self.zone_panel_height = 220
>>>>>>> a4fed485df2ae65fcca9dbd5217824b01ea2cb0e
        self.zone_panel_dragging = False
        self.zone_panel_drag_offset = (0, 0)

        # Camera offset (move with WASD)
        self.camera_offset_x = 0
        self.camera_offset_y = 0
        self.camera_move_speed = 10  # pixels per key press

        # Rotation delay (to prevent too fast rotation)
        self.rotation_cooldown = 0
        self.rotation_cooldown_frames = 5  # Rotate every 5 frames

        # Fonts
        self.font = pygame.font.SysFont("Arial", 24, bold=True)
        self.small_font = pygame.font.SysFont("Arial", 18, bold=True)
        self.tiny_font = pygame.font.SysFont("Arial", 14)
        self.info_font = pygame.font.SysFont("Arial", 16)  # For info display

        self.running = True
        self.clock = pygame.time.Clock()

        # Text input for effect value
        self.text_input_active = False
        self.text_input_value = ""

        # Selected zone for deletion
        self.selected_zone_index = None
        self.hovering_zone_index = None
        self.hovered_zone_info = None  # Store info about zone under mouse

        # Zone editing
        self.editing_zone = False
        self.zone_edit_field = None  # 'pos_x', 'pos_y', 'axis_x', 'axis_y', 'effect'
        self.zone_edit_input = ""

        # Zone selection (double-click to pin info box)
        self.selected_zone_info = None  # Pinned zone info
        self.last_zone_click_time = 0
        self.double_click_threshold = 0.3  # seconds

        # Undo/Redo system
        self.history = []  # List of states for undo
        self.history_index = -1  # Current position in history
        self.max_history = 50  # Maximum number of undo steps
        self.save_state()  # Save initial empty state

        # Load scenario
        self.show_load_dialog = False
<<<<<<< HEAD
        self.available_scenarios = {}  # Dict: folder_name -> list of files
        self.folder_expanded = {}  # Dict: folder_name -> bool (expanded or collapsed)
        self.filtered_scenarios = []  # Search filtered scenarios (flat list)
        self.load_scroll_offset = 0
        self.selected_scenario_index = None
        self.selected_folder = None  # Which folder the selected file is in
=======
        self.available_scenarios = []
        self.filtered_scenarios = []  # Search filtered scenarios
        self.load_scroll_offset = 0
        self.selected_scenario_index = None
>>>>>>> a4fed485df2ae65fcca9dbd5217824b01ea2cb0e
        self.search_query = ""  # Search query
        self.search_active = False  # Search input active

        # New scenario dialog
        self.show_new_scenario_dialog = False
        self.new_scenario_inputs = {
<<<<<<< HEAD
=======
            "scenario_name": "new_scenario",
>>>>>>> a4fed485df2ae65fcca9dbd5217824b01ea2cb0e
            "max_height": str(int(max_field_height)),
            "max_width": str(int(max_field_width)),
            "margin_w": str(int(FIELD_MARGIN_WIDTH)),
            "margin_h": str(int(FIELD_MARGIN_HEIGHT)),
            "unit_spacing": str(self.unit_spacing),
        }
<<<<<<< HEAD
        self.active_input_field = (
            "max_height"  # 'max_height', 'max_width', 'margin_w', 'margin_h', 'unit_spacing'
        )

        # Save scenario dialog
        self.show_save_dialog = False
        self.save_folder_selection = "challenges"  # 'challenges', 'units', 'zones'
        self.save_scenario_name = "new_scenario"  # Scenario name for saving
        self.save_name_input_active = False
=======
        self.active_input_field = "scenario_name"  # 'scenario_name', 'max_height', 'max_width', 'margin_w', 'margin_h', 'unit_spacing'
>>>>>>> a4fed485df2ae65fcca9dbd5217824b01ea2cb0e

        # Calculate field boundaries
        self.calculate_field_boundaries()

    def calculate_field_boundaries(self):
        """Calculate the battlefield boundaries based on field size"""
        # World size is the max field size plus margins
        self.world_width = self.max_field_width + 2 * FIELD_MARGIN_WIDTH
        self.world_height = self.max_field_height + 2 * FIELD_MARGIN_HEIGHT

        # Pixels per world unit (zoom level)
        self.scale = 3.0
        self.min_scale = 1.0
        self.max_scale = 8.0
        self.zoom_speed = 0.1

        # Calculate screen boundaries
        self.update_field_size()

    def update_field_size(self):
        """Update field pixel size based on current scale"""
        self.field_pixel_width = self.world_width * self.scale
        self.field_pixel_height = self.world_height * self.scale

    def save_state(self):
        """Save current state to history for undo/redo"""
        # Create a deep copy of current state
        state = {
            "grid": self.grid.copy(),
            "team_grid": self.team_grid.copy(),
            "rotation_grid": self.rotation_grid.copy(),
            "zones": [zone.copy() for zone in self.zones],
        }

        # Remove any states after current index (when user made new action after undo)
        self.history = self.history[: self.history_index + 1]

        # Add new state
        self.history.append(state)

        # Limit history size
        if len(self.history) > self.max_history:
            self.history.pop(0)
        else:
            self.history_index += 1

    def undo(self):
        """Undo last action"""
        if self.history_index > 0:
            self.history_index -= 1
            self.restore_state(self.history[self.history_index])

    def redo(self):
        """Redo previously undone action"""
        if self.history_index < len(self.history) - 1:
            self.history_index += 1
            self.restore_state(self.history[self.history_index])

    def restore_state(self, state):
        """Restore state from history"""
        self.grid = state["grid"].copy()
        self.team_grid = state["team_grid"].copy()
        self.rotation_grid = state["rotation_grid"].copy()
        self.zones = [zone.copy() for zone in state["zones"]]

    def world_to_screen(self, world_x, world_y):
        """Convert world coordinates to screen coordinates"""
        screen_x = self.battlefield_center_x + world_x * self.scale + self.camera_offset_x
        screen_y = self.battlefield_center_y - world_y * self.scale + self.camera_offset_y
        return int(screen_x), int(screen_y)

    def screen_to_world(self, screen_x, screen_y):
        """Convert screen coordinates to world coordinates"""
        world_x = (screen_x - self.battlefield_center_x - self.camera_offset_x) / self.scale
        world_y = (self.battlefield_center_y + self.camera_offset_y - screen_y) / self.scale
        return world_x, world_y

    def draw_battlefield_background(self):
        """Draw the battlefield with margins and grid"""
        # Draw outer boundary (full field with margins) - apply camera offset
        left = self.battlefield_center_x - self.field_pixel_width / 2 + self.camera_offset_x
        top = self.battlefield_center_y - self.field_pixel_height / 2 + self.camera_offset_y
        pygame.draw.rect(
            self.screen, LIGHT_GRAY, (left, top, self.field_pixel_width, self.field_pixel_height)
        )

        # Draw grid cells
        self.draw_grid_cells()

        # Draw zones
        self.draw_zones()

        # Draw map boundary (thick border) - full map size (including margins)
        # Draw on top of grid
        pygame.draw.rect(
            self.screen,
            BLACK,
            (left, top, self.field_pixel_width, self.field_pixel_height),
            4,  # Thick border (4 pixels)
        )

        # Draw sight range for unit under mouse (if in unit mode)
        if self.edit_mode == "unit":
            self.draw_hovered_unit_sight_range()

        # Draw unit preview at mouse position (if in unit mode and unit selected)
        if self.edit_mode == "unit":
            self.draw_unit_preview_at_mouse()

    def draw_grid_cells(self):
        """Draw the unit placement grid"""
        cell_size_x = int(self.actual_unit_spacing_x * self.scale)
        cell_size_y = int(self.actual_unit_spacing_y * self.scale)

        # First pass: Draw all grid cells
        for i in range(self.grid_height):
            for j in range(self.grid_width):
                # Calculate world position (centered grid)
                world_x = (j - self.grid_width / 2 + 0.5) * self.actual_unit_spacing_x
                world_y = (i - self.grid_height / 2 + 0.5) * self.actual_unit_spacing_y

                screen_x, screen_y = self.world_to_screen(world_x, world_y)

                # Draw cell background
                color = LIGHT_GRAY if (i + j) % 2 == 0 else WHITE
                pygame.draw.rect(
                    self.screen,
                    color,
                    (
                        screen_x - cell_size_x // 2,
                        screen_y - cell_size_y // 2,
                        cell_size_x,
                        cell_size_y,
                    ),
                )
                pygame.draw.rect(
                    self.screen,
                    GRAY,
                    (
                        screen_x - cell_size_x // 2,
                        screen_y - cell_size_y // 2,
                        cell_size_x,
                        cell_size_y,
                    ),
                    1,
                )

        # Second pass: Draw all units on top of grid
        for i in range(self.grid_height):
            for j in range(self.grid_width):
                unit_id = self.grid[i, j]
                if unit_id > 0:
                    # Calculate world position
                    world_x = (j - self.grid_width / 2 + 0.5) * self.actual_unit_spacing_x
                    world_y = (i - self.grid_height / 2 + 0.5) * self.actual_unit_spacing_y
                    screen_x, screen_y = self.world_to_screen(world_x, world_y)

                    # Only draw Mammoth at its top-left cell
                    if unit_id == UnitID.Mammoth:
                        # Skip if not top-left cell
                        if i > 0 and self.grid[i - 1, j] == unit_id:
                            continue
                        if j > 0 and self.grid[i, j - 1] == unit_id:
                            continue

                        # Draw Mammoth at 2x2 center (centered on the 2x2 area)
                        # Top-left at (i,j), so center is at (i+0.5, j+0.5)
                        mammoth_world_x = (
                            j + 0.5 - self.grid_width / 2 + 0.5
                        ) * self.actual_unit_spacing_x
                        mammoth_world_y = (
                            i + 0.5 - self.grid_height / 2 + 0.5
                        ) * self.actual_unit_spacing_y
                        screen_x_mammoth, screen_y_mammoth = self.world_to_screen(
                            mammoth_world_x, mammoth_world_y
                        )

                        team = self.team_grid[i, j]
                        is_ally = team == 0
                        rotation = self.rotation_grid[i, j]
                        draw_size = min(cell_size_x * 2, cell_size_y * 2)

                        self.draw_unit_on_field(
                            unit_id,
                            screen_x_mammoth,
                            screen_y_mammoth,
                            draw_size,
                            is_ally,
                            rotation,
                        )
                    else:
                        # Normal 1x1 unit
                        team = self.team_grid[i, j]
                        is_ally = team == 0
                        rotation = self.rotation_grid[i, j]
                        draw_size = min(cell_size_x, cell_size_y)

                        self.draw_unit_on_field(
                            unit_id,
                            screen_x,
                            screen_y,
                            draw_size,
                            is_ally,
                            rotation,
                        )

    def draw_unit_on_field(self, unit_id, screen_x, screen_y, cell_size, is_ally, rotation=0):
        """Draw a unit on the battlefield with custom rotation"""
        try:
            # Swap images: ally uses enemy image, enemy uses ally image
            suffix = "_enemy" if is_ally else "_ally"
            img_path = (
                f"./src/tabs/visualize/assets/units/{self.unit_names[unit_id - 1]}{suffix}.png"
            )
            unit_img = pygame.image.load(img_path)
            unit_img = pygame.transform.scale(unit_img, (cell_size - 4, cell_size - 4))

            # Rotate base image by 90 degrees
            unit_img = pygame.transform.rotate(unit_img, -90)

            # Apply custom rotation (in degrees, counterclockwise)
            rotation_degrees = (
                -rotation * 180 / np.pi
            )  # Convert radians to degrees, flip for pygame
            unit_img = pygame.transform.rotate(unit_img, rotation_degrees)

            img_rect = unit_img.get_rect(center=(screen_x, screen_y))
            self.screen.blit(unit_img, img_rect)
        except:
            # Fallback to text
            char = UNITID2CHAR.get(unit_id, "?")
            text = self.small_font.render(char, True, BLACK)
            text_rect = text.get_rect(center=(screen_x, screen_y))
            self.screen.blit(text, text_rect)

    def draw_hovered_unit_sight_range(self):
        """Draw sight range and attack range for unit under mouse cursor"""
        mouse_pos = pygame.mouse.get_pos()
        world_x, world_y = self.screen_to_world(mouse_pos[0], mouse_pos[1])

        # Find which grid cell the mouse is over
        for i in range(self.grid_height):
            for j in range(self.grid_width):
                cell_world_x = (j - self.grid_width / 2 + 0.5) * self.actual_unit_spacing_x
                cell_world_y = (i - self.grid_height / 2 + 0.5) * self.actual_unit_spacing_y

                if (
                    abs(world_x - cell_world_x) < self.actual_unit_spacing_x / 2
                    and abs(world_y - cell_world_y) < self.actual_unit_spacing_y / 2
                ):
                    # Found the cell, check if there's a unit
                    unit_id = self.grid[i, j]
                    if unit_id > 0:
                        team = self.team_grid[i, j]
                        is_ally = team == 0
                        rotation = self.rotation_grid[i, j]

                        # Get screen position
                        screen_x, screen_y = self.world_to_screen(cell_world_x, cell_world_y)

                        # For Mammoth, calculate center of 2x2 area
                        if unit_id == UnitID.Mammoth:
                            # Skip if not top-left cell
                            if i > 0 and self.grid[i - 1, j] == unit_id:
                                return
                            if j > 0 and self.grid[i, j - 1] == unit_id:
                                return

                            # Calculate 2x2 center position
                            mammoth_world_x = (
                                j + 0.5 - self.grid_width / 2 + 0.5
                            ) * self.actual_unit_spacing_x
                            mammoth_world_y = (
                                i + 0.5 - self.grid_height / 2 + 0.5
                            ) * self.actual_unit_spacing_y
                            screen_x, screen_y = self.world_to_screen(
                                mammoth_world_x, mammoth_world_y
                            )

                        # Draw sight range
                        self.draw_sight_range((screen_x, screen_y), rotation, is_ally)

                        # Draw attack range
                        self.draw_attack_range((screen_x, screen_y), rotation, unit_id, is_ally)
                    return

    def draw_unit_preview_at_mouse(self):
        """Display semi-transparent preview of selected unit at grid position"""
        if self.selected_unit_id <= 0:
            return

        mouse_x, mouse_y = pygame.mouse.get_pos()
        world_x, world_y = self.screen_to_world(mouse_x, mouse_y)

        # Don't show preview in palette or toolbar area
        if self.edit_mode == "unit":
            panel_x = self.unit_panel_x
            panel_y = self.unit_panel_y
            panel_width = self.unit_panel_width
            panel_height = self.unit_panel_height
            if (
                panel_x <= mouse_x <= panel_x + panel_width
                and panel_y <= mouse_y <= panel_y + panel_height
            ):
                return

        if mouse_y < self.toolbar_y + self.toolbar_height + 50:
            return

        # Check which grid cell the mouse is over
        for i in range(self.grid_height):
            for j in range(self.grid_width):
                cell_world_x = (j - self.grid_width / 2 + 0.5) * self.actual_unit_spacing_x
                cell_world_y = (i - self.grid_height / 2 + 0.5) * self.actual_unit_spacing_y

                if (
                    abs(world_x - cell_world_x) < self.actual_unit_spacing_x / 2
                    and abs(world_y - cell_world_y) < self.actual_unit_spacing_y / 2
                ):
                    # Mouse is over this grid cell
                    # Display as 2x2 if Mammoth
                    is_mammoth = self.selected_unit_id == UnitID.Mammoth

                    # Check if unit can be placed
                    can_place = True
                    if is_mammoth:
                        # Mammoth requires 2x2 area to be empty
                        if i + 1 < self.grid_height and j + 1 < self.grid_width:
                            # Can place if empty or only has MAMMOTH_OCCUPIED markers
                            can_place = True
                            for di in range(2):
                                for dj in range(2):
                                    cell_value = self.grid[i + di, j + dj]
                                    if cell_value != 0 and cell_value != MAMMOTH_OCCUPIED:
                                        can_place = False
                                        break
                                if not can_place:
                                    break
                        else:
                            can_place = False
                    else:
                        # Normal unit requires this cell to be empty
                        # Cannot place on Mammoth
                        has_mammoth = False
                        for mi in range(self.grid_height):
                            for mj in range(self.grid_width):
                                if (
                                    self.grid[mi, mj] == UnitID.Mammoth
                                    and (mi == 0 or self.grid[mi - 1, mj] != UnitID.Mammoth)
                                    and (mj == 0 or self.grid[mi, mj - 1] != UnitID.Mammoth)
                                ):
                                    if mi <= i <= mi + 1 and mj <= j <= mj + 1:
                                        has_mammoth = True
                                        break
                            if has_mammoth:
                                break
                        can_place = self.grid[i, j] == 0 and not has_mammoth

                    if not can_place:
                        return

                    screen_x, screen_y = self.world_to_screen(cell_world_x, cell_world_y)
                    cell_size_x = int(self.actual_unit_spacing_x * self.scale)
                    cell_size_y = int(self.actual_unit_spacing_y * self.scale)
                    draw_size = min(cell_size_x, cell_size_y)

                    # Display as 2x2 if Mammoth
                    if is_mammoth:
                        # Mammoth center position (2x2)
                        mammoth_world_x = (
                            j + 0.5 - self.grid_width / 2 + 0.5
                        ) * self.actual_unit_spacing_x
                        mammoth_world_y = (
                            i + 0.5 - self.grid_height / 2 + 0.5
                        ) * self.actual_unit_spacing_y
                        screen_x, screen_y = self.world_to_screen(mammoth_world_x, mammoth_world_y)
                        draw_size = min(cell_size_x * 2, cell_size_y * 2)

                    # Default rotation
                    rotation = 0 if self.selected_team == 0 else np.pi  # Enemy is pi (180 degrees)

                    # Display preview
                    try:
                        unit_name = self.unit_names[self.selected_unit_id - 1]
                        # Swap images: ally uses enemy image, enemy uses ally image
                        suffix = "_enemy" if self.selected_team == 0 else "_ally"
                        img_path = f"./src/tabs/visualize/assets/units/{unit_name}{suffix}.png"
                        unit_img = pygame.image.load(img_path)
                        unit_img = pygame.transform.scale(unit_img, (draw_size - 4, draw_size - 4))

                        # 90 degree rotation (base)
                        unit_img = pygame.transform.rotate(unit_img, -90)

                        # Apply rotation (convert radians to degrees)
                        rotation_degrees = -rotation * 180 / np.pi
                        unit_img = pygame.transform.rotate(unit_img, rotation_degrees)

                        # Make semi-transparent
                        unit_img.set_alpha(150)

                        # Display at grid position
                        img_rect = unit_img.get_rect(center=(screen_x, screen_y))
                        self.screen.blit(unit_img, img_rect)

                        # Selection indicator (border)
                        pygame.draw.rect(
                            self.screen,
                            YELLOW,
                            (
                                screen_x - draw_size // 2,
                                screen_y - draw_size // 2,
                                draw_size,
                                draw_size,
                            ),
                            2,
                        )
                    except:
                        # Fallback to simple circle if image load fails
                        pygame.draw.circle(
                            self.screen, YELLOW, (screen_x, screen_y), draw_size // 2, 2
                        )
                    return

    def draw_sight_range(self, screen_pos, rotation, is_ally):
        """Draw unit sight range perfectly aligned with pygame-rotated unit"""

        # pygame rotates image by (-rotation)
        screen_angle = -rotation

        sight_angle = SIGHT_ANGLE

        if is_ally:
            sight_color = (0, 255, 255, 40)
            border_color = (0, 255, 255)
        else:
            sight_color = (255, 0, 0, 40)
            border_color = (255, 0, 0)

        max_dist = math.hypot(SCREEN_WIDTH, SCREEN_HEIGHT)

        start = screen_angle - sight_angle / 2
        end = screen_angle + sight_angle / 2

        points = [screen_pos]

        segments = max(12, int(sight_angle * 180 / math.pi / 4))

        for i in range(segments + 1):
            t = i / segments
            a = start + (end - start) * t

            x = screen_pos[0] + max_dist * math.cos(a)
            y = screen_pos[1] - max_dist * math.sin(a)

            points.append((x, y))

        temp = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
        pygame.draw.polygon(temp, sight_color, points)
        self.screen.blit(temp, (0, 0))

        pygame.draw.line(self.screen, border_color, points[0], points[1], 2)
        pygame.draw.line(self.screen, border_color, points[0], points[-1], 2)

    def draw_attack_range(self, screen_pos, rotation, unit_id, is_ally):
        """Draw unit attack range as a rectangle"""
        if unit_id <= 0 or unit_id > len(self.unit_names):
            return

        # Get attack range from unit specs
        attack_range = float(self.all_spec["attack_ranges"][unit_id - 1])

        if attack_range <= 0:
            return

        # pygame rotates image by (-rotation)
        screen_angle = -rotation

        # Attack range color (ally: orange, enemy: red)
        if is_ally:
            attack_color = (255, 150, 0, 40)  # Semi-transparent orange
            border_color = (255, 150, 0)
        else:
            attack_color = (255, 80, 0, 40)  # Semi-transparent orange (darker)
            border_color = (255, 80, 0)

        # Convert world distance to screen distance
        attack_screen_range = attack_range * self.scale

        # Attack range angle (π/4 = 45 degrees)
        attack_angle = math.pi / 4
        cos_half = math.cos(attack_angle / 2)
        sin_half = math.sin(attack_angle / 2)

        # Rectangle points for attack range
        # Based on render.py's rectangular attack range
        width = attack_screen_range
        height = 2 * sin_half * self.actual_unit_spacing_x * self.scale / 2

        rect_points = [
            (
                cos_half * self.actual_unit_spacing_x * self.scale / 2,
                -sin_half * self.actual_unit_spacing_x * self.scale / 2,
            ),
            (
                cos_half * self.actual_unit_spacing_x * self.scale / 2 + width,
                -sin_half * self.actual_unit_spacing_x * self.scale / 2,
            ),
            (
                cos_half * self.actual_unit_spacing_x * self.scale / 2 + width,
                sin_half * self.actual_unit_spacing_x * self.scale / 2,
            ),
            (
                cos_half * self.actual_unit_spacing_x * self.scale / 2,
                sin_half * self.actual_unit_spacing_x * self.scale / 2,
            ),
        ]

        # Apply rotation
        cos_rot = math.cos(screen_angle)
        sin_rot = math.sin(screen_angle)

        rotated_points = []
        for x, y in rect_points:
            new_x = x * cos_rot - y * sin_rot
            new_y = x * sin_rot + y * cos_rot
            screen_x = screen_pos[0] + new_x
            screen_y = screen_pos[1] - new_y
            rotated_points.append((screen_x, screen_y))

        try:
            # Draw rectangle
            temp = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
            pygame.draw.polygon(temp, attack_color, rotated_points)
            self.screen.blit(temp, (0, 0))

            # Draw border
            pygame.draw.line(self.screen, border_color, rotated_points[0], rotated_points[1], 2)
            pygame.draw.line(self.screen, border_color, rotated_points[1], rotated_points[2], 2)
            pygame.draw.line(self.screen, border_color, rotated_points[2], rotated_points[3], 2)
            pygame.draw.line(self.screen, border_color, rotated_points[3], rotated_points[0], 2)
        except:
            pass

    def draw_zones(self):
        """Draw zone scenarios (ellipses for lava/bush/swamp)"""
        for idx, zone in enumerate(self.zones):
            zone_type = zone["type"]
            pos = zone["position"]  # [x, y] in world coordinates
            axes = zone["axes"]  # [semi-width, semi-height] in world coordinates
            effect_value = zone["effect_value"]

            # Convert to screen coordinates
            center_x, center_y = self.world_to_screen(pos[0], pos[1])
            # axes is radius, so multiply by 2 for full width/height
            width = int(axes[0] * 2 * self.scale)
            height = int(axes[1] * 2 * self.scale)

            if width <= 0 or height <= 0:
                continue

            # Create semi-transparent surface for ellipse
            ellipse_surface = pygame.Surface((width, height), pygame.SRCALPHA)
            ellipse_surface.fill((0, 0, 0, 0))

            if zone_type == 1:  # Lava
                color = LAVA_COLOR
                border_color = (255, 50, 50, 160)
                label = "Lava"
            elif zone_type == 2:  # Bush
                color = BUSH_COLOR
                border_color = (50, 255, 50, 160)
                label = "Bush"
            else:  # Swamp (Type 3)
                color = SWAMP_COLOR
                border_color = (100, 150, 200, 160)
                label = "Swamp"

            # Highlight if hovering or selected
            if idx == self.hovering_zone_index or idx == self.selected_zone_index:
                border_color = (255, 255, 0, 255)  # Yellow highlight
                border_width = 3
            else:
                border_width = 2

            # Draw ellipse
            pygame.draw.ellipse(ellipse_surface, color, (0, 0, width, height))
            pygame.draw.ellipse(ellipse_surface, border_color, (0, 0, width, height), border_width)

            # Blit to screen
            self.screen.blit(ellipse_surface, (center_x - width // 2, center_y - height // 2))

            # Draw effect value and label
            if self.scale > 2.0:
                if zone_type == 1:  # Lava - show damage
                    value_text = f"Dmg: {effect_value:.1f}"
                elif zone_type == 3:  # Swamp - show slow factor
                    value_text = f"Slow: {effect_value:.1%}"
                else:  # Bush - no effect value
                    value_text = label

                text = self.tiny_font.render(value_text, True, (255, 255, 255))
                text_rect = text.get_rect(center=(center_x, center_y))
                # Draw text background
                bg_rect = text_rect.inflate(4, 2)
                pygame.draw.rect(self.screen, (0, 0, 0, 128), bg_rect)
                self.screen.blit(text, text_rect)

        # Draw current zone being dragged (preview)
        if self.dragging_zone and self.zone_start_pos and self.current_zone_rect:
            width = self.current_zone_rect[2]
            height = self.current_zone_rect[3]

            if width > 0 and height > 0:
                ellipse_surface = pygame.Surface((width, height), pygame.SRCALPHA)
                ellipse_surface.fill((0, 0, 0, 0))

                if self.selected_zone_type == 1:  # Lava
                    color = LAVA_COLOR
                    border_color = (255, 50, 50, 160)
                elif self.selected_zone_type == 2:  # Bush
                    color = BUSH_COLOR
                    border_color = (50, 255, 50, 160)
                else:  # Swamp
                    color = SWAMP_COLOR
                    border_color = (100, 150, 200, 160)

                pygame.draw.ellipse(ellipse_surface, color, (0, 0, width, height))
                pygame.draw.ellipse(ellipse_surface, border_color, (0, 0, width, height), 2)

                self.screen.blit(
                    ellipse_surface, (self.current_zone_rect[0], self.current_zone_rect[1])
                )

    def handle_zone_info_click(self, pos):
        """Handle clicks on zone info box for editing (only when pinned)"""
        # Only handle clicks if zone is pinned (selected)
        if not self.selected_zone_info:
            return False

        zone_info = self.selected_zone_info

        x, y = pos
        box_width = 260
        box_height = 220
        box_x = SCREEN_WIDTH - box_width - 20
        box_y = SCREEN_HEIGHT - box_height - 20

        # Check if click is inside info box
        if not (box_x <= x <= box_x + box_width and box_y <= y <= box_y + box_height):
            # Click outside - unpin
            self.selected_zone_info = None
            self.editing_zone = False
            self.zone_edit_field = None
            self.zone_edit_input = ""
            return False

        zone_idx = zone_info["index"] - 1
        if zone_idx < 0 or zone_idx >= len(self.zones):
            return True

        zone = self.zones[zone_idx]
        zone_type_id = zone_info["zone_type_id"]

        # Check field clicks
        title_bar_height = 30
        field_y = box_y + title_bar_height + 10 + 25  # title_bar + 10 + hint(25)

        for field_name, label, value in [
            ("pos_x", "Position X:", zone["position"][0]),
            ("pos_y", "Position Y:", zone["position"][1]),
            ("axis_x", "Axis X:", zone["axes"][0]),
            ("axis_y", "Axis Y:", zone["axes"][1]),
        ] + ([("effect", "Effect:", zone["effect_value"])] if zone_type_id in [1, 3] else []):
            input_x = box_x + 110
            input_rect = pygame.Rect(input_x, field_y - 2, 130, 26)

            if input_rect.collidepoint(x, y):
                self.editing_zone = True
                self.zone_edit_field = field_name
                # Convert existing value to string for input field
                if field_name == "effect" and zone_type_id == 3:
                    self.zone_edit_input = f"{value:.2f}"
                else:
                    self.zone_edit_input = f"{value:.1f}"
                return True

            field_y += 30

        return True  # Consume click if inside box

    def apply_zone_edits(self):
        """Apply current zone edits"""
        zone_info = self.selected_zone_info if self.selected_zone_info else self.hovered_zone_info

        if not zone_info:
            return

        zone_idx = zone_info["index"] - 1
        if zone_idx < 0 or zone_idx >= len(self.zones):
            return

        zone = self.zones[zone_idx]

        # Update the zone info
        updated_info = {
            "index": zone_idx + 1,
            "type": {1: "Lava", 2: "Bush", 3: "Swamp"}[zone["type"]],
            "position": zone["position"],
            "axes": zone["axes"],
            "effect_value": zone["effect_value"],
            "zone_type_id": zone["type"],
        }

        # Update both hovered and selected info if they exist
        if self.hovered_zone_info and self.hovered_zone_info["index"] == zone_idx + 1:
            self.hovered_zone_info = updated_info
        if self.selected_zone_info and self.selected_zone_info["index"] == zone_idx + 1:
            self.selected_zone_info = updated_info

    def update_hovered_zone(self, mouse_pos):
        """Update which zone is currently being hovered over"""
        world_x, world_y = self.screen_to_world(mouse_pos[0], mouse_pos[1])

        # Check which zone the mouse is over (in reverse order to prioritize top zones)
        self.hovered_zone_info = None
        for idx in range(len(self.zones) - 1, -1, -1):
            zone = self.zones[idx]
            pos_zone = zone["position"]
            axes = zone["axes"]

            # Check if point is inside ellipse
            dx = (world_x - pos_zone[0]) / axes[0]
            dy = (world_y - pos_zone[1]) / axes[1]

            if dx * dx + dy * dy <= 1:
                # Store the zone info
                zone_type_name = {1: "Lava", 2: "Bush", 3: "Swamp"}[zone["type"]]
                self.hovered_zone_info = {
                    "index": idx + 1,
                    "type": zone_type_name,
                    "position": pos_zone,
                    "axes": axes,
                    "effect_value": zone["effect_value"],
                    "zone_type_id": zone["type"],
                }
                self.hovering_zone_index = idx
                return

        self.hovering_zone_index = None

    def draw_hovered_zone_info(self):
        """Draw info box for hovered/selected zone in bottom right corner"""
        # Check if we should show selected (pinned) or hovered zone
        is_pinned = self.selected_zone_info is not None
        zone_info = self.selected_zone_info if is_pinned else self.hovered_zone_info

        if not zone_info:
            return

        zone_idx = zone_info["index"] - 1
        if zone_idx < 0 or zone_idx >= len(self.zones):
            return

        zone = self.zones[zone_idx]
        zone_type_id = zone_info["zone_type_id"]

        # Bottom right corner - compact
        if is_pinned:
            box_width = 260
            box_height = 220
        else:
            box_width = 260
            box_height = 160

        box_x = SCREEN_WIDTH - box_width - 20
        box_y = SCREEN_HEIGHT - box_height - 20

        # Draw background box - consistent style (same as palette)
        info_surface = pygame.Surface((box_width, box_height), pygame.SRCALPHA)
        info_surface.fill((80, 80, 80, 240))  # Dark gray
        self.screen.blit(info_surface, (box_x, box_y))

        # Black border
        pygame.draw.rect(self.screen, BLACK, (box_x, box_y, box_width, box_height), 2)

        # Title bar
        title_bar_height = 30
        pygame.draw.rect(self.screen, (50, 50, 50), (box_x, box_y, box_width, title_bar_height))
        pygame.draw.rect(self.screen, BLACK, (box_x, box_y, box_width, title_bar_height), 2)

        # Draw title (centered)
        zone_title = self.small_font.render(
            f"{zone_info['type']} Zone #{zone_info['index']}", True, WHITE
        )
        title_x = box_x + (box_width - zone_title.get_width()) // 2
        self.screen.blit(zone_title, (title_x, box_y + 7))

        field_y = box_y + title_bar_height + 10

        if is_pinned:
            # EDITABLE VERSION - Show input boxes
            hint_text = self.tiny_font.render(
                "Click field to edit | Click outside to close", True, LIGHT_GRAY
            )
            hint_x = box_x + (box_width - hint_text.get_width()) // 2
            self.screen.blit(hint_text, (hint_x, field_y))
            field_y += 25

            fields = [
                ("pos_x", "Position X:", zone["position"][0]),
                ("pos_y", "Position Y:", zone["position"][1]),
                ("axis_x", "Axis X:", zone["axes"][0]),
                ("axis_y", "Axis Y:", zone["axes"][1]),
            ]

            # Add effect value field if applicable
            if zone_type_id in [1, 3]:
                fields.append(("effect", "Effect:", zone["effect_value"]))

            for field_name, label, value in fields:
                # Label
                label_text = self.small_font.render(label, True, WHITE)
                self.screen.blit(label_text, (box_x + 10, field_y))

                # Input box
                input_x = box_x + 110
                input_width = 130
                input_height = 26
                input_rect = pygame.Rect(input_x, field_y - 2, input_width, input_height)

                is_active = self.editing_zone and self.zone_edit_field == field_name
                box_color = YELLOW if is_active else LIGHT_GRAY
                pygame.draw.rect(self.screen, box_color, input_rect)
                pygame.draw.rect(self.screen, BLACK, input_rect, 2)

                # Display value (centered)
                if is_active:
                    display_text = self.zone_edit_input + "|"
                else:
                    if field_name == "effect" and zone_type_id == 3:
                        display_text = f"{value:.2f}"
                    else:
                        display_text = f"{value:.1f}"

                value_text = self.small_font.render(display_text, True, BLACK)
                value_x = input_x + (input_width - value_text.get_width()) // 2
                value_y = field_y - 2 + (input_height - value_text.get_height()) // 2
                self.screen.blit(value_text, (value_x, value_y))

                field_y += 30

        else:
            # READ-ONLY VERSION - Show info
            hint_text = self.tiny_font.render("Double-click zone to edit", True, LIGHT_GRAY)
            hint_x = box_x + (box_width - hint_text.get_width()) // 2
            self.screen.blit(hint_text, (hint_x, field_y))
            field_y += 25

            details = [
                f"Position: ({zone['position'][0]:.1f}, {zone['position'][1]:.1f})",
                f"Axes: ({zone['axes'][0]:.1f}, {zone['axes'][1]:.1f})",
            ]

            if zone_type_id == 1:
                details.append(f"Effect: {zone['effect_value']:.1f} damage")
            elif zone_type_id == 3:
                details.append(f"Effect: {zone['effect_value']:.0%} slow")
            else:
                details.append(f"Effect: None (Bush)")

            for detail in details:
                detail_text = self.small_font.render(detail, True, WHITE)
                self.screen.blit(detail_text, (box_x + 15, field_y))
                field_y += 28

    def draw_unit_palette(self):
        """Draw unit selection palette - draggable window"""
        # Only show in unit mode
        if self.edit_mode != "unit":
            return

        panel_x = self.unit_panel_x
        panel_y = self.unit_panel_y
        panel_width = self.unit_panel_width
        panel_height = self.unit_panel_height

        # Panel background - dark gray
        panel_surface = pygame.Surface((panel_width, panel_height), pygame.SRCALPHA)
        panel_surface.fill((80, 80, 80, 240))  # Dark gray
        self.screen.blit(panel_surface, (panel_x, panel_y))
        pygame.draw.rect(self.screen, BLACK, (panel_x, panel_y, panel_width, panel_height), 2)

        # Title bar for dragging (darker background)
        title_bar_height = 30
        pygame.draw.rect(
            self.screen, (50, 50, 50), (panel_x, panel_y, panel_width, title_bar_height)
        )
        pygame.draw.rect(self.screen, BLACK, (panel_x, panel_y, panel_width, title_bar_height), 2)

        # Title
        title = self.font.render("Unit Palette", True, WHITE)
        self.screen.blit(title, (panel_x + 10, panel_y + 5))

        # Team selection buttons
        team_y = panel_y + title_bar_height + 10
        ally_color = PINK if self.selected_team == 0 else LIGHT_GRAY
        enemy_color = LIGHT_GREEN if self.selected_team == 1 else LIGHT_GRAY

        pygame.draw.rect(self.screen, ally_color, (panel_x + 10, team_y, 170, 40))
        pygame.draw.rect(self.screen, BLACK, (panel_x + 10, team_y, 170, 40), 2)
        ally_text = self.font.render("ALLY", True, BLACK)
        ally_x = panel_x + 10 + (170 - ally_text.get_width()) // 2
        ally_y = team_y + (40 - ally_text.get_height()) // 2
        self.screen.blit(ally_text, (ally_x, ally_y))

        pygame.draw.rect(self.screen, enemy_color, (panel_x + 190, team_y, 170, 40))
        pygame.draw.rect(self.screen, BLACK, (panel_x + 190, team_y, 170, 40), 2)
        enemy_text = self.font.render("ENEMY", True, BLACK)
        enemy_x = panel_x + 190 + (170 - enemy_text.get_width()) // 2
        enemy_y = team_y + (40 - enemy_text.get_height()) // 2
        self.screen.blit(enemy_text, (enemy_x, enemy_y))

        # Unit buttons (3x3 grid)
        unit_button_size = 110
        for idx in range(len(self.unit_names)):
            unit_id = idx + 1
            row = idx // 3
            col = idx % 3

            x = panel_x + 10 + col * (unit_button_size + 10)
            y = panel_y + title_bar_height + 60 + row * (unit_button_size + 15)

            # Button background (highlight selected unit)
            button_color = GREEN if self.selected_unit_id == unit_id else LIGHT_GRAY
            pygame.draw.rect(self.screen, button_color, (x, y, unit_button_size, unit_button_size))
            pygame.draw.rect(self.screen, BLACK, (x, y, unit_button_size, unit_button_size), 2)

            # Unit image
            try:
                img_path = f"./src/tabs/visualize/assets/units/{self.unit_names[idx]}.png"
                unit_img = pygame.image.load(img_path)
                unit_img = pygame.transform.scale(unit_img, (60, 60))
                self.screen.blit(unit_img, (x + 25, y + 10))
            except:
                pass

            # Unit name (centered)
            name_text = self.small_font.render(self.unit_names[idx].title()[:8], True, BLACK)
            text_width = name_text.get_width()
            text_x = x + (110 - text_width) // 2  # Center of button width 110
            self.screen.blit(name_text, (text_x, y + 75))

    def draw_zone_palette(self):
        """Draw zone editing palette - draggable window"""
        # Only show in zone mode
        if self.edit_mode != "zone":
            return

        panel_x = self.zone_panel_x
        panel_y = self.zone_panel_y
        panel_width = self.zone_panel_width
        panel_height = self.zone_panel_height

        # Panel background - dark gray
        panel_surface = pygame.Surface((panel_width, panel_height), pygame.SRCALPHA)
        panel_surface.fill((80, 80, 80, 240))  # Dark gray
        self.screen.blit(panel_surface, (panel_x, panel_y))
        pygame.draw.rect(self.screen, BLACK, (panel_x, panel_y, panel_width, panel_height), 2)

        # Title bar for dragging - darker background
        title_bar_height = 30
        pygame.draw.rect(
            self.screen, (50, 50, 50), (panel_x, panel_y, panel_width, title_bar_height)
        )
        pygame.draw.rect(self.screen, BLACK, (panel_x, panel_y, panel_width, title_bar_height), 2)

        # Title
        title = self.font.render("Zone Palette", True, WHITE)
        self.screen.blit(title, (panel_x + 10, panel_y + 5))

        # Instructions
        y_offset = panel_y + title_bar_height + 5
        inst1 = self.small_font.render("Drag to create zone | Double-click to edit", True, WHITE)
        inst1_x = panel_x + (panel_width - inst1.get_width()) // 2
        self.screen.blit(inst1, (inst1_x, y_offset))

        # Total Zones counter
        y_offset += 20
        zone_count_text = self.small_font.render(f"Total Zones: {len(self.zones)}", True, WHITE)
        zone_count_x = panel_x + (panel_width - zone_count_text.get_width()) // 2
        self.screen.blit(zone_count_text, (zone_count_x, y_offset))

        # Zone type buttons
        y_offset += 30
        btn_width = 110
        btn_spacing = 5

        # Lava zone button (Type 1)
        lava_btn_color = CORAL if self.selected_zone_type == 1 else LIGHT_GRAY
        lava_x_pos = panel_x + 10
        pygame.draw.rect(self.screen, lava_btn_color, (lava_x_pos, y_offset, btn_width, 40))
        pygame.draw.rect(self.screen, BLACK, (lava_x_pos, y_offset, btn_width, 40), 2)
        lava_text = self.font.render("LAVA", True, BLACK)
        lava_text_x = lava_x_pos + (btn_width - lava_text.get_width()) // 2
        lava_text_y = y_offset + (40 - lava_text.get_height()) // 2
        self.screen.blit(lava_text, (lava_text_x, lava_text_y))

        # Bush zone button (Type 2)
        bush_btn_color = LIGHT_GREEN if self.selected_zone_type == 2 else LIGHT_GRAY
        bush_x_pos = panel_x + 10 + btn_width + btn_spacing
        pygame.draw.rect(self.screen, bush_btn_color, (bush_x_pos, y_offset, btn_width, 40))
        pygame.draw.rect(self.screen, BLACK, (bush_x_pos, y_offset, btn_width, 40), 2)
        bush_text = self.font.render("BUSH", True, BLACK)
        bush_text_x = bush_x_pos + (btn_width - bush_text.get_width()) // 2
        bush_text_y = y_offset + (40 - bush_text.get_height()) // 2
        self.screen.blit(bush_text, (bush_text_x, bush_text_y))

        # Swamp zone button (Type 3)
        swamp_btn_color = (150, 200, 255) if self.selected_zone_type == 3 else LIGHT_GRAY
        swamp_x_pos = panel_x + 10 + (btn_width + btn_spacing) * 2
        pygame.draw.rect(self.screen, swamp_btn_color, (swamp_x_pos, y_offset, btn_width, 40))
        pygame.draw.rect(self.screen, BLACK, (swamp_x_pos, y_offset, btn_width, 40), 2)
        swamp_text = self.font.render("SWAMP", True, BLACK)
        swamp_text_x = swamp_x_pos + (btn_width - swamp_text.get_width()) // 2
        swamp_text_y = y_offset + (40 - swamp_text.get_height()) // 2
        self.screen.blit(swamp_text, (swamp_text_x, swamp_text_y))

<<<<<<< HEAD
        # Zone description based on selected type
        y_offset += 48
        zone_descriptions = {
            1: "Deals damage over time to units inside",
            2: "Provides cover",
            3: "Slows down unit movement speed",
        }
        desc_text = self.tiny_font.render(
            zone_descriptions[self.selected_zone_type], True, LIGHT_GRAY
        )
        desc_x = panel_x + (panel_width - desc_text.get_width()) // 2
        self.screen.blit(desc_text, (desc_x, y_offset))

        # Effect value input section
        y_offset += 25
=======
        # Effect value input section
        y_offset += 50
>>>>>>> a4fed485df2ae65fcca9dbd5217824b01ea2cb0e
        effect_label = self.small_font.render("Set effect value", True, WHITE)
        effect_label_x = panel_x + (panel_width - effect_label.get_width()) // 2
        self.screen.blit(effect_label, (effect_label_x, y_offset))

        # Input box and Apply button
        y_offset += 25
        input_width = 240
        apply_width = 100
        input_box_rect = pygame.Rect(panel_x + 10, y_offset, input_width, 35)
        input_box_color = YELLOW if self.text_input_active else WHITE
        pygame.draw.rect(self.screen, input_box_color, input_box_rect)
        pygame.draw.rect(self.screen, BLACK, input_box_rect, 2)

        # Display current value or input text
        if self.text_input_active:
            display_text = self.text_input_value + "|"
        else:
            if self.selected_zone_type == 1:
                display_text = f"{self.zone_effect_value:.1f}"
            elif self.selected_zone_type == 3:
                display_text = f"{self.zone_effect_value:.2f}"
            else:
                display_text = "-"

        input_text = self.font.render(display_text, True, BLACK)
        input_text_x = input_box_rect.x + (input_width - input_text.get_width()) // 2
        input_text_y = input_box_rect.y + (35 - input_text.get_height()) // 2
        self.screen.blit(input_text, (input_text_x, input_text_y))

        # Apply button
        apply_btn_rect = pygame.Rect(panel_x + 10 + input_width + 10, y_offset, apply_width, 35)
        pygame.draw.rect(self.screen, YELLOW, apply_btn_rect)
        pygame.draw.rect(self.screen, BLACK, apply_btn_rect, 2)
        apply_text = self.font.render("APPLY", True, BLACK)
        apply_x = apply_btn_rect.x + (apply_width - apply_text.get_width()) // 2
        apply_y = apply_btn_rect.y + (35 - apply_text.get_height()) // 2
        self.screen.blit(apply_text, (apply_x, apply_y))

    def draw_new_scenario_dialog(self):
        """Draw the new scenario creation dialog"""
        # Semi-transparent overlay
        overlay = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 180))
        self.screen.blit(overlay, (0, 0))

        # Dialog box - consistent with palette style
        dialog_width = 400
<<<<<<< HEAD
        dialog_height = 330
=======
        dialog_height = 420
>>>>>>> a4fed485df2ae65fcca9dbd5217824b01ea2cb0e
        dialog_x = (SCREEN_WIDTH - dialog_width) // 2
        dialog_y = (SCREEN_HEIGHT - dialog_height) // 2

        # Background
        dialog_surface = pygame.Surface((dialog_width, dialog_height), pygame.SRCALPHA)
        dialog_surface.fill((80, 80, 80, 240))  # Dark gray
        self.screen.blit(dialog_surface, (dialog_x, dialog_y))
        pygame.draw.rect(self.screen, BLACK, (dialog_x, dialog_y, dialog_width, dialog_height), 2)

        # Title bar
        title_bar_height = 30
        pygame.draw.rect(
            self.screen, (50, 50, 50), (dialog_x, dialog_y, dialog_width, title_bar_height)
        )
        pygame.draw.rect(
            self.screen, BLACK, (dialog_x, dialog_y, dialog_width, title_bar_height), 2
        )

        # Title (centered)
        title = self.small_font.render("New Scenario", True, WHITE)
        title_x = dialog_x + (dialog_width - title.get_width()) // 2
        self.screen.blit(title, (title_x, dialog_y + 7))

<<<<<<< HEAD
        # Input fields (removed scenario_name)
        y_offset = dialog_y + title_bar_height + 15
        fields = [
            ("max_height", "Max Field Height:", self.new_scenario_inputs["max_height"]),
            ("max_width", "Max Field Width:", self.new_scenario_inputs["max_width"]),
            ("margin_h", "Margin Height:", self.new_scenario_inputs["margin_h"]),
            ("margin_w", "Margin Width:", self.new_scenario_inputs["margin_w"]),
=======
        # Input fields
        y_offset = dialog_y + title_bar_height + 15
        fields = [
            ("scenario_name", "Scenario Name:", self.new_scenario_inputs["scenario_name"]),
            ("max_height", "Max Field Height:", self.new_scenario_inputs["max_height"]),
            ("max_width", "Max Field Width:", self.new_scenario_inputs["max_width"]),
            ("margin_w", "Margin Width:", self.new_scenario_inputs["margin_w"]),
            ("margin_h", "Margin Height:", self.new_scenario_inputs["margin_h"]),
>>>>>>> a4fed485df2ae65fcca9dbd5217824b01ea2cb0e
            ("unit_spacing", "Unit Spacing:", self.new_scenario_inputs["unit_spacing"]),
        ]

        for field_name, label, value in fields:
            # Label
            label_text = self.small_font.render(label, True, WHITE)
            self.screen.blit(label_text, (dialog_x + 15, y_offset))

            # Input box
            input_box = pygame.Rect(dialog_x + 180, y_offset - 3, 200, 28)
            is_active = self.active_input_field == field_name
            box_color = YELLOW if is_active else LIGHT_GRAY
            pygame.draw.rect(self.screen, box_color, input_box)
            pygame.draw.rect(self.screen, BLACK, input_box, 2)

            # Display value (centered)
            display_value = value + ("|" if is_active else "")
            value_text = self.small_font.render(display_value, True, BLACK)
            value_x = input_box.x + (200 - value_text.get_width()) // 2
            value_y = input_box.y + (28 - value_text.get_height()) // 2
            self.screen.blit(value_text, (value_x, value_y))

            y_offset += 50

        # Buttons
<<<<<<< HEAD
        button_y = dialog_y + dialog_height - 40
=======
        button_y = dialog_y + dialog_height - 50
>>>>>>> a4fed485df2ae65fcca9dbd5217824b01ea2cb0e
        button_width = 150
        button_height = 35
        create_btn = pygame.Rect(dialog_x + 30, button_y, button_width, button_height)
        cancel_btn = pygame.Rect(dialog_x + 220, button_y, button_width, button_height)

        pygame.draw.rect(self.screen, GREEN, create_btn)
        pygame.draw.rect(self.screen, BLACK, create_btn, 2)
        pygame.draw.rect(self.screen, RED, cancel_btn)
        pygame.draw.rect(self.screen, BLACK, cancel_btn, 2)

        create_text = self.small_font.render("Create", True, BLACK)
        cancel_text = self.small_font.render("Cancel", True, BLACK)
        create_x = create_btn.x + (button_width - create_text.get_width()) // 2
        create_y = create_btn.y + (button_height - create_text.get_height()) // 2
        cancel_x = cancel_btn.x + (button_width - cancel_text.get_width()) // 2
        cancel_y = cancel_btn.y + (button_height - cancel_text.get_height()) // 2
        self.screen.blit(create_text, (create_x, create_y))
        self.screen.blit(cancel_text, (cancel_x, cancel_y))

<<<<<<< HEAD
    def draw_save_dialog(self):
        """Draw the save scenario dialog with folder selection"""
        # Semi-transparent overlay
        overlay = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 180))
        self.screen.blit(overlay, (0, 0))

        # Dialog box - more compact
        dialog_width = 380
        dialog_height = 205
        dialog_x = (SCREEN_WIDTH - dialog_width) // 2
        dialog_y = (SCREEN_HEIGHT - dialog_height) // 2

        # Background
        dialog_surface = pygame.Surface((dialog_width, dialog_height), pygame.SRCALPHA)
        dialog_surface.fill((80, 80, 80, 240))
        self.screen.blit(dialog_surface, (dialog_x, dialog_y))
        pygame.draw.rect(self.screen, BLACK, (dialog_x, dialog_y, dialog_width, dialog_height), 2)

        # Title bar
        title_bar_height = 30
        pygame.draw.rect(
            self.screen, (50, 50, 50), (dialog_x, dialog_y, dialog_width, title_bar_height)
        )
        pygame.draw.rect(
            self.screen, BLACK, (dialog_x, dialog_y, dialog_width, title_bar_height), 2
        )

        # Title (centered)
        title = self.small_font.render("Save Scenario", True, WHITE)
        title_x = dialog_x + (dialog_width - title.get_width()) // 2
        self.screen.blit(title, (title_x, dialog_y + 7))

        # Scenario name input
        name_y = dialog_y + title_bar_height + 15
        name_label = self.small_font.render("Name:", True, WHITE)
        self.screen.blit(name_label, (dialog_x + 20, name_y + 3))

        name_input_box = pygame.Rect(dialog_x + 75, name_y, 285, 28)
        name_color = YELLOW if self.save_name_input_active else LIGHT_GRAY
        pygame.draw.rect(self.screen, name_color, name_input_box)
        pygame.draw.rect(self.screen, BLACK, name_input_box, 2)

        # Display scenario name
        display_name = self.save_scenario_name + ("|" if self.save_name_input_active else "")
        name_text = self.small_font.render(display_name, True, BLACK)
        self.screen.blit(name_text, (name_input_box.x + 8, name_input_box.y + 5))

        # Folder selection - compact 3 buttons in a row
        folder_y = name_y + 45
        folder_label = self.small_font.render("Folder:", True, WHITE)
        self.screen.blit(folder_label, (dialog_x + 20, folder_y + 8))

        button_width = 85
        button_height = 32
        button_spacing = 8
        folders = ["challenges", "units", "zones"]

        for i, folder in enumerate(folders):
            btn_x = dialog_x + 75 + i * (button_width + button_spacing)
            btn_rect = pygame.Rect(btn_x, folder_y, button_width, button_height)

            # Highlight selected folder
            if folder == self.save_folder_selection:
                pygame.draw.rect(self.screen, YELLOW, btn_rect)
                text_color = BLACK
            else:
                pygame.draw.rect(self.screen, (70, 70, 70), btn_rect)
                text_color = WHITE

            pygame.draw.rect(self.screen, BLACK, btn_rect, 2)

            # Draw folder name (abbreviated)
            folder_display = folder[:4].upper() if len(folder) > 8 else folder.upper()
            folder_text = self.tiny_font.render(folder_display, True, text_color)
            text_x = btn_x + (button_width - folder_text.get_width()) // 2
            text_y = folder_y + (button_height - folder_text.get_height()) // 2
            self.screen.blit(folder_text, (text_x, text_y))

        # Description text - compact
        desc_y = folder_y + button_height + 12
        folder_descriptions = {
            "challenges": "Full scenarios (units + zones)",
            "units": "Unit compositions only",
            "zones": "Zone layouts only",
        }
        desc_text = self.tiny_font.render(
            folder_descriptions[self.save_folder_selection], True, LIGHT_GRAY
        )
        desc_x = dialog_x + (dialog_width - desc_text.get_width()) // 2
        self.screen.blit(desc_text, (desc_x, desc_y))

        # Save and Cancel buttons - compact
        save_cancel_y = dialog_y + dialog_height - 45
        save_btn = pygame.Rect(dialog_x + 60, save_cancel_y, 110, 32)
        cancel_btn = pygame.Rect(dialog_x + 210, save_cancel_y, 110, 32)

        pygame.draw.rect(self.screen, GREEN, save_btn)
        pygame.draw.rect(self.screen, BLACK, save_btn, 2)
        pygame.draw.rect(self.screen, RED, cancel_btn)
        pygame.draw.rect(self.screen, BLACK, cancel_btn, 2)

        save_text = self.small_font.render("Save", True, BLACK)
        cancel_text = self.small_font.render("Cancel", True, BLACK)
        save_x = save_btn.x + (110 - save_text.get_width()) // 2
        save_y = save_btn.y + (32 - save_text.get_height()) // 2
        cancel_x = cancel_btn.x + (110 - cancel_text.get_width()) // 2
        cancel_y = cancel_btn.y + (32 - cancel_text.get_height()) // 2
        self.screen.blit(save_text, (save_x, save_y))
        self.screen.blit(cancel_text, (cancel_x, cancel_y))

=======
>>>>>>> a4fed485df2ae65fcca9dbd5217824b01ea2cb0e
    def draw_load_dialog(self):
        """Draw the load scenario dialog with scrollable file list and search"""
        # Semi-transparent overlay
        overlay = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 180))
        self.screen.blit(overlay, (0, 0))

        # Dialog box - palette style
        dialog_width = 450
        dialog_height = 550
        dialog_x = (SCREEN_WIDTH - dialog_width) // 2
        dialog_y = (SCREEN_HEIGHT - dialog_height) // 2

        # Background
        dialog_surface = pygame.Surface((dialog_width, dialog_height), pygame.SRCALPHA)
        dialog_surface.fill((80, 80, 80, 240))  # Dark gray
        self.screen.blit(dialog_surface, (dialog_x, dialog_y))
        pygame.draw.rect(self.screen, BLACK, (dialog_x, dialog_y, dialog_width, dialog_height), 2)

        # Title bar
        title_bar_height = 30
        pygame.draw.rect(
            self.screen, (50, 50, 50), (dialog_x, dialog_y, dialog_width, title_bar_height)
        )
        pygame.draw.rect(
            self.screen, BLACK, (dialog_x, dialog_y, dialog_width, title_bar_height), 2
        )

        # Title (centered)
        title = self.small_font.render("Load Scenario", True, WHITE)
        title_x = dialog_x + (dialog_width - title.get_width()) // 2
        self.screen.blit(title, (title_x, dialog_y + 7))

        # Search box
        search_y = dialog_y + title_bar_height + 10
        search_label = self.small_font.render("Search:", True, WHITE)
        self.screen.blit(search_label, (dialog_x + 15, search_y + 5))

        search_box = pygame.Rect(dialog_x + 85, search_y, dialog_width - 100, 30)
        search_color = YELLOW if self.search_active else LIGHT_GRAY
        pygame.draw.rect(self.screen, search_color, search_box)
        pygame.draw.rect(self.screen, BLACK, search_box, 2)

        # Display search query
        search_display = self.search_query + ("|" if self.search_active else "")
        search_text = self.small_font.render(search_display, True, BLACK)
        self.screen.blit(search_text, (search_box.x + 10, search_box.y + 5))

        # File list area - background rectangle
        list_y = search_y + 45
        list_height = dialog_height - 150
        list_bg_width = dialog_width - 30

        pygame.draw.rect(
            self.screen, (60, 60, 60), (dialog_x + 15, list_y, list_bg_width, list_height)
        )
        pygame.draw.rect(self.screen, BLACK, (dialog_x + 15, list_y, list_bg_width, list_height), 2)

<<<<<<< HEAD
        # Draw file list with scrolling
=======
        # Use filtered scenarios
        scenarios_to_show = (
            self.filtered_scenarios if self.search_query else self.available_scenarios
        )

        # Draw file list with scrolling - reduce width to show background
>>>>>>> a4fed485df2ae65fcca9dbd5217824b01ea2cb0e
        item_height = 32
        item_width = list_bg_width - 10  # 5px margin on each side
        visible_items = list_height // item_height

<<<<<<< HEAD
        if self.search_query:
            # Search mode - show flat filtered list
            scenarios_to_show = self.filtered_scenarios

            for i in range(visible_items):
                item_index = i + self.load_scroll_offset
                if item_index < len(scenarios_to_show):
                    folder, filename = scenarios_to_show[item_index]
                    item_y = list_y + i * item_height + 2
                    item_x = dialog_x + 20

                    # Highlight selected item
                    is_selected = item_index == self.selected_scenario_index
                    if is_selected:
                        pygame.draw.rect(
                            self.screen, YELLOW, (item_x, item_y, item_width, item_height - 2)
                        )
                    else:
                        pygame.draw.rect(
                            self.screen, (70, 70, 70), (item_x, item_y, item_width, item_height - 2)
                        )

                    # Draw folder/filename
                    display_text = f"[{folder}] {filename}"
                    file_text = self.small_font.render(display_text, True, WHITE)
                    self.screen.blit(file_text, (item_x + 10, item_y + 5))

            # Scroll indicator
            if len(scenarios_to_show) > visible_items:
                scroll_info = self.tiny_font.render(
                    f"Showing {min(self.load_scroll_offset + visible_items, len(scenarios_to_show))}/{len(scenarios_to_show)} (Use mouse wheel to scroll)",
                    True,
                    LIGHT_GRAY,
                )
                scroll_x = dialog_x + (dialog_width - scroll_info.get_width()) // 2
                self.screen.blit(scroll_info, (scroll_x, list_y + list_height + 5))
        else:
            # Folder structure mode
            current_y = list_y + 2
            display_index = 0
            skip_count = self.load_scroll_offset

            for folder_name in ["challenges", "units", "zones"]:
                if folder_name not in self.available_scenarios:
                    continue

                # Skip items for scrolling
                if skip_count > 0:
                    skip_count -= 1
                    display_index += 1
                    if self.folder_expanded.get(folder_name, False):
                        files_to_skip = min(skip_count, len(self.available_scenarios[folder_name]))
                        skip_count -= files_to_skip
                        display_index += files_to_skip
                    continue

                # Check if we're out of visible area
                if current_y + item_height > list_y + list_height:
                    break

                item_x = dialog_x + 20

                # Draw folder header
                is_expanded = self.folder_expanded.get(folder_name, False)
                folder_bg_color = (90, 90, 90)
                pygame.draw.rect(
                    self.screen, folder_bg_color, (item_x, current_y, item_width, item_height - 2)
                )

                # Draw expand/collapse icon (use triangle symbols that render better)
                icon = "▼" if is_expanded else "►"
                icon_text = self.font.render(
                    icon, True, WHITE
                )  # Use main font for better rendering
                self.screen.blit(icon_text, (item_x + 10, current_y + 3))

                # Draw folder name
                folder_text = self.small_font.render(folder_name.upper(), True, YELLOW)
                self.screen.blit(folder_text, (item_x + 40, current_y + 5))

                current_y += item_height
                display_index += 1

                # Draw files if expanded
                if is_expanded:
                    for file_idx, filename in enumerate(self.available_scenarios[folder_name]):
                        # Skip for scrolling
                        if skip_count > 0:
                            skip_count -= 1
                            display_index += 1
                            continue

                        # Check if we're out of visible area
                        if current_y + item_height > list_y + list_height:
                            break

                        # Check if this file is selected
                        is_selected = (
                            self.selected_scenario_index == display_index
                            and self.selected_folder == folder_name
                        )

                        if is_selected:
                            pygame.draw.rect(
                                self.screen,
                                YELLOW,
                                (item_x + 20, current_y, item_width - 20, item_height - 2),
                            )
                        else:
                            pygame.draw.rect(
                                self.screen,
                                (70, 70, 70),
                                (item_x + 20, current_y, item_width - 20, item_height - 2),
                            )

                        # Draw filename (indented)
                        file_text = self.small_font.render(filename, True, WHITE)
                        self.screen.blit(file_text, (item_x + 30, current_y + 5))

                        current_y += item_height
                        display_index += 1

            # Scroll indicator (count total items)
            total_items = len(self.available_scenarios)
            for folder_name in self.available_scenarios:
                if self.folder_expanded.get(folder_name, False):
                    total_items += len(self.available_scenarios[folder_name])

            if total_items > visible_items:
                scroll_info = self.tiny_font.render(
                    f"Use mouse wheel to scroll",
                    True,
                    LIGHT_GRAY,
                )
                scroll_x = dialog_x + (dialog_width - scroll_info.get_width()) // 2
                self.screen.blit(scroll_info, (scroll_x, list_y + list_height + 5))
=======
        for i in range(visible_items):
            item_index = i + self.load_scroll_offset
            if item_index < len(scenarios_to_show):
                filename = scenarios_to_show[item_index]
                item_y = list_y + i * item_height + 2  # 2px margin top/bottom
                item_x = dialog_x + 20  # 5px left margin

                # Highlight selected item
                if item_index == self.selected_scenario_index:
                    pygame.draw.rect(
                        self.screen, YELLOW, (item_x, item_y, item_width, item_height - 2)
                    )
                else:
                    pygame.draw.rect(
                        self.screen, (70, 70, 70), (item_x, item_y, item_width, item_height - 2)
                    )

                # Draw filename
                file_text = self.small_font.render(filename, True, WHITE)
                self.screen.blit(file_text, (item_x + 10, item_y + 5))

        # Scroll indicator
        if len(scenarios_to_show) > visible_items:
            scroll_info = self.tiny_font.render(
                f"Showing {min(self.load_scroll_offset + visible_items, len(scenarios_to_show))}/{len(scenarios_to_show)} (Use mouse wheel to scroll)",
                True,
                LIGHT_GRAY,
            )
            scroll_x = dialog_x + (dialog_width - scroll_info.get_width()) // 2
            self.screen.blit(scroll_info, (scroll_x, list_y + list_height + 5))
>>>>>>> a4fed485df2ae65fcca9dbd5217824b01ea2cb0e

        # Buttons
        button_y = dialog_y + dialog_height - 45
        button_width = 150
        button_height = 35
        load_btn = pygame.Rect(dialog_x + 50, button_y, button_width, button_height)
        cancel_btn = pygame.Rect(dialog_x + 250, button_y, button_width, button_height)

        pygame.draw.rect(self.screen, GREEN, load_btn)
        pygame.draw.rect(self.screen, BLACK, load_btn, 2)
        pygame.draw.rect(self.screen, RED, cancel_btn)
        pygame.draw.rect(self.screen, BLACK, cancel_btn, 2)

        load_text = self.small_font.render("Load", True, BLACK)
        cancel_text = self.small_font.render("Cancel", True, BLACK)
        load_x = load_btn.x + (button_width - load_text.get_width()) // 2
        load_y = load_btn.y + (button_height - load_text.get_height()) // 2
        cancel_x = cancel_btn.x + (button_width - cancel_text.get_width()) // 2
        cancel_y = cancel_btn.y + (button_height - cancel_text.get_height()) // 2
        self.screen.blit(load_text, (load_x, load_y))
        self.screen.blit(cancel_text, (cancel_x, cancel_y))

    def draw_ui(self):
        """Draw UI elements - draggable toolbar"""
        # Draw toolbar
        toolbar_x = self.toolbar_x
        toolbar_y = self.toolbar_y
        toolbar_width = self.toolbar_width
        toolbar_height = self.toolbar_height

        # Toolbar background - same dark gray as palette
        toolbar_surface = pygame.Surface((toolbar_width, toolbar_height), pygame.SRCALPHA)
        toolbar_surface.fill((80, 80, 80, 240))
        self.screen.blit(toolbar_surface, (toolbar_x, toolbar_y))
        pygame.draw.rect(
            self.screen, BLACK, (toolbar_x, toolbar_y, toolbar_width, toolbar_height), 2
        )

        # Title bar for dragging
        title_bar_height = 25
        pygame.draw.rect(
            self.screen, (50, 50, 50), (toolbar_x, toolbar_y, toolbar_width, title_bar_height)
        )
        pygame.draw.rect(
            self.screen, BLACK, (toolbar_x, toolbar_y, toolbar_width, title_bar_height), 1
        )
        title_text = self.small_font.render("TABS Scenario Generator", True, WHITE)
        title_x = toolbar_x + (toolbar_width - title_text.get_width()) // 2
        self.screen.blit(title_text, (title_x, toolbar_y + 5))

        # Buttons - 2x3 layout (compact)
        button_width = 110
        button_height = 25
        button_spacing_x = 10
        button_spacing_y = 5
        start_x = toolbar_x + 10
        start_y = toolbar_y + title_bar_height + 5

        buttons = [
            ("CLEAR", 0, 0),
            ("SAVE", 0, 1),
            ("NEW", 0, 2),
            ("UNITS", 1, 0),
            ("ZONES", 1, 1),
            ("LOAD", 1, 2),
        ]

        for label, row, col in buttons:
            button_x = start_x + col * (button_width + button_spacing_x)
            button_y = start_y + row * (button_height + button_spacing_y)

            # Highlight UNITS/ZONES based on current edit mode
            if label == "UNITS":
                color = YELLOW if self.edit_mode == "unit" else WHITE
            elif label == "ZONES":
                color = YELLOW if self.edit_mode == "zone" else WHITE
            else:
                color = WHITE

            pygame.draw.rect(self.screen, color, (button_x, button_y, button_width, button_height))
            pygame.draw.rect(
                self.screen, BLACK, (button_x, button_y, button_width, button_height), 2
            )

            btn_text = self.small_font.render(label, True, BLACK)
            text_rect = btn_text.get_rect(
                center=(button_x + button_width // 2, button_y + button_height // 2)
            )
            self.screen.blit(btn_text, text_rect)

        # Draw scenario info and help text
        info_y = 10

        # Scenario information
        scenario_info = [
            f"Scenario: {self.scenario_name}",
            f"Field: {self.max_field_height:.1f} x {self.max_field_width:.1f} | Grid: {self.grid_height} x {self.grid_width}",
            f"Unit Spacing: {self.unit_spacing:.2f} | Margins: {FIELD_MARGIN_WIDTH:.1f} x {FIELD_MARGIN_HEIGHT:.1f}",
        ]

        for i, text in enumerate(scenario_info):
            info_text = self.info_font.render(text, True, YELLOW)
            self.screen.blit(info_text, (20, info_y + i * 22))

        info_y += len(scenario_info) * 22 + 10

        # Help text - key descriptions
        if self.edit_mode == "unit":
            help_texts = [
                "Q/E: Rotate | Click: Place | Right-click: Delete | WASD: Move | R: Reset",
                "Ctrl+Z: Undo | Ctrl+Y: Redo",
            ]
        else:
            help_texts = [
                "Drag: Create Zone | Double-click Zone: Edit | Right-click: Delete",
                "WASD: Move | R: Reset | Ctrl+Z: Undo | Ctrl+Y: Redo",
            ]

        for i, text in enumerate(help_texts):
            help_text = self.info_font.render(text, True, WHITE)
            self.screen.blit(help_text, (20, info_y + i * 22))

        # Draw new scenario dialog if active
        if self.show_new_scenario_dialog:
            self.draw_new_scenario_dialog()

<<<<<<< HEAD
        # Draw save dialog if active
        if self.show_save_dialog:
            self.draw_save_dialog()

=======
>>>>>>> a4fed485df2ae65fcca9dbd5217824b01ea2cb0e
        # Draw load dialog if active
        if self.show_load_dialog:
            self.draw_load_dialog()

    def handle_battlefield_click(self, pos, is_drag_start=False, is_right_click=False):
        """Handle clicks on the battlefield"""
        if self.edit_mode == "unit":
            if is_right_click:
                # Right click to delete unit
                self.delete_unit_at_pos(pos)
            else:
                self.handle_unit_placement(pos)
        elif self.edit_mode == "zone":
            if is_right_click:
                # Right click to delete zone
                self.delete_zone_at_pos(pos)
            elif is_drag_start:
                # Check for double-click on zone to pin info box
                import time

                current_time = time.time()

                if current_time - self.last_zone_click_time < self.double_click_threshold:
                    # Double-click detected
                    if self.hovered_zone_info:
                        self.selected_zone_info = self.hovered_zone_info.copy()
                    self.last_zone_click_time = 0  # Reset
                else:
                    # Single click - start drag if not on a zone
                    self.last_zone_click_time = current_time
                    if not self.hovered_zone_info:
                        self.start_zone_drag(pos)

    def delete_unit_at_pos(self, pos):
        """Delete unit at the given position"""
        world_x, world_y = self.screen_to_world(pos[0], pos[1])

        # Find which grid cell was clicked
        for i in range(self.grid_height):
            for j in range(self.grid_width):
                cell_world_x = (j - self.grid_width / 2 + 0.5) * self.actual_unit_spacing_x
                cell_world_y = (i - self.grid_height / 2 + 0.5) * self.actual_unit_spacing_y

                if (
                    abs(world_x - cell_world_x) < self.actual_unit_spacing_x / 2
                    and abs(world_y - cell_world_y) < self.actual_unit_spacing_y / 2
                ):
                    cell_value = self.grid[i, j]
                    if cell_value > 0:
                        unit_id = cell_value
                        # Check if it's Mammoth (2x2)
                        if unit_id == UnitID.Mammoth:
                            # Delete 2x2 cells
                            if i + 1 < self.grid_height and j + 1 < self.grid_width:
                                self.grid[i, j] = 0
                                self.grid[i + 1, j] = 0
                                self.grid[i, j + 1] = 0
                                self.grid[i + 1, j + 1] = 0
                                self.team_grid[i, j] = 0
                                self.team_grid[i + 1, j] = 0
                                self.team_grid[i, j + 1] = 0
                                self.team_grid[i + 1, j + 1] = 0
                                self.rotation_grid[i, j] = 0
                                self.save_state()  # Save after deletion
                        else:
                            # Delete 1x1 unit
                            self.grid[i, j] = 0
                            self.team_grid[i, j] = 0
                            self.rotation_grid[i, j] = 0
                            self.save_state()  # Save after deletion
                    elif cell_value == MAMMOTH_OCCUPIED:
                        # Clicked on a Mammoth's occupied cell, find and delete the whole Mammoth
                        # Find the top-left cell of this Mammoth
                        top_i, top_j = i, j
                        # Search up
                        while top_i > 0 and (
                            self.grid[top_i - 1, j] == UnitID.Mammoth
                            or self.grid[top_i - 1, j] == MAMMOTH_OCCUPIED
                        ):
                            top_i -= 1
                        # Search left
                        while top_j > 0 and (
                            self.grid[top_i, top_j - 1] == UnitID.Mammoth
                            or self.grid[top_i, top_j - 1] == MAMMOTH_OCCUPIED
                        ):
                            top_j -= 1

                        # Delete the Mammoth from top-left
                        if self.grid[top_i, top_j] == UnitID.Mammoth:
                            if top_i + 1 < self.grid_height and top_j + 1 < self.grid_width:
                                self.grid[top_i, top_j] = 0
                                self.grid[top_i + 1, top_j] = 0
                                self.grid[top_i, top_j + 1] = 0
                                self.grid[top_i + 1, top_j + 1] = 0
                                self.team_grid[top_i, top_j] = 0
                                self.team_grid[top_i + 1, top_j] = 0
                                self.team_grid[top_i, top_j + 1] = 0
                                self.team_grid[top_i + 1, top_j + 1] = 0
                                self.rotation_grid[top_i, top_j] = 0
                                self.save_state()
                    return

    def delete_zone_at_pos(self, pos):
        """Delete zone at the given position"""
        world_x, world_y = self.screen_to_world(pos[0], pos[1])

        # Check which zone was clicked (in reverse order to prioritize top zones)
        for idx in range(len(self.zones) - 1, -1, -1):
            zone = self.zones[idx]
            pos_zone = zone["position"]
            axes = zone["axes"]

            # Check if point is inside ellipse
            dx = (world_x - pos_zone[0]) / axes[0]
            dy = (world_y - pos_zone[1]) / axes[1]

            if dx * dx + dy * dy <= 1:
                del self.zones[idx]
                self.save_state()  # Save after zone deletion
                return

    def handle_unit_placement(self, pos):
        """Place or remove units on the battlefield"""
        world_x, world_y = self.screen_to_world(pos[0], pos[1])

        # Check which grid cell was clicked
        for i in range(self.grid_height):
            for j in range(self.grid_width):
                cell_world_x = (j - self.grid_width / 2 + 0.5) * self.actual_unit_spacing_x
                cell_world_y = (i - self.grid_height / 2 + 0.5) * self.actual_unit_spacing_y

                if (
                    abs(world_x - cell_world_x) < self.actual_unit_spacing_x / 2
                    and abs(world_y - cell_world_y) < self.actual_unit_spacing_y / 2
                ):
                    # Check if it's a special unit (Mammoth = 2x2)
                    is_mammoth = self.selected_unit_id == UnitID.Mammoth

                    if is_mammoth:
                        # Mammoth occupies 2x2 space
                        # Check if 2x2 space is available (for placement) or occupied (for removal)
                        if self.selected_unit_id > 0:
                            # Check if bounds are valid
                            if not (i + 1 < self.grid_height and j + 1 < self.grid_width):
                                return

                            # Check if all 4 cells are available (empty or MAMMOTH_OCCUPIED)
                            can_place = True
                            for di in range(2):
                                for dj in range(2):
                                    cell_value = self.grid[i + di, j + dj]
                                    if cell_value != 0 and cell_value != MAMMOTH_OCCUPIED:
                                        can_place = False
                                        break
                                if not can_place:
                                    break

                            if not can_place:
                                return

                            # Place Mammoth: actual unit_id only at top-left, rest are MAMMOTH_OCCUPIED
                            self.grid[i, j] = self.selected_unit_id
                            self.grid[i + 1, j] = MAMMOTH_OCCUPIED
                            self.grid[i, j + 1] = MAMMOTH_OCCUPIED
                            self.grid[i + 1, j + 1] = MAMMOTH_OCCUPIED

                            self.team_grid[i, j] = self.selected_team
                            self.team_grid[i + 1, j] = self.selected_team
                            self.team_grid[i, j + 1] = self.selected_team
                            self.team_grid[i + 1, j + 1] = self.selected_team

                            # Default rotation for top-left cell only
                            if self.selected_team == 0:
                                self.rotation_grid[i, j] = 0  # Ally faces right
                            else:
                                self.rotation_grid[i, j] = np.pi  # Enemy faces left
                            self.save_state()  # Save after placement
                        else:
                            # Remove Mammoth (erase mode)
                            if i + 1 < self.grid_height and j + 1 < self.grid_width:
                                self.grid[i, j] = 0
                                self.grid[i + 1, j] = 0
                                self.grid[i, j + 1] = 0
                                self.grid[i + 1, j + 1] = 0
                                self.team_grid[i, j] = 0
                                self.team_grid[i + 1, j] = 0
                                self.team_grid[i, j + 1] = 0
                                self.team_grid[i + 1, j + 1] = 0
                                self.rotation_grid[i, j] = 0
                    else:
                        # Normal 1x1 unit placement
                        # Check if placing on Mammoth - not allowed
                        if self.selected_unit_id > 0:
                            cell_value = self.grid[i, j]
                            if cell_value == UnitID.Mammoth or cell_value == MAMMOTH_OCCUPIED:
                                return

                        self.grid[i, j] = self.selected_unit_id
                        if self.selected_unit_id > 0:
                            self.team_grid[i, j] = self.selected_team
                            # Default rotation: 0 (facing right for ally, left for enemy)
                            if self.selected_team == 0:
                                self.rotation_grid[i, j] = 0  # Ally faces right
                            else:
                                self.rotation_grid[i, j] = np.pi  # Enemy faces left
                            self.save_state()  # Save after placement
                    return

    def start_zone_drag(self, pos):
        """Start dragging to create a zone"""
        self.dragging_zone = True
        self.zone_start_pos = pos
        self.current_zone_rect = None

    def update_zone_drag(self, pos):
        """Update zone rectangle while dragging"""
        if self.dragging_zone and self.zone_start_pos:
            x1, y1 = self.zone_start_pos
            x2, y2 = pos

            left = min(x1, x2)
            top = min(y1, y2)
            width = abs(x2 - x1)
            height = abs(y2 - y1)

            self.current_zone_rect = (left, top, width, height)

    def finish_zone_drag(self, pos):
        """Finish dragging and create the zone"""
        if self.dragging_zone and self.zone_start_pos and self.current_zone_rect:
            # Convert screen rectangle to world coordinates
            x1, y1 = self.zone_start_pos
            x2, y2 = pos

            world_x1, world_y1 = self.screen_to_world(x1, y1)
            world_x2, world_y2 = self.screen_to_world(x2, y2)

            # Calculate center and half-extents
            center_x = (world_x1 + world_x2) / 2
            center_y = (world_y1 + world_y2) / 2
            half_width = abs(world_x2 - world_x1) / 2
            half_height = abs(world_y2 - world_y1) / 2

            # Only create zone if it has some size
            if half_width > 0.5 and half_height > 0.5:
                zone = {
                    "type": self.selected_zone_type,
                    "position": [center_x, center_y],
                    "axes": [half_width, half_height],
                    "effect_value": self.zone_effect_value,
                }
                self.zones.append(zone)
                self.save_state()  # Save after zone creation

        self.dragging_zone = False
        self.zone_start_pos = None
        self.current_zone_rect = None

    def clear_scenario(self):
        """Clear all units and zones"""
        self.grid = np.zeros((self.grid_height, self.grid_width), dtype=int)
        self.team_grid = np.zeros((self.grid_height, self.grid_width), dtype=int)
        self.rotation_grid = np.zeros((self.grid_height, self.grid_width), dtype=float)
        self.zones = []
        self.save_state()  # Save state after clearing

    def handle_ui_click(self, pos):
        """Handle UI button clicks. Returns True if a UI element was clicked."""
        x, y = pos

        # Handle new scenario dialog if open
        if self.show_new_scenario_dialog:
            return self.handle_new_scenario_dialog_click(pos)

<<<<<<< HEAD
        # Handle save dialog if open
        if self.show_save_dialog:
            return self.handle_save_dialog_click(pos)

=======
>>>>>>> a4fed485df2ae65fcca9dbd5217824b01ea2cb0e
        # Handle load dialog if open
        if self.show_load_dialog:
            return self.handle_load_dialog_click(pos)

        # Check toolbar buttons - 2x3 layout
        toolbar_x = self.toolbar_x
        toolbar_y = self.toolbar_y
        title_bar_height = 25
        start_x = toolbar_x + 10
        start_y = toolbar_y + title_bar_height + 5
        button_width = 110
        button_height = 25
        button_spacing_x = 10
        button_spacing_y = 5

        buttons = [
            ("CLEAR", 0, 0),
            ("SAVE", 0, 1),
            ("NEW", 0, 2),
            ("UNITS", 1, 0),
            ("ZONES", 1, 1),
            ("LOAD", 1, 2),
        ]

        for label, row, col in buttons:
            button_x = start_x + col * (button_width + button_spacing_x)
            button_y = start_y + row * (button_height + button_spacing_y)
            if (
                button_x <= x <= button_x + button_width
                and button_y <= y <= button_y + button_height
            ):
                if label == "CLEAR":
                    self.clear_scenario()
                elif label == "SAVE":
                    self.save_scenario()
                elif label == "NEW":
                    self.show_new_scenario_dialog = True
                    self.active_input_field = "max_height"
                elif label == "UNITS":
                    self.edit_mode = "unit"
                elif label == "ZONES":
                    self.edit_mode = "zone"
                elif label == "LOAD":
                    self.load_scenario()
                return True

        # Check Unit palette (only if in unit mode)
        if self.edit_mode == "unit":
            panel_x = self.unit_panel_x
            panel_y = self.unit_panel_y
            panel_width = self.unit_panel_width
            panel_height = self.unit_panel_height
            title_bar_height = 30

            # Check if clicking inside panel
            if panel_x <= x <= panel_x + panel_width and panel_y <= y <= panel_y + panel_height:
                # Team selection buttons
                team_y = panel_y + title_bar_height + 10
                if panel_x + 10 <= x <= panel_x + 180 and team_y <= y <= team_y + 40:
                    self.selected_team = 0
                    return True
                if panel_x + 190 <= x <= panel_x + 360 and team_y <= y <= team_y + 40:
                    self.selected_team = 1
                    return True

                # Unit buttons
                unit_button_size = 110
                for idx in range(len(self.unit_names)):
                    unit_id = idx + 1
                    row = idx // 3
                    col = idx % 3
                    btn_x = panel_x + 10 + col * (unit_button_size + 10)
                    btn_y = panel_y + title_bar_height + 80 + row * (unit_button_size + 15)
                    if (
                        btn_x <= x <= btn_x + unit_button_size
                        and btn_y <= y <= btn_y + unit_button_size
                    ):
                        self.selected_unit_id = unit_id
                        return True
                return True  # Clicked inside panel

        # Check Zone palette (only if in zone mode)
<<<<<<< HEAD
        # Check Zone palette (only if in zone mode)
=======
>>>>>>> a4fed485df2ae65fcca9dbd5217824b01ea2cb0e
        if self.edit_mode == "zone":
            panel_x = self.zone_panel_x
            panel_y = self.zone_panel_y
            panel_width = self.zone_panel_width
            panel_height = self.zone_panel_height
            title_bar_height = 30

            # Check if clicking inside panel
            if panel_x <= x <= panel_x + panel_width and panel_y <= y <= panel_y + panel_height:
<<<<<<< HEAD
                # ---- EXACT y_offset flow from draw_zone_palette ----
                y_offset = panel_y + title_bar_height + 5  # instructions
                y_offset += 20  # zone count
                y_offset += 30  # space before buttons

                btn_width = 110
                btn_height = 40
                btn_spacing = 5

                # Zone type buttons
                lava_x = panel_x + 10
                bush_x = panel_x + 10 + btn_width + btn_spacing
                swamp_x = panel_x + 10 + (btn_width + btn_spacing) * 2

                if lava_x <= x <= lava_x + btn_width and y_offset <= y <= y_offset + btn_height:
=======
                # y_offset matches draw_zone_palette: title(30) + inst(5+20) + count(20+30) = 85
                y_offset = panel_y + title_bar_height + 55
                btn_width = 110
                btn_spacing = 5

                # Zone type buttons (110x40 with 5px spacing)
                lava_x_pos = panel_x + 10
                if lava_x_pos <= x <= lava_x_pos + btn_width and y_offset <= y <= y_offset + 40:
>>>>>>> a4fed485df2ae65fcca9dbd5217824b01ea2cb0e
                    self.selected_zone_type = 1
                    self.zone_effect_value = 10.0
                    return True

<<<<<<< HEAD
                if bush_x <= x <= bush_x + btn_width and y_offset <= y <= y_offset + btn_height:
=======
                bush_x_pos = panel_x + 10 + btn_width + btn_spacing
                if bush_x_pos <= x <= bush_x_pos + btn_width and y_offset <= y <= y_offset + 40:
>>>>>>> a4fed485df2ae65fcca9dbd5217824b01ea2cb0e
                    self.selected_zone_type = 2
                    self.zone_effect_value = 0.0
                    return True

<<<<<<< HEAD
                if swamp_x <= x <= swamp_x + btn_width and y_offset <= y <= y_offset + btn_height:
=======
                swamp_x_pos = panel_x + 10 + (btn_width + btn_spacing) * 2
                if swamp_x_pos <= x <= swamp_x_pos + btn_width and y_offset <= y <= y_offset + 40:
>>>>>>> a4fed485df2ae65fcca9dbd5217824b01ea2cb0e
                    self.selected_zone_type = 3
                    self.zone_effect_value = 0.2
                    return True

<<<<<<< HEAD
                # Zone description
                y_offset += 48

                # Effect label
                y_offset += 25

                # Input row
                y_offset += 25
                input_width = 240
                apply_width = 100

                input_box_rect = pygame.Rect(panel_x + 10, y_offset, input_width, 35)
                apply_btn_rect = pygame.Rect(
                    panel_x + 10 + input_width + 10, y_offset, apply_width, 35
=======
                # Effect value input (button + 50 + label + 25)
                effect_input_y = y_offset + 75
                input_width = 240
                apply_width = 100
                input_box_rect = pygame.Rect(panel_x + 10, effect_input_y, input_width, 35)
                apply_btn_rect = pygame.Rect(
                    panel_x + 10 + input_width + 10, effect_input_y, apply_width, 35
>>>>>>> a4fed485df2ae65fcca9dbd5217824b01ea2cb0e
                )

                if input_box_rect.collidepoint(x, y) and self.selected_zone_type != 2:
                    self.text_input_active = True
                    self.text_input_value = ""
                    return True

                if apply_btn_rect.collidepoint(x, y) and self.selected_zone_type != 2:
                    if self.text_input_value:
                        try:
                            new_value = float(self.text_input_value)
                            if self.selected_zone_type == 1:
                                if 0 <= new_value <= 100:
                                    self.zone_effect_value = new_value
                            else:
                                if 0 <= new_value <= 1:
                                    self.zone_effect_value = new_value
                        except ValueError:
                            pass
<<<<<<< HEAD

                    self.text_input_active = False
                    self.text_input_value = ""
                    return True

=======
                    self.text_input_active = False
                    self.text_input_value = ""
                    return True
>>>>>>> a4fed485df2ae65fcca9dbd5217824b01ea2cb0e
                return True  # Clicked inside panel

        return False

    def handle_new_scenario_dialog_click(self, pos):
<<<<<<< HEAD
        """Handle clicks in the new scenario dialog (aligned with draw_new_scenario_dialog)"""
        x, y = pos

        # MUST match draw_new_scenario_dialog
        dialog_width = 400
        dialog_height = 330
        dialog_x = (SCREEN_WIDTH - dialog_width) // 2
        dialog_y = (SCREEN_HEIGHT - dialog_height) // 2

        title_bar_height = 30

        # Input fields (same order as draw, scenario_name removed)
        y_offset = dialog_y + title_bar_height + 15
        fields = [
            "max_height",
            "max_width",
            "margin_h",
            "margin_w",
=======
        """Handle clicks in the new scenario dialog"""
        x, y = pos

        dialog_width = 400
        dialog_height = 420
        dialog_x = (SCREEN_WIDTH - dialog_width) // 2
        dialog_y = (SCREEN_HEIGHT - dialog_height) // 2

        # Check input field clicks
        title_bar_height = 30
        y_offset = dialog_y + title_bar_height + 15
        fields = [
            "scenario_name",
            "max_height",
            "max_width",
            "margin_w",
            "margin_h",
>>>>>>> a4fed485df2ae65fcca9dbd5217824b01ea2cb0e
            "unit_spacing",
        ]

        for field_name in fields:
            input_box = pygame.Rect(dialog_x + 180, y_offset - 3, 200, 28)
            if input_box.collidepoint(x, y):
                self.active_input_field = field_name
                return True
            y_offset += 50

<<<<<<< HEAD
        # Buttons (must match draw)
        button_y = dialog_y + dialog_height - 40
        button_width = 150
        button_height = 35

=======
        # Check button clicks
        button_y = dialog_y + dialog_height - 50
        button_width = 150
        button_height = 35
>>>>>>> a4fed485df2ae65fcca9dbd5217824b01ea2cb0e
        create_btn = pygame.Rect(dialog_x + 30, button_y, button_width, button_height)
        cancel_btn = pygame.Rect(dialog_x + 220, button_y, button_width, button_height)

        if create_btn.collidepoint(x, y):
            self.create_new_scenario()
            return True
<<<<<<< HEAD

        if cancel_btn.collidepoint(x, y):
=======
        elif cancel_btn.collidepoint(x, y):
>>>>>>> a4fed485df2ae65fcca9dbd5217824b01ea2cb0e
            self.show_new_scenario_dialog = False
            self.active_input_field = None
            return True

        return True  # Consume all clicks when dialog is open

<<<<<<< HEAD
    def handle_save_dialog_click(self, pos):
        """Handle clicks in the save scenario dialog"""
        x, y = pos

        # MUST match draw_save_dialog()
        dialog_width = 380
        dialog_height = 205
        dialog_x = (SCREEN_WIDTH - dialog_width) // 2
        dialog_y = (SCREEN_HEIGHT - dialog_height) // 2
        title_bar_height = 30

        # Scenario name input box
        name_y = dialog_y + title_bar_height + 15
        name_input_box = pygame.Rect(dialog_x + 75, name_y, 285, 28)

        if name_input_box.collidepoint(x, y):
            self.save_name_input_active = True
            return True
        else:
            self.save_name_input_active = False

        # Folder selection buttons
        folder_y = name_y + 45
        button_width = 85
        button_height = 32
        button_spacing = 8
        folders = ["challenges", "units", "zones"]

        for i, folder in enumerate(folders):
            btn_x = dialog_x + 75 + i * (button_width + button_spacing)
            btn_rect = pygame.Rect(btn_x, folder_y, button_width, button_height)

            if btn_rect.collidepoint(x, y):
                self.save_folder_selection = folder
                return True

        # Save and Cancel buttons
        save_cancel_y = dialog_y + dialog_height - 45
        save_btn = pygame.Rect(dialog_x + 60, save_cancel_y, 110, 32)
        cancel_btn = pygame.Rect(dialog_x + 210, save_cancel_y, 110, 32)

        if save_btn.collidepoint(x, y):
            self.save_scenario_file()
            return True
        elif cancel_btn.collidepoint(x, y):
            self.show_save_dialog = False
            self.save_name_input_active = False
            return True

        return True

=======
>>>>>>> a4fed485df2ae65fcca9dbd5217824b01ea2cb0e
    def handle_load_dialog_click(self, pos):
        """Handle clicks in the load scenario dialog"""
        x, y = pos

        dialog_width = 450
        dialog_height = 550
        dialog_x = (SCREEN_WIDTH - dialog_width) // 2
        dialog_y = (SCREEN_HEIGHT - dialog_height) // 2
        title_bar_height = 30

        # Check search box click
        search_y = dialog_y + title_bar_height + 10
        search_box = pygame.Rect(dialog_x + 85, search_y, dialog_width - 100, 30)
        if search_box.collidepoint(x, y):
            self.search_active = True
            return True
        else:
            # Click outside search box - deactivate
            if self.search_active:
                self.search_active = False

        # Check if clicking on file list
        list_y = search_y + 45
        list_height = dialog_height - 150
        list_bg_width = dialog_width - 30
        item_height = 32
        item_width = list_bg_width - 10

<<<<<<< HEAD
        if dialog_x + 20 <= x <= dialog_x + 20 + item_width and list_y <= y <= list_y + list_height:
            if self.search_query:
                # Search mode - flat list
                relative_y = y - list_y - 2  # Account for 2px top margin
                clicked_row = relative_y // item_height
                actual_index = clicked_row + self.load_scroll_offset
                if actual_index < len(self.filtered_scenarios):
                    self.selected_scenario_index = actual_index
                    self.selected_folder = self.filtered_scenarios[actual_index][0]
                    return True
            else:
                # Folder structure mode - use actual Y position matching
                current_y = list_y + 2
                display_index = 0
                skip_count = self.load_scroll_offset

                for folder_name in ["challenges", "units", "zones"]:
                    if folder_name not in self.available_scenarios:
                        continue

                    # Skip items for scrolling
                    if skip_count > 0:
                        skip_count -= 1
                        display_index += 1
                        if self.folder_expanded.get(folder_name, False):
                            files_to_skip = min(
                                skip_count, len(self.available_scenarios[folder_name])
                            )
                            skip_count -= files_to_skip
                            display_index += files_to_skip
                        continue

                    # Check if we're out of visible area
                    if current_y + item_height > list_y + list_height:
                        break

                    # Check if folder header was clicked
                    if current_y <= y < current_y + item_height:
                        # Toggle folder expansion
                        self.folder_expanded[folder_name] = not self.folder_expanded.get(
                            folder_name, False
                        )
                        return True

                    current_y += item_height
                    display_index += 1

                    # Check files if expanded
                    if self.folder_expanded.get(folder_name, False):
                        for file_idx, filename in enumerate(self.available_scenarios[folder_name]):
                            # Skip for scrolling
                            if skip_count > 0:
                                skip_count -= 1
                                display_index += 1
                                continue

                            # Check if we're out of visible area
                            if current_y + item_height > list_y + list_height:
                                break

                            # Check if this file was clicked
                            if current_y <= y < current_y + item_height:
                                # File was clicked
                                self.selected_scenario_index = display_index
                                self.selected_folder = folder_name
                                return True

                            current_y += item_height
                            display_index += 1
=======
        scenarios_to_show = (
            self.filtered_scenarios if self.search_query else self.available_scenarios
        )

        if dialog_x + 20 <= x <= dialog_x + 20 + item_width and list_y <= y <= list_y + list_height:
            # Calculate which item was clicked
            relative_y = y - list_y - 2  # Account for 2px top margin
            item_index = relative_y // item_height
            actual_index = item_index + self.load_scroll_offset

            if actual_index < len(scenarios_to_show):
                self.selected_scenario_index = actual_index
                return True
>>>>>>> a4fed485df2ae65fcca9dbd5217824b01ea2cb0e

        # Check button clicks
        button_y = dialog_y + dialog_height - 45
        button_width = 150
        button_height = 35
        load_btn = pygame.Rect(dialog_x + 50, button_y, button_width, button_height)
        cancel_btn = pygame.Rect(dialog_x + 250, button_y, button_width, button_height)

        if load_btn.collidepoint(x, y):
<<<<<<< HEAD
            if self.selected_scenario_index is not None and self.selected_folder is not None:
                if self.search_query:
                    folder, filename = self.filtered_scenarios[self.selected_scenario_index]
                    self.load_scenario_file(folder, filename)
                else:
                    # Find the selected file in folder structure
                    display_index = 0
                    skip_count = self.load_scroll_offset

                    for folder_name in ["challenges", "units", "zones"]:
                        if folder_name not in self.available_scenarios:
                            continue

                        if skip_count > 0:
                            skip_count -= 1
                            display_index += 1
                            if self.folder_expanded.get(folder_name, False):
                                files_to_skip = min(
                                    skip_count, len(self.available_scenarios[folder_name])
                                )
                                skip_count -= files_to_skip
                                display_index += files_to_skip
                            continue

                        # Skip folder header
                        display_index += 1

                        if self.folder_expanded.get(folder_name, False):
                            for filename in self.available_scenarios[folder_name]:
                                if skip_count > 0:
                                    skip_count -= 1
                                    display_index += 1
                                    continue

                                if display_index == self.selected_scenario_index:
                                    self.load_scenario_file(folder_name, filename)
                                    return True

                                display_index += 1
            return True
        elif cancel_btn.collidepoint(x, y):
            self.show_load_dialog = False
            # Keep selected scenario and folder (don't reset)
=======
            if self.selected_scenario_index is not None:
                filename = scenarios_to_show[self.selected_scenario_index]
                self.load_scenario_file(filename)
            return True
        elif cancel_btn.collidepoint(x, y):
            self.show_load_dialog = False
            self.selected_scenario_index = None
>>>>>>> a4fed485df2ae65fcca9dbd5217824b01ea2cb0e
            self.search_query = ""
            self.search_active = False
            return True

        return True  # Consume all clicks when dialog is open

    def create_new_scenario(self):
        """Create a new scenario with the specified parameters"""
        try:
            new_max_height = float(self.new_scenario_inputs["max_height"])
            new_max_width = float(self.new_scenario_inputs["max_width"])
            new_margin_w = float(self.new_scenario_inputs["margin_w"])
            new_margin_h = float(self.new_scenario_inputs["margin_h"])
            new_unit_spacing = float(self.new_scenario_inputs["unit_spacing"])

            # Validate inputs
            if new_max_height <= 0 or new_max_width <= 0:
                return
            if new_max_height > 200 or new_max_width > 200:
                return
            if new_unit_spacing <= 0:
                return

            # Update field size
            self.max_field_height = new_max_height
            self.max_field_width = new_max_width
            self.unit_spacing = new_unit_spacing
<<<<<<< HEAD
=======
            self.scenario_name = self.new_scenario_inputs.get("scenario_name", "new_scenario")
>>>>>>> a4fed485df2ae65fcca9dbd5217824b01ea2cb0e

            # Calculate grid size based on fixed unit spacing
            self.grid_height = int(new_max_height / self.unit_spacing)
            self.grid_width = int(new_max_width / self.unit_spacing)

            # Update actual unit spacing (use fixed value)
            self.actual_unit_spacing_x = self.unit_spacing
            self.actual_unit_spacing_y = self.unit_spacing

            # Update margins (globally)
            global FIELD_MARGIN_WIDTH, FIELD_MARGIN_HEIGHT
            FIELD_MARGIN_WIDTH = new_margin_w
            FIELD_MARGIN_HEIGHT = new_margin_h

            # Recreate grids
            self.grid = np.zeros((self.grid_height, self.grid_width), dtype=int)
            self.team_grid = np.zeros((self.grid_height, self.grid_width), dtype=int)
            self.rotation_grid = np.zeros((self.grid_height, self.grid_width), dtype=float)

            # Clear zones
            self.zones = []

            # Recalculate field boundaries
            self.calculate_field_boundaries()

            # Reset history
            self.history = []
            self.history_index = -1
            self.save_state()  # Save initial empty state

            # Close dialog
            self.show_new_scenario_dialog = False
            self.active_input_field = None

        except ValueError:
            pass

    def handle_zoom(self, direction):
        """Handle zoom in/out with mouse wheel"""
        if direction > 0:  # Zoom in
            new_scale = self.scale + self.zoom_speed
        else:  # Zoom out
            new_scale = self.scale - self.zoom_speed

        # Clamp scale to min/max
        new_scale = max(self.min_scale, min(self.max_scale, new_scale))

        if new_scale != self.scale:
            self.scale = new_scale
            self.update_field_size()

    def rotate_unit_at_mouse(self, angle_delta):
        """Rotate unit at current mouse position"""
        if self.edit_mode != "unit":
            return

        mouse_pos = pygame.mouse.get_pos()
        world_x, world_y = self.screen_to_world(mouse_pos[0], mouse_pos[1])

        # Find which grid cell the mouse is over
        for i in range(self.grid_height):
            for j in range(self.grid_width):
                cell_world_x = (j - self.grid_width / 2 + 0.5) * self.actual_unit_spacing_x
                cell_world_y = (i - self.grid_height / 2 + 0.5) * self.actual_unit_spacing_y

                if (
                    abs(world_x - cell_world_x) < self.actual_unit_spacing_x / 2
                    and abs(world_y - cell_world_y) < self.actual_unit_spacing_y / 2
                ):
                    # Found the cell, rotate the unit if present
                    if self.grid[i, j] > 0:
                        self.rotation_grid[i, j] += angle_delta
                        # Snap to 15 degree (π/12 radian) increments
                        angle_step = np.pi / 12  # 15 degrees
                        self.rotation_grid[i, j] = (
                            round(self.rotation_grid[i, j] / angle_step) * angle_step
                        )
                        # Normalize angle to [0, 2π)
                        self.rotation_grid[i, j] = self.rotation_grid[i, j] % (2 * np.pi)
                        self.save_state()  # Save after rotation
                    return

    def save_scenario(self):
<<<<<<< HEAD
        """Show save scenario dialog"""
        self.show_save_dialog = True
        # Set default folder based on current content
        if len(self.zones) > 0 and len([u for row in self.grid for u in row if u > 0]) == 0:
            self.save_folder_selection = "zones"
        elif len(self.zones) == 0 and len([u for row in self.grid for u in row if u > 0]) > 0:
            self.save_folder_selection = "units"
        else:
            self.save_folder_selection = "challenges"

    def save_scenario_file(self):
=======
>>>>>>> a4fed485df2ae65fcca9dbd5217824b01ea2cb0e
        """Save scenario as JSON (env_params.json format)"""
        # Extract unit information from grid data (separate ally and enemy)
        ally_units = []  # (position, unit_id, team, rotation)
        enemy_units = []

        for i in range(self.grid_height):
            for j in range(self.grid_width):
                if self.grid[i, j] > 0:
                    unit_id = int(self.grid[i, j])

                    # Save Mammoth only from top-left cell (avoid 4-cell duplication)
                    if unit_id == UnitID.Mammoth:
                        if i > 0 and self.grid[i - 1, j] == UnitID.Mammoth:
                            continue
                        if j > 0 and self.grid[i, j - 1] == UnitID.Mammoth:
                            continue

                    team = int(self.team_grid[i, j])
                    rotation = float(self.rotation_grid[i, j])

                    # Convert to world coordinates
                    if unit_id == UnitID.Mammoth:
                        # Mammoth position calculated based on 2x2 center
                        # If top-left is (i,j), center is (i+0.5, j+0.5)
                        world_x = (j + 0.5 - self.grid_width / 2 + 0.5) * self.actual_unit_spacing_x
                        world_y = (
                            i + 0.5 - self.grid_height / 2 + 0.5
                        ) * self.actual_unit_spacing_y
                    else:
                        # Normal unit at cell center
                        world_x = (j - self.grid_width / 2 + 0.5) * self.actual_unit_spacing_x
                        world_y = (i - self.grid_height / 2 + 0.5) * self.actual_unit_spacing_y

                    unit_data = ([world_x, world_y], unit_id - 1, team, rotation)

                    # Separate by team
                    if team == 0:  # Ally
                        ally_units.append(unit_data)
                    else:  # Enemy
                        enemy_units.append(unit_data)

        # Combine Ally first, Enemy later
        all_units = ally_units + enemy_units

        # Reconstruct data
        positions = []
        unit_ids = []
        teams = []
        rotations = []

        for pos, uid, team, rot in all_units:
            positions.append(pos)
            unit_ids.append([uid])
            teams.append([team])
            rotations.append([rot])

        # Load unit specs
        all_spec = self.all_spec

        # Calculate pos_min, pos_max (based on max_field_width, max_field_height)
        # pos_max = (max_field_width/2, max_field_height/2) - body_radius
        pos_max = []
        pos_min = []
        for unit_id in unit_ids:
            idx = unit_id[0]
            if idx < len(all_spec["body_radiuses"]):
                body_radius = float(all_spec["body_radiuses"][idx])
            else:
                body_radius = 1.0
            pos_max_x = self.max_field_width / 2 - body_radius
            pos_max_y = self.max_field_height / 2 - body_radius
            pos_max.append([pos_max_x, pos_max_y])
            pos_min.append([-pos_max_x, -pos_max_y])

        # Extract unit specs
        attack_cooldowns = []
        attack_damages = []
        attack_ranges = []
        attack_types = []
        body_radiuss = []
        body_weights = []
        healths = []
        is_alive = []
        is_disabled = []
        sight_angles = []
        speeds = []

        for unit_id in unit_ids:
            idx = unit_id[0]
            if idx < len(all_spec["attack_cooldown"]):
                attack_cooldowns.append([float(all_spec["attack_cooldown"][idx])])
                attack_damages.append([float(all_spec["attack_damages"][idx])])
                attack_ranges.append([float(all_spec["attack_ranges"][idx])])
                attack_types.append(
                    [1 if float(all_spec["attack_damages"][idx]) < 0 else 0]
                )  # heal = 1
                body_radiuss.append([float(all_spec["body_radiuses"][idx])])
                body_weights.append([float(all_spec["body_weights"][idx])])
                healths.append([float(all_spec["healths"][idx])])
                sight_angles.append([float(all_spec["sight_angles"][idx])])
                speeds.append([float(all_spec["speeds"][idx])])
            else:
                # Default values
                attack_cooldowns.append([1.0])
                attack_damages.append([0.0])
                attack_ranges.append([1.0])
                attack_types.append([0])
                body_radiuss.append([1.0])
                body_weights.append([1.0])
                healths.append([1.0])
                sight_angles.append([1.5708])  # π/2
                speeds.append([1.0])

            is_alive.append([True])
            is_disabled.append([False])

        # Extract zone data
        zone_types = []
        zone_positions = []
        zone_axes = []
        zone_effects = []

        for zone in self.zones:
            zone_types.append([zone["type"]])
            zone_positions.append(zone["position"])
            zone_axes.append(zone["axes"])
            zone_effects.append([zone["effect_value"]])

        # Create JSON structure (env_params.json format, excluding physics_params)
        scenario_data = {
            "grid_info": {
                "grid_width": self.grid_width,
                "grid_height": self.grid_height,
                "max_field_width": self.max_field_width,
                "max_field_height": self.max_field_height,
                "margin_width": FIELD_MARGIN_WIDTH,
                "margin_height": FIELD_MARGIN_HEIGHT,
            },
            "scenario": {
                "attack_cooldowns": attack_cooldowns if attack_cooldowns else [],
                "attack_damages": attack_damages if attack_damages else [],
                "attack_ranges": attack_ranges if attack_ranges else [],
                "attack_types": attack_types if attack_types else [],
                "body_radiuss": body_radiuss if body_radiuss else [],
                "body_weights": body_weights if body_weights else [],
                "healths": healths if healths else [],
                "is_alive": is_alive if is_alive else [],
                "is_disabled": is_disabled if is_disabled else [],
                "pos_max": pos_max if pos_max else [],
                "pos_min": pos_min if pos_min else [],
                "positions": positions if positions else [],
                "rotations": rotations if rotations else [],
                "sight_angles": sight_angles if sight_angles else [],
                "speeds": speeds if speeds else [],
                "teams": teams if teams else [],
                "unit_ids": unit_ids if unit_ids else [],
            },
            "zone_scenario": {
                "axes": zone_axes if zone_axes else [],
                "effect_value": zone_effects if zone_effects else [],
                "n_zone": len(self.zones),
                "position": zone_positions if zone_positions else [],
                "zone_type": zone_types if zone_types else [],
            },
        }

<<<<<<< HEAD
        # Create scenarios folder (if not exists)
        folder_path = f"src/scenarios/{self.save_folder_selection}"
        os.makedirs(folder_path, exist_ok=True)

        # Save JSON file with scenario name
        scenario_name = self.save_scenario_name if self.save_scenario_name else "new_scenario"
        file_path = f"{folder_path}/{scenario_name}.json"
        with open(file_path, "w") as f:
            json.dump(scenario_data, f, indent=2)

        # Close save dialog
        self.show_save_dialog = False
        self.save_name_input_active = False

    def load_scenario(self):
        """Display scenario load dialog"""
        # Load JSON files from all scenario subfolders
        scenarios_base = "src/scenarios"
        self.available_scenarios = {}
        self.folder_expanded = {}

        if os.path.exists(scenarios_base):
            # Get all subdirectories
            for folder_name in ["challenges", "units", "zones"]:
                folder_path = os.path.join(scenarios_base, folder_name)
                if os.path.exists(folder_path):
                    files = [f for f in os.listdir(folder_path) if f.endswith(".json")]
                    if files:
                        self.available_scenarios[folder_name] = sorted(files, reverse=True)
                        self.folder_expanded[folder_name] = True  # Start expanded
=======
        # Create scenarios/challenges folder (if not exists)
        os.makedirs("src/scenarios/challenges", exist_ok=True)

        # Save JSON file with scenario name
        scenario_name = self.new_scenario_inputs.get("scenario_name", "new_scenario")
        file_path = f"src/scenarios/challenges/{scenario_name}.json"
        with open(file_path, "w") as f:
            json.dump(scenario_data, f, indent=2)

    def load_scenario(self):
        """Display scenario load dialog"""
        # Load JSON file list from scenarios/challenges folder
        scenarios_path = "src/scenarios/challenges"
        if os.path.exists(scenarios_path):
            files = [f for f in os.listdir(scenarios_path) if f.endswith(".json")]
            self.available_scenarios = sorted(files, reverse=True)  # Latest files first
        else:
            self.available_scenarios = []
>>>>>>> a4fed485df2ae65fcca9dbd5217824b01ea2cb0e

        self.show_load_dialog = True
        self.load_scroll_offset = 0
        self.selected_scenario_index = None
<<<<<<< HEAD
        self.selected_folder = None
=======
>>>>>>> a4fed485df2ae65fcca9dbd5217824b01ea2cb0e
        self.search_query = ""
        self.search_active = False
        self.filtered_scenarios = []

<<<<<<< HEAD
    def load_scenario_file(self, folder, filename):
        """Load selected scenario file"""
        file_path = f"src/scenarios/{folder}/{filename}"
=======
    def load_scenario_file(self, filename):
        """Load selected scenario file"""
        file_path = f"src/scenarios/challenges/{filename}"
>>>>>>> a4fed485df2ae65fcca9dbd5217824b01ea2cb0e
        try:
            with open(file_path, "r") as f:
                data = json.load(f)

            # Update scenario name (remove .json from filename)
            self.scenario_name = filename.replace(".json", "")

            # Restore saved grid information
            grid_info = data.get("grid_info", {})
            if grid_info:
                self.grid_width = int(grid_info.get("grid_width", self.grid_width))
                self.grid_height = int(grid_info.get("grid_height", self.grid_height))
                self.max_field_width = float(grid_info.get("max_field_width", self.max_field_width))
                self.max_field_height = float(
                    grid_info.get("max_field_height", self.max_field_height)
                )

                # Restore margin information
                global FIELD_MARGIN_WIDTH, FIELD_MARGIN_HEIGHT
                FIELD_MARGIN_WIDTH = float(grid_info.get("margin_width", FIELD_MARGIN_WIDTH))
                FIELD_MARGIN_HEIGHT = float(grid_info.get("margin_height", FIELD_MARGIN_HEIGHT))

                self.actual_unit_spacing_x = (
                    self.max_field_width / self.grid_width if self.grid_width > 0 else UNIT_SPACING
                )
                self.actual_unit_spacing_y = (
                    self.max_field_height / self.grid_height
                    if self.grid_height > 0
                    else UNIT_SPACING
                )

                # Update world_width, world_height
                self.world_width = self.max_field_width + 2 * FIELD_MARGIN_WIDTH
                self.world_height = self.max_field_height + 2 * FIELD_MARGIN_HEIGHT

            scenario = data.get("scenario", {})

            # Load unit data
            positions = scenario.get("positions", [])
            unit_ids = scenario.get("unit_ids", [])
            teams = scenario.get("teams", [])
            rotations = scenario.get("rotations", [])

            # Initialize grid
            self.grid = np.zeros((self.grid_height, self.grid_width), dtype=int)
            self.team_grid = np.zeros((self.grid_height, self.grid_width), dtype=int)
            self.rotation_grid = np.zeros((self.grid_height, self.grid_width), dtype=float)

            for pos, unit_id_list, team_list, rot_list in zip(
                positions, unit_ids, teams, rotations
            ):
                unit_id = int(unit_id_list[0]) + 1  # Stored as 0-indexed in JSON, so +1
                team = int(team_list[0])
                rotation = float(rot_list[0])

                # Convert world coordinates to grid coordinates
                world_x, world_y = pos

                # Mammoth stored as center coordinates, convert to top-left
                if unit_id == UnitID.Mammoth:
                    # Center coordinates -> top-left coordinates
                    # If center is (world_x, world_y), top-left is (world_x - 0.5*spacing, world_y - 0.5*spacing)
                    top_left_world_x = world_x - 0.5 * self.actual_unit_spacing_x
                    top_left_world_y = world_y - 0.5 * self.actual_unit_spacing_y
                    j = int(
                        round(
                            top_left_world_x / self.actual_unit_spacing_x
                            + self.grid_width / 2
                            - 0.5
                        )
                    )
                    i = int(
                        round(
                            top_left_world_y / self.actual_unit_spacing_y
                            + self.grid_height / 2
                            - 0.5
                        )
                    )
                else:
                    # Normal unit
                    j = int(round(world_x / self.actual_unit_spacing_x + self.grid_width / 2 - 0.5))
                    i = int(
                        round(world_y / self.actual_unit_spacing_y + self.grid_height / 2 - 0.5)
                    )

                if 0 <= i < self.grid_height and 0 <= j < self.grid_width:
                    # Place Mammoth as 2x2
                    if unit_id == UnitID.Mammoth:
                        if i + 1 < self.grid_height and j + 1 < self.grid_width:
                            self.grid[i, j] = unit_id
                            self.grid[i + 1, j] = MAMMOTH_OCCUPIED
                            self.grid[i, j + 1] = MAMMOTH_OCCUPIED
                            self.grid[i + 1, j + 1] = MAMMOTH_OCCUPIED
                            self.team_grid[i, j] = team
                            self.team_grid[i + 1, j] = team
                            self.team_grid[i, j + 1] = team
                            self.team_grid[i + 1, j + 1] = team
                            self.rotation_grid[i, j] = rotation
                    else:
                        self.grid[i, j] = unit_id
                        self.team_grid[i, j] = team
                        self.rotation_grid[i, j] = rotation

            # Load zone data
            self.zones = []
            zone_scenario = data.get("zone_scenario", {})
            zone_types = zone_scenario.get("zone_type", [])
            zone_positions = zone_scenario.get("position", [])
            zone_axes = zone_scenario.get("axes", [])
            zone_effects = zone_scenario.get("effect_value", [])

            for z_type, z_pos, z_axes, z_effect in zip(
                zone_types, zone_positions, zone_axes, zone_effects
            ):
                zone = {
                    "type": int(z_type[0]),
                    "position": z_pos,
                    "axes": z_axes,
                    "effect_value": float(z_effect[0]),
                }
                self.zones.append(zone)

            self.show_load_dialog = False
            self.save_state()  # Save to history
        except Exception as e:
            pass

    def run(self):
        """Main loop"""
        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False

                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:  # Left click
                        mouse_x, mouse_y = event.pos

                        # Check zone info box clicks first (if in zone mode and zone is selected/hovered)
                        if self.edit_mode == "zone" and (
                            self.selected_zone_info or self.hovered_zone_info
                        ):
                            if self.handle_zone_info_click(event.pos):
                                continue

                        # Check if dragging toolbar
                        toolbar_title_bar = pygame.Rect(
                            self.toolbar_x, self.toolbar_y, self.toolbar_width, 25
                        )
                        if toolbar_title_bar.collidepoint(mouse_x, mouse_y):
                            self.toolbar_dragging = True
                            self.toolbar_drag_offset = (
                                mouse_x - self.toolbar_x,
                                mouse_y - self.toolbar_y,
                            )

                        # Check if dragging unit panel (only in unit mode)
                        elif self.edit_mode == "unit":
                            unit_title_bar = pygame.Rect(
                                self.unit_panel_x, self.unit_panel_y, self.unit_panel_width, 30
                            )
                            if unit_title_bar.collidepoint(mouse_x, mouse_y):
                                self.unit_panel_dragging = True
                                self.unit_panel_drag_offset = (
                                    mouse_x - self.unit_panel_x,
                                    mouse_y - self.unit_panel_y,
                                )

                        # Check if dragging zone panel (only in zone mode)
                        elif self.edit_mode == "zone":
                            zone_title_bar = pygame.Rect(
                                self.zone_panel_x, self.zone_panel_y, self.zone_panel_width, 30
                            )
                            if zone_title_bar.collidepoint(mouse_x, mouse_y):
                                self.zone_panel_dragging = True
                                self.zone_panel_drag_offset = (
                                    mouse_x - self.zone_panel_x,
                                    mouse_y - self.zone_panel_y,
                                )

                        # Check UI first
                        if not self.handle_ui_click(event.pos):
                            # Then check battlefield (only if dialog not open)
                            if not self.show_new_scenario_dialog:
                                self.handle_battlefield_click(event.pos, is_drag_start=True)
                    elif event.button == 3:  # Right click
                        # Right click to delete
                        if not self.show_new_scenario_dialog:
                            self.handle_battlefield_click(event.pos, is_right_click=True)
                    elif event.button == 4:  # Wheel up (zoom in) - legacy mouse
                        if not self.show_new_scenario_dialog and not self.show_load_dialog:
                            self.handle_zoom(1)
                    elif event.button == 5:  # Wheel down (zoom out) - legacy mouse
                        if not self.show_new_scenario_dialog and not self.show_load_dialog:
                            self.handle_zoom(-1)

                elif event.type == pygame.MOUSEBUTTONUP:
                    if event.button == 1:
                        # Stop dragging
                        self.toolbar_dragging = False
                        self.unit_panel_dragging = False
                        self.zone_panel_dragging = False

                        if self.dragging_zone:
                            self.finish_zone_drag(event.pos)

                elif event.type == pygame.MOUSEMOTION:
                    # Handle dragging
                    if self.toolbar_dragging:
                        mouse_x, mouse_y = event.pos
                        self.toolbar_x = mouse_x - self.toolbar_drag_offset[0]
                        self.toolbar_y = mouse_y - self.toolbar_drag_offset[1]
                    elif self.unit_panel_dragging:
                        mouse_x, mouse_y = event.pos
                        self.unit_panel_x = mouse_x - self.unit_panel_drag_offset[0]
                        self.unit_panel_y = mouse_y - self.unit_panel_drag_offset[1]
                        # Synchronize zone palette to same position
                        self.zone_panel_x = self.unit_panel_x
                        self.zone_panel_y = self.unit_panel_y
                    elif self.zone_panel_dragging:
                        mouse_x, mouse_y = event.pos
                        self.zone_panel_x = mouse_x - self.zone_panel_drag_offset[0]
                        self.zone_panel_y = mouse_y - self.zone_panel_drag_offset[1]
                        # Synchronize unit palette to same position
                        self.unit_panel_x = self.zone_panel_x
                        self.unit_panel_y = self.zone_panel_y
                    elif self.dragging_zone:
                        self.update_zone_drag(event.pos)

                elif event.type == pygame.MOUSEWHEEL:
                    # Check if dialog is open first
<<<<<<< HEAD
                    if (
                        self.show_load_dialog
                        or self.show_new_scenario_dialog
                        or self.show_save_dialog
                    ):
=======
                    if self.show_load_dialog or self.show_new_scenario_dialog:
>>>>>>> a4fed485df2ae65fcca9dbd5217824b01ea2cb0e
                        # Handle scroll only when load dialog is open
                        if self.show_load_dialog:
                            mouse_x, mouse_y = pygame.mouse.get_pos()
                            dialog_width = 450
                            dialog_height = 550
                            dialog_x = (SCREEN_WIDTH - dialog_width) // 2
                            dialog_y = (SCREEN_HEIGHT - dialog_height) // 2

                            # Scroll only inside dialog
                            if (
                                dialog_x <= mouse_x <= dialog_x + dialog_width
                                and dialog_y <= mouse_y <= dialog_y + dialog_height
                            ):
<<<<<<< HEAD
                                if self.search_query:
                                    # Search mode - flat list
                                    max_scroll = max(0, len(self.filtered_scenarios) - 10)
                                else:
                                    # Folder structure mode - count total visible items
                                    total_items = len(self.available_scenarios)
                                    for folder_name in self.available_scenarios:
                                        if self.folder_expanded.get(folder_name, False):
                                            total_items += len(
                                                self.available_scenarios[folder_name]
                                            )
                                    max_scroll = max(0, total_items - 10)

=======
                                scenarios_to_show = (
                                    self.filtered_scenarios
                                    if self.search_query
                                    else self.available_scenarios
                                )
                                max_scroll = max(
                                    0, len(scenarios_to_show) - 10
                                )  # Show 10 at a time
>>>>>>> a4fed485df2ae65fcca9dbd5217824b01ea2cb0e
                                self.load_scroll_offset = max(
                                    0, min(max_scroll, self.load_scroll_offset - event.y)
                                )
                        # Block background zoom completely when dialog is open
                    else:
                        # Normal zoom only when all dialogs are closed
                        self.handle_zoom(event.y)

                elif event.type == pygame.KEYDOWN:
                    if self.editing_zone and self.zone_edit_field:
                        # Handle zone editing input
                        if event.key == pygame.K_RETURN or event.key == pygame.K_KP_ENTER:
                            # Apply the edit immediately
                            zone_info = (
                                self.selected_zone_info
                                if self.selected_zone_info
                                else self.hovered_zone_info
                            )
                            if self.zone_edit_input and zone_info:
                                try:
                                    zone_idx = zone_info["index"] - 1
                                    if 0 <= zone_idx < len(self.zones):
                                        zone = self.zones[zone_idx]
                                        new_value = float(self.zone_edit_input)

                                        if self.zone_edit_field == "pos_x":
                                            zone["position"][0] = new_value
                                        elif self.zone_edit_field == "pos_y":
                                            zone["position"][1] = new_value
                                        elif self.zone_edit_field == "axis_x":
                                            if new_value > 0:
                                                zone["axes"][0] = new_value
                                        elif self.zone_edit_field == "axis_y":
                                            if new_value > 0:
                                                zone["axes"][1] = new_value
                                        elif self.zone_edit_field == "effect":
                                            zone["effect_value"] = new_value

                                        # Update zone info immediately
                                        self.apply_zone_edits()
                                        self.save_state()
                                except ValueError:
                                    pass  # Invalid input, ignore
                            self.editing_zone = False
                            self.zone_edit_field = None
                            self.zone_edit_input = ""
                        elif event.key == pygame.K_ESCAPE:
                            # Cancel edit
                            self.editing_zone = False
                            self.zone_edit_field = None
                            self.zone_edit_input = ""
                        elif event.key == pygame.K_BACKSPACE:
                            self.zone_edit_input = self.zone_edit_input[:-1]
                        else:
                            # Add character if valid
                            if event.unicode in "0123456789.-":
                                self.zone_edit_input += event.unicode
                    elif self.show_new_scenario_dialog and self.active_input_field:
                        # Handle new scenario dialog input
                        if event.key == pygame.K_RETURN or event.key == pygame.K_KP_ENTER:
                            # Move to next field or create
                            fields = [
                                "scenario_name",
                                "max_height",
                                "max_width",
                                "margin_w",
                                "margin_h",
                                "unit_spacing",
                            ]
                            current_idx = fields.index(self.active_input_field)
                            if current_idx < len(fields) - 1:
                                self.active_input_field = fields[current_idx + 1]
                            else:
                                self.create_new_scenario()
                        elif event.key == pygame.K_TAB:
                            # Move to next field
                            fields = [
                                "scenario_name",
                                "max_height",
                                "max_width",
                                "margin_w",
                                "margin_h",
                                "unit_spacing",
                            ]
                            current_idx = fields.index(self.active_input_field)
                            self.active_input_field = fields[(current_idx + 1) % len(fields)]
                        elif event.key == pygame.K_ESCAPE:
                            self.show_new_scenario_dialog = False
                            self.active_input_field = None
                        elif event.key == pygame.K_BACKSPACE:
                            self.new_scenario_inputs[self.active_input_field] = (
                                self.new_scenario_inputs[self.active_input_field][:-1]
                            )
                        else:
                            if event.unicode in "0123456789.":
                                self.new_scenario_inputs[self.active_input_field] += event.unicode
<<<<<<< HEAD
                    elif self.show_save_dialog and self.save_name_input_active:
                        # Handle scenario name input in save dialog
                        if event.key == pygame.K_RETURN or event.key == pygame.K_KP_ENTER:
                            self.save_name_input_active = False
                        elif event.key == pygame.K_ESCAPE:
                            self.save_name_input_active = False
                        elif event.key == pygame.K_BACKSPACE:
                            self.save_scenario_name = self.save_scenario_name[:-1]
                        else:
                            # Add character (alphanumeric, underscore, hyphen)
                            if event.unicode.isprintable() and event.unicode not in '/\\:*?"<>|':
                                self.save_scenario_name += event.unicode
=======
>>>>>>> a4fed485df2ae65fcca9dbd5217824b01ea2cb0e
                    elif self.show_load_dialog and self.search_active:
                        # Handle search input in load dialog
                        if event.key == pygame.K_RETURN or event.key == pygame.K_KP_ENTER:
                            self.search_active = False
                        elif event.key == pygame.K_ESCAPE:
                            self.search_query = ""
                            self.search_active = False
                            self.filtered_scenarios = []
                            self.load_scroll_offset = 0
                        elif event.key == pygame.K_BACKSPACE:
                            self.search_query = self.search_query[:-1]
                            # Update filtered list
                            if self.search_query:
<<<<<<< HEAD
                                self.filtered_scenarios = []
                                for folder, files in self.available_scenarios.items():
                                    for filename in files:
                                        if self.search_query.lower() in filename.lower():
                                            self.filtered_scenarios.append((folder, filename))
=======
                                self.filtered_scenarios = [
                                    f
                                    for f in self.available_scenarios
                                    if self.search_query.lower() in f.lower()
                                ]
>>>>>>> a4fed485df2ae65fcca9dbd5217824b01ea2cb0e
                            else:
                                self.filtered_scenarios = []
                            self.load_scroll_offset = 0
                            self.selected_scenario_index = None
                        else:
                            # Add character
                            if event.unicode.isprintable():
                                self.search_query += event.unicode
                                # Update filtered list
<<<<<<< HEAD
                                self.filtered_scenarios = []
                                for folder, files in self.available_scenarios.items():
                                    for filename in files:
                                        if self.search_query.lower() in filename.lower():
                                            self.filtered_scenarios.append((folder, filename))
=======
                                self.filtered_scenarios = [
                                    f
                                    for f in self.available_scenarios
                                    if self.search_query.lower() in f.lower()
                                ]
>>>>>>> a4fed485df2ae65fcca9dbd5217824b01ea2cb0e
                                self.load_scroll_offset = 0
                                self.selected_scenario_index = None
                    elif self.text_input_active:
                        if event.key == pygame.K_RETURN or event.key == pygame.K_KP_ENTER:
                            # Apply the value
                            if self.text_input_value:
                                try:
                                    new_value = float(self.text_input_value)
                                    # Validate range
                                    if self.selected_zone_type == 1:  # Lava
                                        if 0 <= new_value <= 100:
                                            self.zone_effect_value = new_value
                                    else:  # Swamp
                                        if 0 <= new_value <= 1:
                                            self.zone_effect_value = new_value
                                except ValueError:
                                    pass  # Invalid input, ignore
                            self.text_input_active = False
                            self.text_input_value = ""
                        elif event.key == pygame.K_ESCAPE:
                            # Cancel input
                            self.text_input_active = False
                            self.text_input_value = ""
                        elif event.key == pygame.K_BACKSPACE:
                            # Delete last character
                            self.text_input_value = self.text_input_value[:-1]
                        else:
                            # Add character if it's a valid number character
                            if event.unicode in "0123456789.":
                                self.text_input_value += event.unicode
                    else:
                        # Reset camera (R key)
                        if event.key == pygame.K_r:
                            self.camera_offset_x = 0
                            self.camera_offset_y = 0

                        # Undo/Redo (Ctrl+Z / Ctrl+Y)
                        elif event.key == pygame.K_z and pygame.key.get_mods() & pygame.KMOD_CTRL:
                            self.undo()
                        elif event.key == pygame.K_y and pygame.key.get_mods() & pygame.KMOD_CTRL:
                            self.redo()

            # Check keyboard state (continuous application when held down)
            if (
                not self.show_new_scenario_dialog
                and not self.text_input_active
                and not self.editing_zone
            ):
                keys = pygame.key.get_pressed()
                # Camera movement (WASD keys)
                if keys[pygame.K_w]:
                    self.camera_offset_y += self.camera_move_speed
                if keys[pygame.K_s]:
                    self.camera_offset_y -= self.camera_move_speed
                if keys[pygame.K_a]:
                    self.camera_offset_x += self.camera_move_speed
                if keys[pygame.K_d]:
                    self.camera_offset_x -= self.camera_move_speed

                # Unit rotation (Q/E keys) - rotate in 15 degree increments (with cooldown)
                if self.rotation_cooldown > 0:
                    self.rotation_cooldown -= 1
                else:
                    if keys[pygame.K_q]:
                        self.rotate_unit_at_mouse(-np.pi / 12)  # 15° counterclockwise
                        self.rotation_cooldown = self.rotation_cooldown_frames
                    elif keys[pygame.K_e]:
                        self.rotate_unit_at_mouse(np.pi / 12)  # 15° clockwise
                        self.rotation_cooldown = self.rotation_cooldown_frames

            # Update hovered zone info if in zone mode
            if self.edit_mode == "zone" and not self.show_new_scenario_dialog:
                mouse_pos = pygame.mouse.get_pos()
                self.update_hovered_zone(mouse_pos)

            # Draw screen
            self.screen.fill((60, 60, 60))  # Dark gray background

            # Draw battlefield (first)
            self.draw_battlefield_background()

            # Draw UI (on top)
            self.draw_ui()

            # Draw palette (topmost) - display based on current mode
            self.draw_unit_palette()
            self.draw_zone_palette()

            # Draw hovered zone info if in zone mode
            if self.edit_mode == "zone":
                self.draw_hovered_zone_info()

            # New scenario dialog (very top)
            if self.show_new_scenario_dialog:
                self.draw_new_scenario_dialog()

            pygame.display.flip()
            self.clock.tick(60)

        pygame.quit()


if __name__ == "__main__":
    editor = MapEditor()
    editor.run()
