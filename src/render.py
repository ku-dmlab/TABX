from src.maenv.environments.base_maenv import BaseMAEnv
from src.maenv.tabs.tabs_battle_simulator.tabs_battle_simulator import (
    DefaultUnit,
    UnitStatus,
    GameManager,
    AttackType,
)
from src.maenv.physics import Transform, RigidBody, CircleCollider, physics_step
from easydict import EasyDict
from typing import Dict
import jax.numpy as jnp


import pygame
import numpy as np
import math
from src.maenv.tabs.tabs_battle_simulator.tabs_battle_simulator import UnitAction


class PygameRenderer:
    """실시간으로 게임 객체들을 렌더링하는 pygame 렌더러"""

    def __init__(
        self,
        width=1920,
        height=1080,
        fps=60,
        world_scale=20,
        env=None,
        scenario=None,
        reset_fn=None,
    ):
        """
        초기화

        Args:
            width: 창 너비
            height: 창 높이
            fps: 프레임 레이트
            world_scale: 게임 월드 좌표를 픽셀로 변환하는 스케일
            env: 환경 객체 (env.reset()을 위해 필요)
            scenario: 시나리오 객체 (reset할 때 사용)
            reset_fn: 리셋 함수 (env.reset 또는 jit된 버전)
        """
        pygame.init()

        self.width = width
        self.height = height
        self.fps = fps
        self.world_scale = world_scale

        # Environment components for reset functionality
        self.env = env
        self.scenario = scenario
        self.reset_fn = reset_fn
        self.reset_requested = False

        # Scenario selection
        self.available_scenarios = {}
        self.scenario_names = []
        self.selected_scenario_index = 0
        self._preload_scenarios()

        # Policy selection
        self.policy_names = ["Heuristic", "Idle"]
        self.selected_policy_index = 0  # 기본적으로 Heuristic 정책 사용

        # 화면 설정
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("TABS Battle Renderer")

        # 클럭 설정
        self.clock = pygame.time.Clock()

        # 색상 정의
        self.colors = {
            "background": (50, 50, 50),
            "team_0": (255, 100, 100),  # 빨간색 팀
            "team_1": (100, 100, 255),  # 파란색 팀
            "health_bg": (200, 200, 200),  # 체력바 배경
            "health_fg": (100, 255, 100),  # 체력바 전경
            "cooldown_bg": (150, 150, 150),  # 쿨다운바 배경
            "cooldown_fg": (255, 255, 100),  # 쿨다운바 전경 (노란색)
            "cooldown_ready": (100, 255, 100),  # 공격 준비됨 (초록색)
            "attack_range": (255, 255, 0, 50),  # 공격 범위 (반투명 노란색)
            "sight_range": (0, 255, 255, 30),  # 시야 범위 (반투명 청록색)
            "attacking": (255, 255, 255, 100),  # 공격 중 표시
            "selected": (255, 255, 255),  # 선택된 유닛 표시
            "target_matrix_bg": (40, 40, 40),  # target matrix 배경
            "target_true": (255, 255, 100),  # target=True인 셀
            "target_false": (80, 80, 80),  # target=False인 셀
            "target_text": (255, 255, 255),  # target matrix 텍스트
            "visible_matrix_bg": (30, 30, 50),  # visible matrix 배경
            "visible_true": (100, 255, 150),  # visible=True인 셀 (초록색)
            "visible_false": (60, 60, 90),  # visible=False인 셀
            "visible_text": (255, 255, 255),  # visible matrix 텍스트
            "visible_selected_true": (
                150,
                255,
                100,
            ),  # 선택된 유닛의 visible=True 셀 (더 밝은 초록색)
            "visible_selected_false": (80, 100, 60),  # 선택된 유닛의 visible=False 셀
            "visible_dimmed_true": (50, 120, 70),  # 선택되지 않은 유닛의 visible=True 셀 (흐림)
            "visible_dimmed_false": (30, 30, 45),  # 선택되지 않은 유닛의 visible=False 셀 (흐림)
        }

        # 폰트 설정
        self.font = pygame.font.Font(None, 24)
        self.small_font = pygame.font.Font(None, 16)

        # 카메라 설정
        self.camera_x = 0
        self.camera_y = 0
        self.zoom = 1.0

        # 유닛 선택 및 컨트롤 관련
        self.selected_unit = None
        self.mouse_pos = (0, 0)
        self.user_controlled_actions = {}  # 유저가 컨트롤하는 유닛들의 액션 저장
        self.selected_unit_obs = None  # 선택된 유닛의 obs 배열 저장
        self.obs_scroll_offset = 0  # obs 배열 스크롤 오프셋

        # UI 토글 설정
        self.ui_panel = {
            "show_sight_range": False,
            "show_attack_range": False,
            "show_visible_matrix": False,
            "show_distance_matrix": False,
            "show_unit_info": False,
            "show_grid": False,
            "show_obs_array": False,  # obs 배열 표시 옵션 추가 (기본적으로 비활성화)
        }

        # UI 패널 설정
        self.panel_width = 200
        self.panel_x = self.width - self.panel_width
        self.checkbox_size = 15
        self.checkbox_spacing = 25
        self.panel_visible = True  # 패널 표시 상태

        # 패널 토글 버튼 설정
        self.toggle_button_width = 30
        self.toggle_button_height = 20

        # 게임 화면 영역 조정 (패널 상태에 따라)
        self.update_game_area()

        self.running = True

    def _preload_scenarios(self):
        """모든 시나리오를 미리 로드"""
        try:
            from src.maenv.tabs.scenarios import (
                get_scenario_name_list,
                generate_scenario,
                default_tabs_conf,
            )

            self.scenario_names = get_scenario_name_list()

            # 현재 시나리오가 목록에 있으면 해당 인덱스로 설정
            if self.scenario:
                # 현재 scenario의 이름을 추정 (scenario 객체에서 직접 가져올 수 없으므로)
                # default_tabs_conf에서 scenario_name을 확인
                current_scenario_name = getattr(self.scenario, "scenario_name", None)
                if current_scenario_name and current_scenario_name in self.scenario_names:
                    self.selected_scenario_index = self.scenario_names.index(current_scenario_name)
                else:
                    # 첫 번째 시나리오를 기본으로 설정
                    self.selected_scenario_index = 0

            # 모든 시나리오 생성
            for scenario_name in self.scenario_names:
                conf = default_tabs_conf.replace(scenario_name=scenario_name)
                self.available_scenarios[scenario_name] = generate_scenario(conf)

        except Exception as e:
            print(f"Failed to preload scenarios: {e}")
            self.scenario_names = []
            self.available_scenarios = {}

    def update_game_area(self):
        """패널 상태에 따라 게임 영역 업데이트"""
        if self.panel_visible:
            self.game_width = self.width - self.panel_width
        else:
            self.game_width = self.width

    def world_to_screen(self, world_pos):
        """월드 좌표를 스크린 좌표로 변환 (수학적 좌표계: y증가 = 위쪽)"""
        world_x, world_y = world_pos
        screen_x = (world_x - self.camera_x) * self.world_scale * self.zoom + self.game_width // 2
        screen_y = self.height // 2 - (world_y - self.camera_y) * self.world_scale * self.zoom
        return int(screen_x), int(screen_y)

    def screen_to_world(self, screen_pos):
        """스크린 좌표를 월드 좌표로 변환 (수학적 좌표계: y증가 = 위쪽)"""
        screen_x, screen_y = screen_pos
        world_x = (screen_x - self.game_width // 2) / (self.world_scale * self.zoom) + self.camera_x
        world_y = (self.height // 2 - screen_y) / (self.world_scale * self.zoom) + self.camera_y
        return world_x, world_y

    def find_unit_at_position(self, world_pos, objects):
        """주어진 월드 좌표에서 가장 가까운 유닛을 찾기"""
        world_x, world_y = world_pos
        closest_unit = None
        closest_distance = float("inf")

        for obj_name, obj in objects.items():
            if "unit" in obj_name.lower() and hasattr(obj, "transform"):
                try:
                    if hasattr(obj.transform.position, "__array__"):
                        unit_pos = np.array(obj.transform.position)
                    else:
                        unit_pos = obj.transform.position

                    if hasattr(obj.collider.radius, "__array__"):
                        radius = float(np.array(obj.collider.radius))
                    else:
                        radius = float(obj.collider.radius)

                    distance = math.sqrt(
                        (world_x - unit_pos[0]) ** 2 + (world_y - unit_pos[1]) ** 2
                    )

                    # 유닛의 반지름 내에 클릭이 있고, 가장 가까운 유닛인 경우
                    if distance <= radius and distance < closest_distance:
                        closest_distance = distance
                        closest_unit = obj_name

                except Exception:
                    continue

        return closest_unit

    def calculate_rotation_to_mouse(self, unit_pos):
        """유닛 위치에서 마우스 방향으로의 회전각도 계산 (라디안)"""
        mouse_world_pos = self.screen_to_world(self.mouse_pos)
        dx = mouse_world_pos[0] - unit_pos[0]
        dy = mouse_world_pos[1] - unit_pos[1]
        target_angle = math.atan2(dy, dx)
        return target_angle

    def handle_events(self, objects):
        """이벤트 처리"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False
                elif event.key == pygame.K_SPACE:
                    # 스페이스바로 카메라 리셋
                    self.camera_x = 0
                    self.camera_y = 0
                    self.zoom = 1.0
                elif event.key == pygame.K_TAB:
                    # TAB키로 패널 토글
                    self.panel_visible = not self.panel_visible
                    self.update_game_area()
                elif event.key == pygame.K_r:
                    # R키로 환경 리셋
                    if self.reset_fn and self.scenario:
                        try:
                            # 환경 리셋 요청 플래그 설정
                            self.reset_requested = True
                        except Exception as e:
                            print(f"Failed to reset environment: {e}")
                    # obs 스크롤도 함께 리셋
                    self.obs_scroll_offset = 0
            elif event.type == pygame.MOUSEWHEEL:
                # obs 배열이 표시되고 있을 때는 스크롤로 obs 배열 탐색
                if self.ui_panel["show_obs_array"] and self.selected_unit_obs is not None:
                    scroll_speed = 3  # 한 번에 스크롤할 줄 수
                    if event.y > 0:
                        self.obs_scroll_offset = max(0, self.obs_scroll_offset - scroll_speed)
                    else:
                        self.obs_scroll_offset += scroll_speed
                else:
                    # 일반적인 경우 줌
                    zoom_factor = 1.1
                    if event.y > 0:
                        self.zoom *= zoom_factor
                    else:
                        self.zoom /= zoom_factor
                    self.zoom = max(0.1, min(5.0, self.zoom))
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # 왼쪽 마우스 버튼
                    # 시나리오 선택기 클릭 확인 (최우선)
                    if self.handle_scenario_selector_click(event.pos):
                        pass
                    # 정책 선택기 클릭 확인
                    elif self.handle_policy_selector_click(event.pos):
                        pass
                    # 토글 버튼 클릭 확인
                    elif self.handle_toggle_button_click(event.pos):
                        pass
                    # UI 패널 클릭 확인 (패널이 보일 때만)
                    elif (
                        self.panel_visible
                        and hasattr(self, "checkbox_rects")
                        and self.handle_ui_click(event.pos)
                    ):
                        # UI가 클릭되었으면 다른 처리는 하지 않음
                        pass
                    else:
                        # UI가 클릭되지 않았으면 기존 로직 실행
                        world_pos = self.screen_to_world(event.pos)
                        clicked_unit = self.find_unit_at_position(world_pos, objects)

                        if clicked_unit:
                            # 유닛을 클릭한 경우 선택
                            if self.selected_unit != clicked_unit:  # 다른 유닛을 선택한 경우
                                self.selected_unit = clicked_unit
                                self.selected_unit_obs = None  # 새로운 유닛이므로 obs 초기화
                                self.obs_scroll_offset = 0  # 스크롤 오프셋 리셋

                        else:
                            # 땅을 클릭한 경우 선택 해제
                            self.selected_unit = None
                            self.selected_unit_obs = None
                            self.obs_scroll_offset = 0  # 스크롤 오프셋도 리셋

            elif event.type == pygame.MOUSEMOTION:
                self.mouse_pos = event.pos

        # 선택된 유닛이 있을 때만 키보드 컨트롤 처리
        if self.selected_unit:
            keys = pygame.key.get_pressed()
            unit = objects.get(self.selected_unit)

            if unit and hasattr(unit.transform, "position"):
                # 유닛 위치 가져오기
                if hasattr(unit.transform.position, "__array__"):
                    unit_pos = np.array(unit.transform.position)
                else:
                    unit_pos = unit.transform.position

                # 마우스 방향으로의 회전각도 계산
                target_rotation = self.calculate_rotation_to_mouse(unit_pos)
                current_rotation = (
                    float(np.array(unit.transform.rotation))
                    if hasattr(unit.transform.rotation, "__array__")
                    else float(unit.transform.rotation)
                )

                # 회전 각도 차이 계산 (최단 거리로)
                angle_diff = target_rotation - current_rotation
                while angle_diff > math.pi:
                    angle_diff -= 2 * math.pi
                while angle_diff < -math.pi:
                    angle_diff += 2 * math.pi

                # 회전 속도 제한 (너무 빠르게 회전하지 않도록)
                max_rotation_speed = 0.2
                rotation_amount = max(-max_rotation_speed, min(max_rotation_speed, angle_diff))

                # 키보드 입력에 따른 액션 결정
                action = UnitAction.IDLE
                if keys[pygame.K_w] or keys[pygame.K_UP]:
                    action = UnitAction.UP
                elif keys[pygame.K_s] or keys[pygame.K_DOWN]:
                    action = UnitAction.DOWN
                elif keys[pygame.K_a] or keys[pygame.K_LEFT]:
                    action = UnitAction.LEFT
                elif keys[pygame.K_d] or keys[pygame.K_RIGHT]:
                    action = UnitAction.RIGHT
                elif keys[pygame.K_LCTRL] or keys[pygame.K_RCTRL]:
                    action = UnitAction.ATTACK

                # 유저 컨트롤 액션 저장
                self.user_controlled_actions[self.selected_unit] = jnp.array(
                    [rotation_amount, action]
                )
        else:
            # 선택된 유닛이 없을 때는 기존처럼 카메라 이동
            keys = pygame.key.get_pressed()
            move_speed = 0.5 / self.zoom
            if keys[pygame.K_w] or keys[pygame.K_UP]:
                self.camera_y += move_speed
            if keys[pygame.K_s] or keys[pygame.K_DOWN]:
                self.camera_y -= move_speed
            if keys[pygame.K_a] or keys[pygame.K_LEFT]:
                self.camera_x -= move_speed
            if keys[pygame.K_d] or keys[pygame.K_RIGHT]:
                self.camera_x += move_speed

    def draw_unit(self, unit_name, unit, show_ranges=None, alpha=255):
        """개별 유닛을 그리기"""
        try:
            # JAX 배열에서 numpy 배열로 변환
            if hasattr(unit.transform.position, "__array__"):
                pos = np.array(unit.transform.position)
            else:
                pos = unit.transform.position

            if hasattr(unit.transform.rotation, "__array__"):
                rotation = float(np.array(unit.transform.rotation))
            else:
                rotation = float(unit.transform.rotation)

            if hasattr(unit.collider.radius, "__array__"):
                radius = float(np.array(unit.collider.radius))
            else:
                radius = float(unit.collider.radius)

            if hasattr(unit.team, "__array__"):
                team = int(np.array(unit.team))
            else:
                team = int(unit.team)

            if hasattr(unit.status.health, "__array__"):
                health = float(np.array(unit.status.health))
            else:
                health = float(unit.status.health)

            if hasattr(unit.status.attack_range, "__array__"):
                attack_range = float(np.array(unit.status.attack_range))
            else:
                attack_range = float(unit.status.attack_range)

            # if hasattr(unit.status.sight_radius, "__array__"):
            #     sight_radius = float(np.array(unit.status.sight_radius))
            # else:
            #     sight_radius = float(unit.status.sight_radius)

            if hasattr(unit.status.sight_angle, "__array__"):
                sight_angle = float(np.array(unit.status.sight_angle))
            else:
                sight_angle = float(unit.status.sight_angle)

            # 공격 중인지 확인
            attacking = False
            if hasattr(unit, "attacking"):
                if hasattr(unit.attacking, "__array__"):
                    attacking = bool(np.array(unit.attacking))
                else:
                    attacking = bool(unit.attacking)

            screen_pos = self.world_to_screen(pos)
            screen_radius = int(radius * self.world_scale * self.zoom)

            # 화면에 보이는지 확인
            if (
                screen_pos[0] < -screen_radius
                or screen_pos[0] > self.width + screen_radius
                or screen_pos[1] < -screen_radius
                or screen_pos[1] > self.height + screen_radius
            ):
                return

            # 유닛 몸체
            team_color = self.colors.get(f"team_{team}", (128, 128, 128))

            if alpha < 255:
                # 투명도가 적용된 경우, 별도 surface에 그린 후 알파 블렌딩
                circle_surface = pygame.Surface(
                    (screen_radius * 2 + 10, screen_radius * 2 + 10), pygame.SRCALPHA
                )
                circle_surface.fill((0, 0, 0, 0))  # 투명 배경
                circle_center = (screen_radius + 5, screen_radius + 5)
                pygame.draw.circle(
                    circle_surface, (*team_color, alpha), circle_center, max(1, screen_radius)
                )
                self.screen.blit(
                    circle_surface,
                    (screen_pos[0] - screen_radius - 5, screen_pos[1] - screen_radius - 5),
                )
            else:
                # 일반적인 경우
                pygame.draw.circle(self.screen, team_color, screen_pos, max(1, screen_radius))

            # 범위 표시 (옵션) - 유닛 위에 그려짐
            if show_ranges and self.zoom > 0.3:
                # 투명도에 따라 범위도 조정
                range_alpha = alpha if alpha < 255 else None

                # 시야 범위 (부채꼴)
                if isinstance(show_ranges, dict) and show_ranges.get("sight", True):
                    self.draw_fan_sight_range(screen_pos, rotation, sight_angle, range_alpha)
                elif show_ranges is True:  # 이전 버전과의 호환성
                    self.draw_fan_sight_range(screen_pos, rotation, sight_angle, range_alpha)

                # 공격 범위 (직사각형) - TABS 스타일
                if isinstance(show_ranges, dict) and show_ranges.get("attack", True):
                    self.draw_rectangular_attack_range(
                        screen_pos, rotation, attack_range, radius, range_alpha, sight_angle
                    )
                elif show_ranges is True:  # 이전 버전과의 호환성
                    self.draw_rectangular_attack_range(
                        screen_pos, rotation, attack_range, radius, range_alpha, sight_angle
                    )

            # 선택된 유닛 표시
            if unit_name == self.selected_unit:
                pygame.draw.circle(
                    self.screen, self.colors["selected"], screen_pos, max(1, screen_radius + 3), 3
                )

            # 방향 표시 (회전 정보가 있을 때) - 수학적 좌표계
            if screen_radius > 3:
                direction_end = (
                    screen_pos[0] + int(math.cos(rotation) * screen_radius * 0.8),
                    screen_pos[1] - int(math.sin(rotation) * screen_radius * 0.8),
                )

                if alpha < 255:
                    # 투명한 유닛의 방향 표시도 흐리게
                    direction_surface = pygame.Surface(
                        (
                            abs(direction_end[0] - screen_pos[0]) + 10,
                            abs(direction_end[1] - screen_pos[1]) + 10,
                        ),
                        pygame.SRCALPHA,
                    )
                    direction_surface.fill((0, 0, 0, 0))
                    local_start = (5, 5)
                    local_end = (
                        direction_end[0] - screen_pos[0] + 5,
                        direction_end[1] - screen_pos[1] + 5,
                    )
                    pygame.draw.line(
                        direction_surface, (255, 255, 255, alpha), local_start, local_end, 2
                    )
                    self.screen.blit(
                        direction_surface,
                        (
                            min(screen_pos[0], direction_end[0]) - 5,
                            min(screen_pos[1], direction_end[1]) - 5,
                        ),
                    )
                else:
                    pygame.draw.line(self.screen, (255, 255, 255), screen_pos, direction_end, 2)

            # 공격 중 표시
            if attacking and screen_radius > 2:
                attack_surface = pygame.Surface(
                    (screen_radius * 3, screen_radius * 3), pygame.SRCALPHA
                )
                pygame.draw.circle(
                    attack_surface,
                    self.colors["attacking"],
                    (screen_radius * 1.5, screen_radius * 1.5),
                    screen_radius * 1.5,
                )
                self.screen.blit(
                    attack_surface,
                    (screen_pos[0] - screen_radius * 1.5, screen_pos[1] - screen_radius * 1.5),
                )

            # 체력바 (줌이 충분할 때만)
            if self.zoom > 0.5 and screen_radius > 5:
                health_bar_width = screen_radius * 2
                health_bar_height = 4
                health_bar_y = screen_pos[1] - screen_radius - health_bar_height - 2

                # 최대 체력 추정 (일반적으로 100)
                max_health = 100.0
                if hasattr(unit.status, "max_health"):
                    max_health = float(np.array(unit.status.max_health))

                health_ratio = max(0, min(1, health / max_health))

                # 체력 숫자 표시 (체력바 위에)
                if self.zoom > 0.7:  # 줌이 충분할 때만 숫자 표시
                    health_text = f"{int(health)}"
                    if alpha < 255:
                        # 투명한 유닛의 체력 텍스트도 흐리게
                        health_surface = self.small_font.render(health_text, True, (255, 255, 255))
                        health_text_rect = health_surface.get_rect(
                            center=(screen_pos[0], health_bar_y - 8)
                        )

                        # 알파 적용을 위한 surface 생성
                        text_surface = pygame.Surface(health_surface.get_size(), pygame.SRCALPHA)
                        text_surface.fill((0, 0, 0, 0))
                        text_surface.blit(health_surface, (0, 0))
                        text_surface.set_alpha(alpha)

                        # 텍스트 배경도 투명하게
                        text_bg_rect = health_text_rect.inflate(4, 2)
                        bg_surface = pygame.Surface(
                            (text_bg_rect.width, text_bg_rect.height), pygame.SRCALPHA
                        )
                        bg_surface.fill((0, 0, 0, min(180, alpha)))
                        self.screen.blit(bg_surface, text_bg_rect)
                        self.screen.blit(text_surface, health_text_rect)
                    else:
                        health_surface = self.small_font.render(health_text, True, (255, 255, 255))
                        health_text_rect = health_surface.get_rect(
                            center=(screen_pos[0], health_bar_y - 8)
                        )
                        # 텍스트 배경 (가독성을 위해)
                        text_bg_rect = health_text_rect.inflate(4, 2)
                        pygame.draw.rect(self.screen, (0, 0, 0, 180), text_bg_rect)
                        self.screen.blit(health_surface, health_text_rect)

                # 체력바 투명도 적용
                if alpha < 255:
                    # 체력바 Surface 생성
                    health_bar_surface = pygame.Surface(
                        (health_bar_width, health_bar_height), pygame.SRCALPHA
                    )
                    health_bar_surface.fill((0, 0, 0, 0))

                    # 배경
                    health_bg_color = (*self.colors["health_bg"], alpha)
                    pygame.draw.rect(
                        health_bar_surface,
                        health_bg_color,
                        (0, 0, health_bar_width, health_bar_height),
                    )

                    # 전경
                    if health_ratio > 0:
                        health_fg_color = (*self.colors["health_fg"], alpha)
                        pygame.draw.rect(
                            health_bar_surface,
                            health_fg_color,
                            (0, 0, int(health_bar_width * health_ratio), health_bar_height),
                        )

                    # 화면에 그리기
                    self.screen.blit(
                        health_bar_surface, (screen_pos[0] - health_bar_width // 2, health_bar_y)
                    )
                else:
                    # 체력바 배경
                    pygame.draw.rect(
                        self.screen,
                        self.colors["health_bg"],
                        (
                            screen_pos[0] - health_bar_width // 2,
                            health_bar_y,
                            health_bar_width,
                            health_bar_height,
                        ),
                    )

                    # 체력바 전경
                    if health_ratio > 0:
                        pygame.draw.rect(
                            self.screen,
                            self.colors["health_fg"],
                            (
                                screen_pos[0] - health_bar_width // 2,
                                health_bar_y,
                                int(health_bar_width * health_ratio),
                                health_bar_height,
                            ),
                        )

                # 쿨다운바 (체력바 아래에 표시)
                if hasattr(unit.status, "cooldown") and hasattr(unit.status, "attack_cooldown"):
                    if hasattr(unit.status.cooldown, "__array__"):
                        cooldown = float(np.array(unit.status.cooldown))
                    else:
                        cooldown = float(unit.status.cooldown)

                    if hasattr(unit.status.attack_cooldown, "__array__"):
                        attack_cooldown = float(np.array(unit.status.attack_cooldown))
                    else:
                        attack_cooldown = float(unit.status.attack_cooldown)

                    # 쿨다운 진행도 계산 (0: 쿨다운 중, 1: 공격 준비됨)
                    cooldown_ratio = (
                        min(1.0, cooldown / attack_cooldown) if attack_cooldown > 0 else 1.0
                    )

                    cooldown_bar_width = health_bar_width
                    cooldown_bar_height = 3  # 체력바보다 조금 작게
                    cooldown_bar_y = health_bar_y + health_bar_height + 1  # 체력바 바로 아래

                    # 쿨다운바 배경
                    pygame.draw.rect(
                        self.screen,
                        self.colors["cooldown_bg"],
                        (
                            screen_pos[0] - cooldown_bar_width // 2,
                            cooldown_bar_y,
                            cooldown_bar_width,
                            cooldown_bar_height,
                        ),
                    )

                    # 쿨다운바 전경
                    if cooldown_ratio > 0:
                        # 공격 준비되면 초록색, 쿨다운 중이면 노란색
                        cooldown_color = (
                            self.colors["cooldown_ready"]
                            if cooldown_ratio >= 1.0
                            else self.colors["cooldown_fg"]
                        )
                        pygame.draw.rect(
                            self.screen,
                            cooldown_color,
                            (
                                screen_pos[0] - cooldown_bar_width // 2,
                                cooldown_bar_y,
                                int(cooldown_bar_width * cooldown_ratio),
                                cooldown_bar_height,
                            ),
                        )

            # 유닛 이름 및 정보 (줌이 충분할 때만)
            if self.zoom > 1.0 and screen_radius > 10:
                # 유닛 타입 이름 가져오기
                unit_type_names = [
                    "",
                    "Farmer",
                    "Archer",
                    "TheKing",
                    "BombThrower",
                    "Mammoth",
                    "Deadeye",
                    "Healer",
                ]

                # unit_id 가져오기
                if hasattr(unit.status.unit_id, "__array__"):
                    unit_id = int(np.array(unit.status.unit_id))
                else:
                    unit_id = int(unit.status.unit_id)

                # unit_id가 유효한 범위인지 확인
                if 0 <= unit_id < len(unit_type_names):
                    display_name = f"{unit_name} ({unit_type_names[unit_id]})"
                else:
                    display_name = unit_name

                name_text = self.small_font.render(display_name, True, (255, 255, 255))
                name_rect = name_text.get_rect(
                    center=(screen_pos[0], screen_pos[1] + screen_radius + 15)
                )
                self.screen.blit(name_text, name_rect)

            # disabled 상태 확인 및 표시
            is_disabled = False
            if hasattr(unit.status, "is_disabled"):
                if hasattr(unit.status.is_disabled, "__array__"):
                    is_disabled = bool(np.array(unit.status.is_disabled))
                else:
                    is_disabled = bool(unit.status.is_disabled)

            # disabled 유닛에 "DISABLED" 텍스트 표시
            if is_disabled and screen_radius > 5:
                disabled_text = self.font.render("DISABLED", True, (255, 50, 50))  # 빨간색 텍스트
                disabled_rect = disabled_text.get_rect(
                    center=(screen_pos[0], screen_pos[1] + screen_radius + 35)  # 유닛 이름 아래
                )

                # 텍스트 배경 (가독성을 위해)
                bg_rect = disabled_rect.inflate(6, 4)
                bg_surface = pygame.Surface((bg_rect.width, bg_rect.height), pygame.SRCALPHA)
                bg_surface.fill((0, 0, 0, 200))  # 반투명 검정 배경
                self.screen.blit(bg_surface, bg_rect)

                # 텍스트 표시
                self.screen.blit(disabled_text, disabled_rect)

        except Exception:
            pass

    def draw_target_matrix(self, objects):
        """Target matrix 시각화"""
        if "game_manager" not in objects:
            return

        game_manager = objects["game_manager"]
        if not hasattr(game_manager, "attack_target"):
            return

        target_matrix = np.array(game_manager.attackable_matrix)

        # Target matrix가 1차원이면 2차원으로 변환 시도
        if target_matrix.ndim == 1:
            # 유닛 개수를 추정 (unit으로 시작하는 키들의 개수)
            n_units = sum(1 for key in objects.keys() if key.startswith("unit"))
            if len(target_matrix) == n_units * n_units:
                target_matrix = target_matrix.reshape(n_units, n_units)
            else:
                return  # 크기가 맞지 않으면 표시하지 않음

        if target_matrix.ndim != 2:
            return

        n_units = target_matrix.shape[0]

        # Target matrix 표시 위치 (화면 하단 중앙)
        matrix_size = 25  # 각 셀의 크기
        matrix_width = n_units * matrix_size
        matrix_height = n_units * matrix_size

        start_x = (self.width - matrix_width) // 2
        start_y = self.height - matrix_height - 50

        # 배경 그리기
        pygame.draw.rect(
            self.screen,
            self.colors["target_matrix_bg"],
            (start_x - 5, start_y - 30, matrix_width + 10, matrix_height + 35),
        )

        # 제목 그리기
        title_text = self.small_font.render("Target Matrix", True, self.colors["target_text"])
        title_rect = title_text.get_rect(center=(start_x + matrix_width // 2, start_y - 15))
        self.screen.blit(title_text, title_rect)

        # Matrix 셀들 그리기
        for i in range(n_units):
            for j in range(n_units):
                cell_x = start_x + j * matrix_size
                cell_y = start_y + i * matrix_size

                # 셀 색상 결정
                if target_matrix[i, j]:
                    cell_color = self.colors["target_true"]
                else:
                    cell_color = self.colors["target_false"]

                # 셀 그리기
                pygame.draw.rect(
                    self.screen, cell_color, (cell_x, cell_y, matrix_size - 1, matrix_size - 1)
                )

                # 테두리 그리기
                pygame.draw.rect(
                    self.screen,
                    self.colors["target_text"],
                    (cell_x, cell_y, matrix_size - 1, matrix_size - 1),
                    1,
                )

                # 값 텍스트 그리기 (선택적)
                if matrix_size >= 20:  # 셀이 충분히 클 때만
                    value_text = "1" if target_matrix[i, j] else "0"
                    text_surface = self.small_font.render(value_text, True, (0, 0, 0))
                    text_rect = text_surface.get_rect(
                        center=(cell_x + matrix_size // 2, cell_y + matrix_size // 2)
                    )
                    self.screen.blit(text_surface, text_rect)

                # 축 라벨 그리기 - battle_simulator와 동일한 순서로 unit key 생성
        unit_keys = []
        for i in range(n_units):
            unit_key = f"unit_{i}"
            if unit_key in objects:
                unit_keys.append(unit_key)

        for i in range(n_units):
            # Y축 라벨 (왼쪽)
            unit_suffix = unit_keys[i].split("_")[-1] if i < len(unit_keys) else str(i)
            label_text = self.small_font.render(f"{unit_suffix}", True, self.colors["target_text"])
            self.screen.blit(
                label_text, (start_x - 20, start_y + i * matrix_size + matrix_size // 2 - 8)
            )

            # X축 라벨 (위쪽)
            unit_suffix = unit_keys[i].split("_")[-1] if i < len(unit_keys) else str(i)
            label_text = self.small_font.render(f"{unit_suffix}", True, self.colors["target_text"])
            text_rect = label_text.get_rect(
                center=(start_x + i * matrix_size + matrix_size // 2, start_y - 5)
            )
            self.screen.blit(label_text, text_rect)

    def draw_visible_matrix(self, objects):
        """Visible matrix 시각화"""
        if "game_manager" not in objects:
            return

        game_manager = objects["game_manager"]
        if not hasattr(game_manager, "visible_matrix"):
            return

        visible_matrix = np.array(game_manager.visible_matrix)

        # Visible matrix가 1차원이면 2차원으로 변환 시도
        if visible_matrix.ndim == 1:
            # 유닛 개수를 추정 (unit으로 시작하는 키들의 개수)
            n_units = sum(1 for key in objects.keys() if key.startswith("unit"))
            if len(visible_matrix) == n_units * n_units:
                visible_matrix = visible_matrix.reshape(n_units, n_units)
            else:
                return  # 크기가 맞지 않으면 표시하지 않음

        if visible_matrix.ndim != 2:
            return

        n_units = visible_matrix.shape[0]

        # 선택된 유닛이 있으면 해당 유닛의 시야만 표시
        selected_unit_index = None
        if self.selected_unit:
            # 유닛 키 목록을 battle_simulator와 동일한 방식으로 생성
            unit_keys = []
            for i in range(n_units):
                unit_key = f"unit_{i}"
                if unit_key in objects:
                    unit_keys.append(unit_key)

            if self.selected_unit in unit_keys:
                selected_unit_index = unit_keys.index(self.selected_unit)

        # 선택된 유닛이 있으면 해당 행만 하이라이트, 나머지는 흐리게

        # Visible matrix 표시 위치 (화면 하단 우측)
        matrix_size = 25  # 각 셀의 크기
        matrix_width = n_units * matrix_size
        matrix_height = n_units * matrix_size

        start_x = self.width - matrix_width - 20  # 우측에 배치
        start_y = self.height - matrix_height - 50

        # 배경 그리기
        pygame.draw.rect(
            self.screen,
            self.colors["visible_matrix_bg"],
            (start_x - 5, start_y - 30, matrix_width + 10, matrix_height + 35),
        )

        # 제목 그리기
        if selected_unit_index is not None:
            title = f"Visible Matrix ({self.selected_unit} view)"
        else:
            title = "Visible Matrix"
        title_text = self.small_font.render(title, True, self.colors["visible_text"])
        title_rect = title_text.get_rect(center=(start_x + matrix_width // 2, start_y - 15))
        self.screen.blit(title_text, title_rect)

        # Matrix 셀들 그리기
        for i in range(n_units):
            for j in range(n_units):
                cell_x = start_x + j * matrix_size
                cell_y = start_y + i * matrix_size

                # 셀 색상 결정
                if selected_unit_index is not None:
                    # 선택된 유닛이 있는 경우
                    if i == selected_unit_index:
                        # 선택된 유닛의 행 (해당 유닛이 보는 것들)
                        if visible_matrix[i, j]:
                            cell_color = self.colors["visible_selected_true"]
                        else:
                            cell_color = self.colors["visible_selected_false"]
                    else:
                        # 선택되지 않은 유닛의 행 (흐리게 표시)
                        if visible_matrix[i, j]:
                            cell_color = self.colors["visible_dimmed_true"]
                        else:
                            cell_color = self.colors["visible_dimmed_false"]
                else:
                    # 선택된 유닛이 없는 경우 (기본 표시)
                    if visible_matrix[i, j]:
                        cell_color = self.colors["visible_true"]
                    else:
                        cell_color = self.colors["visible_false"]

                # 셀 그리기
                pygame.draw.rect(
                    self.screen, cell_color, (cell_x, cell_y, matrix_size - 1, matrix_size - 1)
                )

                # 테두리 그리기
                pygame.draw.rect(
                    self.screen,
                    self.colors["visible_text"],
                    (cell_x, cell_y, matrix_size - 1, matrix_size - 1),
                    1,
                )

                # 값 텍스트 그리기 (선택적)
                if matrix_size >= 20:  # 셀이 충분히 클 때만
                    value_text = "1" if visible_matrix[i, j] else "0"
                    text_surface = self.small_font.render(value_text, True, (0, 0, 0))
                    text_rect = text_surface.get_rect(
                        center=(cell_x + matrix_size // 2, cell_y + matrix_size // 2)
                    )
                    self.screen.blit(text_surface, text_rect)

                # 축 라벨 그리기 - battle_simulator와 동일한 순서로 unit key 생성
        unit_keys = []
        for i in range(n_units):
            unit_key = f"unit_{i}"
            if unit_key in objects:
                unit_keys.append(unit_key)

        for i in range(n_units):
            # Y축 라벨 (왼쪽) - 선택된 유닛은 하이라이트
            if selected_unit_index is not None and i == selected_unit_index:
                label_color = self.colors["visible_selected_true"]  # 선택된 유닛은 밝은 색
                # unit key에서 숫자만 추출하여 표시
                unit_suffix = unit_keys[i].split("_")[-1] if i < len(unit_keys) else str(i)
                label_text = self.small_font.render(f"{unit_suffix}*", True, label_color)
            else:
                label_color = self.colors["visible_text"]
                # unit key에서 숫자만 추출하여 표시
                unit_suffix = unit_keys[i].split("_")[-1] if i < len(unit_keys) else str(i)
                label_text = self.small_font.render(f"{unit_suffix}", True, label_color)
            self.screen.blit(
                label_text, (start_x - 20, start_y + i * matrix_size + matrix_size // 2 - 8)
            )

            # X축 라벨 (위쪽)
            unit_suffix = unit_keys[i].split("_")[-1] if i < len(unit_keys) else str(i)
            label_text = self.small_font.render(f"{unit_suffix}", True, self.colors["visible_text"])
            text_rect = label_text.get_rect(
                center=(start_x + i * matrix_size + matrix_size // 2, start_y - 5)
            )
            self.screen.blit(label_text, text_rect)

    def draw_ui(self, objects):
        """UI 정보 그리기"""
        y_offset = 10

        # 컨트롤 정보
        if self.selected_unit:
            controls = [
                "Unit Control Mode:",
                f"Selected: {self.selected_unit}",
                "WASD/Arrow Keys: Move Unit",
                "Ctrl: Attack",
                "Mouse: Rotation Direction",
                "R: Reset Environment",
                "Tab: Toggle UI Panel",
                "Left Click Unit: Select",
                "Left Click Ground: Deselect",
                "ESC: Exit",
            ]
        else:
            controls = [
                "Camera Control Mode:",
                "WASD/Arrow Keys: Move Camera",
                "Mouse Wheel: Zoom",
                "Space: Reset Camera",
                "R: Reset Environment",
                "Tab: Toggle UI Panel",
                "Left Click Unit: Select",
                "ESC: Exit",
            ]

        for i, text in enumerate(controls):
            color = (255, 255, 255) if i == 0 else (200, 200, 200)
            rendered_text = self.small_font.render(text, True, color)
            self.screen.blit(rendered_text, (10, y_offset))
            y_offset += 20

        # 게임 정보
        y_offset += 10
        unit_count = sum(1 for key in objects.keys() if "unit" in key.lower())
        info_text = self.font.render(f"Units: {unit_count}", True, (255, 255, 255))
        self.screen.blit(info_text, (10, y_offset))

        y_offset += 25
        zoom_text = self.small_font.render(f"Zoom: {self.zoom:.2f}", True, (255, 255, 255))
        self.screen.blit(zoom_text, (10, y_offset))

        # 범례
        legend_x = self.width - 150
        legend_y = 10

        legend_items = [
            ("Team 0", self.colors["team_0"]),
            ("Team 1", self.colors["team_1"]),
            ("Attack Range", self.colors["attack_range"][:3]),
            ("Sight Range", self.colors["sight_range"][:3]),
        ]

        legend_title = self.small_font.render("Legend:", True, (255, 255, 255))
        self.screen.blit(legend_title, (legend_x, legend_y))
        legend_y += 20

        for name, color in legend_items:
            pygame.draw.circle(self.screen, color, (legend_x + 10, legend_y + 8), 6)
            text = self.small_font.render(name, True, (255, 255, 255))
            self.screen.blit(text, (legend_x + 25, legend_y))
            legend_y += 18

        # Target matrix 그리기
        if self.ui_panel["show_distance_matrix"]:
            self.draw_target_matrix(objects)

        # Visible matrix 그리기
        if self.ui_panel["show_visible_matrix"]:
            self.draw_visible_matrix(objects)

    def draw_obs_array(self):
        """선택된 유닛의 obs 배열을 화면에 표시 (스크롤 가능)"""
        if not self.selected_unit or self.selected_unit_obs is None:
            return

        # obs 배열 표시 패널 설정
        panel_width = 500  # 폭을 늘려서 더 많은 정보 표시
        panel_height = min(700, self.height - 20)  # 화면 높이에 맞춰 조정
        panel_x = (self.game_width - panel_width) // 2  # 게임 화면 중앙에 배치
        panel_y = 10

        # 반투명 배경 그리기
        panel_surface = pygame.Surface((panel_width, panel_height), pygame.SRCALPHA)
        panel_surface.fill((0, 0, 0, 220))  # 조금 더 어둡게
        self.screen.blit(panel_surface, (panel_x, panel_y))

        # 테두리 그리기
        pygame.draw.rect(
            self.screen, (150, 150, 150), (panel_x, panel_y, panel_width, panel_height), 2
        )

        # 제목 그리기
        import time

        current_time = time.time()
        title_text = f"{self.selected_unit} Observation Array"
        title_surface = self.font.render(title_text, True, (255, 255, 255))
        title_rect = title_surface.get_rect(center=(panel_x + panel_width // 2, panel_y + 20))
        self.screen.blit(title_surface, title_rect)

        # 업데이트 시간 표시 (디버깅용)
        time_text = f"Last updated: {current_time:.2f}"
        time_surface = self.small_font.render(time_text, True, (150, 150, 150))
        self.screen.blit(time_surface, (panel_x + 10, panel_y + 30))

        # obs 배열을 텍스트로 표시
        try:
            import numpy as np

            obs_array = np.array(self.selected_unit_obs)

            # obs 배열의 구조 정보 표시
            info_y = panel_y + 60  # 업데이트 시간 표시 공간 확보
            info_texts = [
                f"Selected Unit: {self.selected_unit}",
                f"Array Shape: {obs_array.shape}",
                f"Array Size: {obs_array.size}",
                f"Data Type: {obs_array.dtype}",
                f"Scroll Offset: {self.obs_scroll_offset}",
                "",
                "Array Values (scroll with mouse wheel):",
            ]

            for text in info_texts:
                if text:  # 빈 문자열이 아닌 경우에만 렌더링
                    text_surface = self.small_font.render(text, True, (200, 200, 200))
                    self.screen.blit(text_surface, (panel_x + 10, info_y))
                info_y += 16

            # 스크롤 가능한 배열 값들 표시
            content_start_y = info_y
            content_height = panel_height - (content_start_y - panel_y) - 100  # 통계 정보 공간 확보
            line_height = 16
            max_visible_lines = content_height // line_height

            # 스크롤 오프셋 범위 제한
            max_scroll = max(0, len(obs_array) - max_visible_lines)
            self.obs_scroll_offset = max(0, min(self.obs_scroll_offset, max_scroll))

            # 표시할 요소들의 범위 계산
            start_idx = self.obs_scroll_offset
            end_idx = min(start_idx + max_visible_lines, len(obs_array))

            for i in range(start_idx, end_idx):
                display_y = content_start_y + (i - start_idx) * line_height

                # 인덱스를 3자리로 포맷팅하여 더 많은 요소 표시 가능
                value_text = f"[{i:3d}]: {obs_array[i]:.6f}"

                # 값의 크기에 따라 색상 변경
                if abs(obs_array[i]) > 1.0:
                    color = (255, 200, 200)  # 큰 값은 빨간색 계열
                elif abs(obs_array[i]) > 0.1:
                    color = (200, 255, 200)  # 중간 값은 초록색 계열
                else:
                    color = (200, 200, 255)  # 작은 값은 파란색 계열

                text_surface = self.small_font.render(value_text, True, color)
                self.screen.blit(text_surface, (panel_x + 10, display_y))

            # 스크롤 인디케이터 표시
            if len(obs_array) > max_visible_lines:
                scroll_info_y = content_start_y + max_visible_lines * line_height + 10
                scroll_text = f"Showing {start_idx}-{end_idx - 1} of {len(obs_array)} elements"
                scroll_surface = self.small_font.render(scroll_text, True, (150, 150, 150))
                self.screen.blit(scroll_surface, (panel_x + 10, scroll_info_y))

                # 스크롤바 그리기
                scrollbar_x = panel_x + panel_width - 20
                scrollbar_y = content_start_y
                scrollbar_height = max_visible_lines * line_height
                scrollbar_width = 10

                # 스크롤바 배경
                pygame.draw.rect(
                    self.screen,
                    (80, 80, 80),
                    (scrollbar_x, scrollbar_y, scrollbar_width, scrollbar_height),
                )

                # 스크롤바 핸들
                if max_scroll > 0:
                    handle_height = max(20, scrollbar_height * max_visible_lines // len(obs_array))
                    handle_y = (
                        scrollbar_y
                        + (scrollbar_height - handle_height) * self.obs_scroll_offset // max_scroll
                    )
                    pygame.draw.rect(
                        self.screen,
                        (150, 150, 150),
                        (scrollbar_x, handle_y, scrollbar_width, handle_height),
                    )

            # 통계 정보 표시
            stats_y = panel_y + panel_height - 80
            stats_texts = [
                f"Min: {obs_array.min():.6f}",
                f"Max: {obs_array.max():.6f}",
                f"Mean: {obs_array.mean():.6f}",
                f"Std: {obs_array.std():.6f}",
            ]

            for text in stats_texts:
                text_surface = self.small_font.render(text, True, (255, 255, 100))
                self.screen.blit(text_surface, (panel_x + 10, stats_y))
                stats_y += 16

            # 조작 안내 표시
            help_y = panel_y + panel_height - 20
            help_text = "Mouse wheel: scroll, R: reset scroll"
            help_surface = self.small_font.render(help_text, True, (180, 180, 180))
            self.screen.blit(help_surface, (panel_x + 10, help_y))

        except Exception as e:
            # 에러 발생시 에러 메시지 표시
            error_text = f"Error displaying obs array: {str(e)}"
            error_surface = self.small_font.render(error_text, True, (255, 100, 100))
            self.screen.blit(error_surface, (panel_x + 10, panel_y + 50))

    def draw_unit_info(self, objects):
        """선택된 유닛의 상세 정보를 왼쪽 아래에 표시"""
        if not self.selected_unit or self.selected_unit not in objects:
            return

        unit = objects[self.selected_unit]

        # 정보 패널 배경 설정
        panel_width = 300
        panel_height = 200
        panel_x = 10
        panel_y = self.height - panel_height - 10

        # 반투명 배경 그리기
        panel_surface = pygame.Surface((panel_width, panel_height), pygame.SRCALPHA)
        panel_surface.fill((0, 0, 0, 180))  # 반투명 검정
        self.screen.blit(panel_surface, (panel_x, panel_y))

        # 테두리 그리기
        pygame.draw.rect(
            self.screen, (100, 100, 100), (panel_x, panel_y, panel_width, panel_height), 2
        )

        # 유닛 정보 수집
        try:
            # 위치 정보
            if hasattr(unit.transform.position, "__array__"):
                pos = np.array(unit.transform.position)
            else:
                pos = unit.transform.position

            # 회전 각도 (라디안을 도로 변환)
            if hasattr(unit.transform.rotation, "__array__"):
                rotation_rad = float(np.array(unit.transform.rotation))
            else:
                rotation_rad = float(unit.transform.rotation)
            rotation_deg = math.degrees(rotation_rad)

            # 체력 정보
            if hasattr(unit.status.health, "__array__"):
                health = float(np.array(unit.status.health))
            else:
                health = float(unit.status.health)

            # 팀 정보
            if hasattr(unit.team, "__array__"):
                team = int(np.array(unit.team))
            else:
                team = int(unit.team)

            # 공격 정보
            if hasattr(unit.status.attack_damage, "__array__"):
                attack_damage = float(np.array(unit.status.attack_damage))
            else:
                attack_damage = float(unit.status.attack_damage)

            if hasattr(unit.status.attack_range, "__array__"):
                attack_range = float(np.array(unit.status.attack_range))
            else:
                attack_range = float(unit.status.attack_range)

            # 쿨다운 정보
            if hasattr(unit.status.cooldown, "__array__"):
                cooldown = float(np.array(unit.status.cooldown))
            else:
                cooldown = float(unit.status.cooldown)

            if hasattr(unit.status.attack_cooldown, "__array__"):
                attack_cooldown = float(np.array(unit.status.attack_cooldown))
            else:
                attack_cooldown = float(unit.status.attack_cooldown)

            # 공격 중인지 확인
            attacking = False
            if hasattr(unit, "attacking"):
                if hasattr(unit.attacking, "__array__"):
                    attacking = bool(np.array(unit.attacking))
                else:
                    attacking = bool(unit.attacking)

            # 생존 상태 확인
            is_alive = True  # 기본값
            if hasattr(unit.status, "is_alive"):
                if hasattr(unit.status.is_alive, "__array__"):
                    is_alive = bool(np.array(unit.status.is_alive))
                else:
                    is_alive = bool(unit.status.is_alive)

            # 정보 텍스트 생성
            info_lines = [
                f"Unit: {self.selected_unit}",
                f"Position: ({pos[0]:.2f}, {pos[1]:.2f})",
                f"Rotation: {rotation_deg:.1f}°",
                f"Team: {team}",
                f"Health: {health:.1f}",
                f"Is Alive: {'Yes' if is_alive else 'No'}",
                f"Attack Damage: {attack_damage:.1f}",
                f"Attack Range: {attack_range:.1f}",
                f"Cooldown: {cooldown:.1f}/{attack_cooldown:.1f}",
                f"Attacking: {'Yes' if attacking else 'No'}",
            ]

        except Exception as e:
            info_lines = [
                f"Unit: {self.selected_unit}",
                "Error reading unit data",
                f"Error: {str(e)}",
            ]

        # 텍스트 렌더링
        text_y = panel_y + 10
        for line in info_lines:
            text_surface = self.small_font.render(line, True, (255, 255, 255))
            self.screen.blit(text_surface, (panel_x + 10, text_y))
            text_y += 20

    def draw_distance_matrix_visualization(self, objects):
        """선택된 유닛의 distance_matrix를 시각화하여 다른 유닛들과의 상대 좌표를 표시"""
        if not self.selected_unit or self.selected_unit not in objects:
            return

        if "game_manager" not in objects:
            return

        game_manager = objects["game_manager"]
        if not hasattr(game_manager, "distance_matrix"):
            return

        try:
            # distance_matrix 가져오기
            distance_matrix = np.array(game_manager.distance_matrix)

            # 유닛 목록 구성
            unit_keys = [key for key in objects.keys() if "unit" in key.lower()]
            unit_keys.sort()  # unit1, unit2, unit3 순서로 정렬

            if self.selected_unit not in unit_keys:
                return

            # 선택된 유닛의 인덱스 찾기
            selected_index = unit_keys.index(self.selected_unit)

            # 선택된 유닛의 위치 가져오기
            selected_unit = objects[self.selected_unit]
            if hasattr(selected_unit.transform.position, "__array__"):
                selected_pos = np.array(selected_unit.transform.position)
            else:
                selected_pos = selected_unit.transform.position

            selected_screen_pos = self.world_to_screen(selected_pos)

            # distance_matrix[selected_index]를 사용하여 다른 유닛들과의 상대 좌표 시각화
            if distance_matrix.ndim >= 2 and selected_index < len(distance_matrix):
                relative_positions = distance_matrix[selected_index]

                for i, relative_pos in enumerate(relative_positions):
                    if i == selected_index:  # 자기 자신은 건너뛰기
                        continue

                    if i < len(unit_keys):
                        target_unit_name = unit_keys[i]

                        # 상대 좌표를 절대 좌표로 변환
                        if len(relative_pos) >= 2:
                            relative_x, relative_y = relative_pos[0], relative_pos[1]

                            # 상대 좌표를 화면 좌표로 변환
                            # distance_matrix의 값이 이미 월드 좌표 상의 상대 위치라고 가정
                            target_world_pos = (
                                selected_pos[0] + relative_x,
                                selected_pos[1] + relative_y,
                            )
                            target_screen_pos = self.world_to_screen(target_world_pos)

                            # 실제 유닛 위치와 비교 (검증용)
                            if target_unit_name in objects:
                                actual_unit = objects[target_unit_name]
                                if hasattr(actual_unit.transform.position, "__array__"):
                                    actual_pos = np.array(actual_unit.transform.position)
                                else:
                                    actual_pos = actual_unit.transform.position
                                actual_screen_pos = self.world_to_screen(actual_pos)

                                # 상대 좌표 벡터를 화살표로 표시
                                self.draw_relative_position_arrow(
                                    selected_screen_pos,
                                    target_screen_pos,
                                    actual_screen_pos,
                                    target_unit_name,
                                    (relative_x, relative_y),
                                )

        except Exception as e:
            # 디버그용 - 에러가 발생하면 화면에 표시
            error_text = self.small_font.render(
                f"Distance Matrix Error: {str(e)}", True, (255, 0, 0)
            )
            self.screen.blit(error_text, (10, 50))

    def draw_relative_position_arrow(
        self, selected_pos, calculated_pos, actual_pos, target_name, relative_coords
    ):
        """상대 좌표를 화살표로 시각화 (유닛 radius 고려)"""
        # 선택된 유닛에서 계산된 위치로의 화살표 (파란색)
        self.draw_arrow(selected_pos, calculated_pos, (0, 150, 255), 3, "calculated")

        # 선택된 유닛에서 실제 위치로의 화살표 (초록색)
        self.draw_arrow(selected_pos, actual_pos, (0, 255, 0), 2, "actual")

        # 계산된 위치와 실제 위치의 차이를 빨간색 선으로 표시 (오차 시각화)
        distance_diff = math.sqrt(
            (calculated_pos[0] - actual_pos[0]) ** 2 + (calculated_pos[1] - actual_pos[1]) ** 2
        )
        if distance_diff > 5:  # 5픽셀 이상 차이날 때만 표시
            pygame.draw.line(self.screen, (255, 0, 0), calculated_pos, actual_pos, 1)

        # 상대 좌표 값을 텍스트로 표시
        mid_pos = (
            (selected_pos[0] + actual_pos[0]) // 2,
            (selected_pos[1] + actual_pos[1]) // 2 - 15,
        )

        # 실제 중심 간 거리 계산
        actual_distance = math.sqrt(relative_coords[0] ** 2 + relative_coords[1] ** 2)
        coord_text = f"{target_name}: ({relative_coords[0]:.2f}, {relative_coords[1]:.2f}) d={actual_distance:.2f}"
        text_surface = self.small_font.render(coord_text, True, (255, 255, 255))

        # 텍스트 배경 (가독성을 위해)
        text_rect = text_surface.get_rect()
        text_rect.center = mid_pos

        bg_surface = pygame.Surface((text_rect.width + 4, text_rect.height + 2), pygame.SRCALPHA)
        bg_surface.fill((0, 0, 0, 150))
        self.screen.blit(bg_surface, (text_rect.x - 2, text_rect.y - 1))
        self.screen.blit(text_surface, text_rect)

    def draw_arrow(self, start_pos, end_pos, color, thickness, arrow_type):
        """화살표를 그리는 헬퍼 함수"""
        # 메인 라인
        pygame.draw.line(self.screen, color, start_pos, end_pos, thickness)

        # 화살표 머리 계산
        dx = end_pos[0] - start_pos[0]
        dy = end_pos[1] - start_pos[1]
        length = math.sqrt(dx * dx + dy * dy)

        if length > 0:
            # 정규화
            dx /= length
            dy /= length

            # 화살표 머리 크기
            arrow_length = 10
            arrow_angle = math.pi / 6  # 30도

            # 화살표 머리의 두 점 계산
            arrow_x1 = end_pos[0] - arrow_length * (
                dx * math.cos(arrow_angle) - dy * math.sin(arrow_angle)
            )
            arrow_y1 = end_pos[1] - arrow_length * (
                dy * math.cos(arrow_angle) + dx * math.sin(arrow_angle)
            )

            arrow_x2 = end_pos[0] - arrow_length * (
                dx * math.cos(-arrow_angle) - dy * math.sin(-arrow_angle)
            )
            arrow_y2 = end_pos[1] - arrow_length * (
                dy * math.cos(-arrow_angle) + dx * math.sin(-arrow_angle)
            )

            # 화살표 머리 그리기
            pygame.draw.line(self.screen, color, end_pos, (arrow_x1, arrow_y1), thickness)
            pygame.draw.line(self.screen, color, end_pos, (arrow_x2, arrow_y2), thickness)

    def render(self, objects, show_ranges=True, obs=None):
        """
        객체들을 렌더링

        Args:
            objects: 게임 객체들의 딕셔너리 (key는 이름, value는 unit 객체)
            show_ranges: 공격/시야 범위를 표시할지 여부
            obs: 관찰값 딕셔너리 (key는 유닛 이름, value는 obs 배열)
        """
        if not self.running:
            return False

        # 이벤트 처리
        self.handle_events(objects)

        # obs가 제공되고 선택된 유닛이 있을 때 obs 배열 저장
        if obs and self.selected_unit:
            if self.selected_unit in obs:
                # 항상 최신 obs로 업데이트 (실시간 변화 반영)
                new_obs = obs[self.selected_unit]
                self.selected_unit_obs = new_obs

                # 디버깅: 첫 번째 유닛 선택 시 obs 딕셔너리 구조 출력
                if not hasattr(self, "_debug_printed"):
                    self._debug_printed = True
            else:
                self.selected_unit_obs = None

        # 화면 클리어
        self.screen.fill(self.colors["background"])

        # 격자 그리기 (옵션)
        if self.zoom > 0.5 and self.ui_panel["show_grid"]:
            self.draw_grid()

        # 선택된 유닛의 시야 정보 가져오기
        visibility_info = None
        if self.selected_unit and "game_manager" in objects:
            game_manager = objects["game_manager"]
            if hasattr(game_manager, "visible_matrix"):
                try:
                    visible_matrix = np.array(game_manager.visible_matrix)

                    if visible_matrix.ndim == 1:
                        # 유닛 개수 추정
                        n_units = sum(1 for key in objects.keys() if key.startswith("unit"))
                        if len(visible_matrix) == n_units * n_units:
                            visible_matrix = visible_matrix.reshape(n_units, n_units)

                        else:
                            pass
                    elif visible_matrix.ndim == 2:
                        pass
                    else:
                        visible_matrix = None

                    if visible_matrix is not None and visible_matrix.ndim == 2:
                        # 유닛 키 목록을 battle_simulator와 동일한 방식으로 생성
                        # battle_simulator에서는 self.unit_keys를 사용하므로 같은 순서로 만들어야 함
                        unit_keys = []
                        for i in range(visible_matrix.shape[0]):
                            unit_key = f"unit_{i}"
                            if unit_key in objects:
                                unit_keys.append(unit_key)

                        if self.selected_unit in unit_keys:
                            selected_index = unit_keys.index(self.selected_unit)
                            visibility_info = {
                                "matrix": visible_matrix,
                                "selected_index": selected_index,
                                "unit_keys": unit_keys,
                            }

                        else:
                            pass

                except Exception:
                    pass

        # 유닛들 그리기

        if visibility_info:
            pass

        for obj_name, obj in objects.items():
            if "unit" in obj_name.lower() and hasattr(obj, "transform"):
                # 투명도 계산
                alpha = 255  # 기본값: 완전 불투명
                if visibility_info and obj_name != self.selected_unit:
                    # 선택된 유닛이 있고, 현재 유닛이 선택된 유닛이 아닌 경우
                    if obj_name in visibility_info["unit_keys"]:
                        target_index = visibility_info["unit_keys"].index(obj_name)
                        # visible_matrix에서 matrix[i, j]는 unit i가 unit j를 볼 수 있는지를 나타냄
                        # 따라서 selected_unit이 target_unit을 볼 수 있는지 확인하려면
                        # matrix[selected_index, target_index]를 확인해야 함
                        can_see = visibility_info["matrix"][
                            visibility_info["selected_index"], target_index
                        ]

                        if can_see:
                            # 선택된 유닛이 현재 유닛을 볼 수 있는 경우
                            alpha = 200  # 반투명하게 표시 (조금 흐리게)
                        else:
                            # 선택된 유닛이 현재 유닛을 볼 수 없는 경우
                            alpha = 60  # 더 투명하게 표시 (거의 투명)

                # 선택된 유닛의 범위를 자동으로 표시
                if obj_name == self.selected_unit:
                    # 선택된 유닛은 항상 시야와 공격 범위 표시
                    unit_show_ranges = {
                        "sight": True,
                        "attack": True,
                    }
                else:
                    # 다른 유닛들은 UI 패널 설정에 따라 표시
                    unit_show_ranges = {
                        "sight": self.ui_panel["show_sight_range"],
                        "attack": self.ui_panel["show_attack_range"],
                    }
                self.draw_unit(obj_name, obj, unit_show_ranges, alpha)

        # UI 그리기
        self.draw_ui(objects)

        # 선택된 유닛 정보 표시
        if self.ui_panel["show_unit_info"]:
            self.draw_unit_info(objects)

        # distance_matrix 시각화
        if self.ui_panel["show_distance_matrix"]:
            self.draw_distance_matrix_visualization(objects)

        # obs 배열 표시
        if self.ui_panel["show_obs_array"]:
            self.draw_obs_array()

        # UI 패널과 토글 버튼 그리기 (가장 마지막에)
        self.draw_toggle_button()
        if self.panel_visible:
            self.draw_ui_panel()

        # 시나리오 선택기 그리기 (가장 위에 표시)
        self.draw_scenario_selector()

        # 정책 선택기 그리기 (시나리오 선택기 아래)
        self.draw_policy_selector()

        # 화면 업데이트
        pygame.display.flip()
        self.clock.tick(self.fps)

        return self.running

    def draw_rectangular_attack_range(
        self,
        screen_pos,
        rotation,
        attack_range,
        unit_radius,
        alpha=None,
        attack_range_angle=math.pi / 4,
    ):
        """
        TABS 스타일의 직사각형 공격 범위를 그리기

        Args:
            screen_pos: 유닛의 화면 좌표
            rotation: 유닛의 회전 각도 (라디안)
            attack_range: 공격 범위 거리
            unit_radius: 유닛의 반지름
            attack_range_angle: 공격 각도의 절반 (기본값: π/4)
        """
        if attack_range * self.world_scale * self.zoom < 10:
            return

        # 공격 범위의 스크린 크기 계산
        attack_screen_range = attack_range * self.world_scale * self.zoom
        unit_screen_radius = unit_radius * self.world_scale * self.zoom

        # battle_simulator.py의 로직과 정확히 일치하도록 공격 범위 계산
        cos_half_angle = math.cos(attack_range_angle / 2)
        sin_half_angle = math.sin(attack_range_angle / 2)

        # 공격 범위 직사각형 정의 (battle_simulator.py 로직을 정확히 따름)
        rx = cos_half_angle * unit_screen_radius
        width = attack_screen_range

        # battle_simulator.py에서 height는 p1과 p2 사이의 거리
        # p1 = [cos, -sin] * r, p2 = [cos, sin] * r
        # height = ||p1 - p2|| = ||(0, -2*sin)|| = 2*sin*r
        # 그리고 ry - height부터 ry까지가 범위가 아니라, ry - height/2부터 ry + height/2까지가 범위

        rect_points = [
            (rx, -sin_half_angle * unit_screen_radius),  # 좌측 하단 (p1의 y좌표)
            (rx + width, -sin_half_angle * unit_screen_radius),  # 우측 하단
            (rx + width, sin_half_angle * unit_screen_radius),  # 우측 상단 (p2의 y좌표)
            (rx, sin_half_angle * unit_screen_radius),  # 좌측 상단
        ]

        # 회전 변환 적용
        cos_rot = math.cos(rotation)
        sin_rot = math.sin(rotation)

        rotated_points = []
        for x, y in rect_points:
            # 회전 행렬 적용
            new_x = x * cos_rot - y * sin_rot
            new_y = x * sin_rot + y * cos_rot
            # 유닛 위치에 상대적으로 배치 (수학적 좌표계)
            screen_x = screen_pos[0] + new_x
            screen_y = screen_pos[1] - new_y
            rotated_points.append((screen_x, screen_y))

        # 반투명 직사각형 그리기
        try:
            # 충분한 크기의 surface 생성
            max_dim = max(self.width, self.height)
            attack_surface = pygame.Surface((max_dim, max_dim), pygame.SRCALPHA)

            # 상대 좌표로 변환 (surface 중심 기준)
            surface_center = max_dim // 2
            surface_points = []
            for x, y in rotated_points:
                surface_x = x - screen_pos[0] + surface_center
                surface_y = y - screen_pos[1] + surface_center
                surface_points.append((surface_x, surface_y))

            # 투명도에 따른 색상 조정
            if alpha is not None and alpha < 255:
                attack_color = (*self.colors["attack_range"][:3], alpha // 3)  # 더 투명하게
                border_color = (*self.colors["attack_range"][:3], alpha // 2)
            else:
                attack_color = self.colors["attack_range"]
                border_color = (*self.colors["attack_range"][:3], 150)

            # 다각형 그리기
            pygame.draw.polygon(attack_surface, attack_color, surface_points)

            # 외곽선 그리기
            pygame.draw.polygon(attack_surface, border_color, surface_points, 2)

            # 화면에 블릿
            blit_x = screen_pos[0] - surface_center
            blit_y = screen_pos[1] - surface_center
            self.screen.blit(attack_surface, (blit_x, blit_y))

        except Exception:
            # 에러 발생시 간단한 원형으로 대체
            attack_screen_radius = int(attack_range * self.world_scale * self.zoom)
            if attack_screen_radius > 3:
                attack_surface = pygame.Surface(
                    (attack_screen_radius * 2, attack_screen_radius * 2), pygame.SRCALPHA
                )
                pygame.draw.circle(
                    attack_surface,
                    self.colors["attack_range"],
                    (attack_screen_radius, attack_screen_radius),
                    attack_screen_radius,
                )
                self.screen.blit(
                    attack_surface,
                    (screen_pos[0] - attack_screen_radius, screen_pos[1] - attack_screen_radius),
                )

    def draw_fan_sight_range(self, screen_pos, rotation, sight_angle, alpha=None):
        """
        부채꼴 모양의 시야 범위를 그리기 (무한 범위)

        Args:
            screen_pos: 유닛의 화면 좌표
            rotation: 유닛의 회전 각도 (라디안)
            sight_angle: 시야 각도 (degree 또는 radian)
        """
        # sight_angle이 도 단위인지 라디안 단위인지 확인하여 변환
        # 일반적으로 60도(약 1.047 라디안) 정도가 기본값
        if sight_angle > 6.28:  # 2*PI보다 크면 도 단위로 간주
            sight_angle_rad = math.radians(sight_angle)
        else:
            sight_angle_rad = sight_angle

        # 시야 각도가 너무 작으면 그리지 않음
        if sight_angle_rad < 0.01:  # 약 0.6도
            return

        # 화면 끝까지 뻗어나가는 충분한 거리 계산
        max_screen_distance = math.sqrt(self.width**2 + self.height**2)

        # 회전 각도를 -π ~ π 범위로 정규화 (360도 제한 제거)
        normalized_rotation = math.atan2(math.sin(rotation), math.cos(rotation))

        # 부채꼴의 시작 각도와 끝 각도 계산 (유닛 회전 방향 기준)
        start_angle = normalized_rotation - sight_angle_rad / 2
        end_angle = normalized_rotation + sight_angle_rad / 2

        # 부채꼴 점들 계산
        # 중심점에서 시작
        fan_points = [screen_pos]

        # 원호를 따라 점들 생성 (각도 단위로)
        num_segments = max(8, int(sight_angle_rad * 180 / math.pi / 5))  # 각도에 비례한 세그먼트 수

        for i in range(num_segments + 1):
            # 선형 보간으로 각도 계산 (360도 래핑 고려)
            t = i / num_segments
            angle = start_angle + (end_angle - start_angle) * t

            # 각도를 정규화하여 안정적인 cos/sin 계산
            angle = math.atan2(math.sin(angle), math.cos(angle))

            x = screen_pos[0] + max_screen_distance * math.cos(angle)
            y = screen_pos[1] - max_screen_distance * math.sin(angle)  # 수학적 좌표계
            fan_points.append((x, y))

        # 다시 중심점으로 돌아와서 부채꼴 완성
        fan_points.append(screen_pos)

        try:
            # 부채꼴 영역을 반투명하게 그리기
            if len(fan_points) > 3:  # 최소 점의 개수 확인
                # 화면 좌표계에서 직접 그리기 (surface 변환 없이)
                screen_fan_points = []
                for x, y in fan_points:
                    # 유효한 좌표인지 확인 (NaN이나 무한대 방지)
                    if math.isfinite(x) and math.isfinite(y):
                        # 화면 경계 내로 클리핑 (더 여유있게)
                        screen_x = max(-self.width, min(self.width * 2, x))
                        screen_y = max(-self.height, min(self.height * 2, y))
                        screen_fan_points.append((screen_x, screen_y))
                    else:
                        # 잘못된 좌표는 중심점으로 대체
                        screen_fan_points.append(screen_pos)

                if len(screen_fan_points) > 3:
                    # 투명도에 따른 색상 조정
                    if alpha is not None and alpha < 255:
                        sight_color = (*self.colors["sight_range"][:3], alpha // 4)  # 더 투명하게
                    else:
                        sight_color = self.colors["sight_range"]

                    # 별도 surface 생성해서 반투명 효과
                    temp_surface = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
                    pygame.draw.polygon(temp_surface, sight_color, screen_fan_points)
                    self.screen.blit(temp_surface, (0, 0))

                    # 디버그: 부채꼴의 경계선을 그리기 (동일한 점들 사용)
                    # 첫 번째와 마지막 점은 중심점이므로 제외하고 경계선만 그리기
                    if len(screen_fan_points) > 2:
                        # 시작 경계선 (첫 번째 호 점과 중심 연결)
                        pygame.draw.line(
                            self.screen, (255, 0, 0), screen_fan_points[0], screen_fan_points[1], 2
                        )
                        # 끝 경계선 (마지막 호 점과 중심 연결)
                        pygame.draw.line(
                            self.screen, (255, 0, 0), screen_fan_points[0], screen_fan_points[-2], 2
                        )

        except Exception:
            # 에러 발생시 간단한 부채꼴로 대체
            try:
                # 간단한 부채꼴 그리기 (polygon 실패시 선으로 대체)
                # 시야 경계선만 그리기
                edge_distance = min(max_screen_distance, self.width + self.height)

                # 시작 경계선
                start_x = screen_pos[0] + edge_distance * math.cos(start_angle)
                start_y = screen_pos[1] + edge_distance * math.sin(start_angle)
                pygame.draw.line(
                    self.screen, self.colors["sight_range"][:3], screen_pos, (start_x, start_y), 2
                )

                # 끝 경계선
                end_x = screen_pos[0] + edge_distance * math.cos(end_angle)
                end_y = screen_pos[1] + edge_distance * math.sin(end_angle)
                pygame.draw.line(
                    self.screen, self.colors["sight_range"][:3], screen_pos, (end_x, end_y), 2
                )

            except Exception:
                pass  # 완전히 실패하면 아무것도 그리지 않음

    def draw_grid(self):
        """격자 그리기"""
        grid_spacing = 5  # 월드 단위
        grid_color = (70, 70, 70)

        # 화면에 보이는 격자 범위 계산 (게임 영역에만)
        left = self.camera_x - (self.game_width // 2) / (self.world_scale * self.zoom)
        right = self.camera_x + (self.game_width // 2) / (self.world_scale * self.zoom)
        top = self.camera_y - (self.height // 2) / (self.world_scale * self.zoom)
        bottom = self.camera_y + (self.height // 2) / (self.world_scale * self.zoom)

        # 세로선
        start_x = int(left // grid_spacing) * grid_spacing
        x = start_x
        while x <= right:
            screen_x, _ = self.world_to_screen((x, 0))
            if 0 <= screen_x <= self.game_width:  # 게임 영역에만 그리기
                pygame.draw.line(self.screen, grid_color, (screen_x, 0), (screen_x, self.height))
            x += grid_spacing

        # 가로선
        start_y = int(top // grid_spacing) * grid_spacing
        y = start_y
        while y <= bottom:
            _, screen_y = self.world_to_screen((0, y))
            if 0 <= screen_y <= self.height:
                pygame.draw.line(
                    self.screen, grid_color, (0, screen_y), (self.game_width, screen_y)
                )  # 게임 영역에만 그리기
            y += grid_spacing

    def draw_checkbox(self, x, y, checked, label):
        """체크박스 그리기"""
        # 체크박스 배경
        checkbox_rect = pygame.Rect(x, y, self.checkbox_size, self.checkbox_size)
        pygame.draw.rect(self.screen, (255, 255, 255), checkbox_rect)
        pygame.draw.rect(self.screen, (0, 0, 0), checkbox_rect, 2)

        # 체크 표시
        if checked:
            pygame.draw.line(
                self.screen,
                (0, 150, 0),
                (x + 3, y + self.checkbox_size // 2),
                (x + self.checkbox_size // 2, y + self.checkbox_size - 3),
                3,
            )
            pygame.draw.line(
                self.screen,
                (0, 150, 0),
                (x + self.checkbox_size // 2, y + self.checkbox_size - 3),
                (x + self.checkbox_size - 3, y + 3),
                3,
            )

        # 라벨 텍스트
        text = self.small_font.render(label, True, (255, 255, 255))
        self.screen.blit(text, (x + self.checkbox_size + 5, y - 2))

        return checkbox_rect

    def draw_toggle_button(self):
        """패널 토글 버튼 그리기"""
        button_x = self.width - self.toggle_button_width - 5
        button_y = 5
        button_rect = pygame.Rect(
            button_x, button_y, self.toggle_button_width, self.toggle_button_height
        )

        # 버튼 배경
        button_color = (60, 60, 60) if self.panel_visible else (40, 40, 40)
        pygame.draw.rect(self.screen, button_color, button_rect)
        pygame.draw.rect(self.screen, (120, 120, 120), button_rect, 2)

        # 버튼 텍스트 (화살표)
        arrow_text = "◀" if self.panel_visible else "▶"
        text_surface = self.small_font.render(arrow_text, True, (255, 255, 255))
        text_rect = text_surface.get_rect(center=button_rect.center)
        self.screen.blit(text_surface, text_rect)

    def draw_scenario_selector(self):
        """화면 가운데 위에 시나리오 선택 UI 그리기"""
        if not self.scenario_names:
            return

        # UI 설정
        selector_height = 50
        button_width = 150
        button_height = 30
        button_spacing = 10
        total_width = (
            len(self.scenario_names) * button_width
            + (len(self.scenario_names) - 1) * button_spacing
        )

        # 화면 가운데 위에 배치
        start_x = (self.game_width - total_width) // 2
        start_y = 10

        # 배경 그리기
        bg_rect = pygame.Rect(start_x - 10, start_y - 5, total_width + 20, selector_height)
        bg_surface = pygame.Surface((bg_rect.width, bg_rect.height), pygame.SRCALPHA)
        bg_surface.fill((0, 0, 0, 180))
        self.screen.blit(bg_surface, bg_rect)
        pygame.draw.rect(self.screen, (100, 100, 100), bg_rect, 2)

        # 제목 그리기
        title_text = self.font.render("Scenario:", True, (255, 255, 255))
        title_rect = title_text.get_rect()
        title_rect.centerx = start_x + total_width // 2
        title_rect.y = start_y
        self.screen.blit(title_text, title_rect)

        # 시나리오 버튼들 그리기
        button_y = start_y + 25
        self.scenario_button_rects = {}

        for i, scenario_name in enumerate(self.scenario_names):
            button_x = start_x + i * (button_width + button_spacing)
            button_rect = pygame.Rect(button_x, button_y, button_width, button_height)

            # 선택된 시나리오는 다른 색상으로 표시
            if i == self.selected_scenario_index:
                button_color = (80, 150, 80)  # 선택된 버튼은 초록색
                text_color = (255, 255, 255)
            else:
                button_color = (60, 60, 60)  # 선택되지 않은 버튼은 회색
                text_color = (200, 200, 200)

            # 버튼 배경
            pygame.draw.rect(self.screen, button_color, button_rect)
            pygame.draw.rect(self.screen, (150, 150, 150), button_rect, 2)

            # 버튼 텍스트
            text_surface = self.small_font.render(scenario_name, True, text_color)
            text_rect = text_surface.get_rect(center=button_rect.center)
            self.screen.blit(text_surface, text_rect)

            # 클릭 감지를 위해 저장
            self.scenario_button_rects[scenario_name] = button_rect

    def draw_policy_selector(self):
        """시나리오 선택기 아래에 정책 선택 UI 그리기"""
        if not self.policy_names:
            return

        # UI 설정
        selector_height = 40
        button_width = 120
        button_height = 25
        button_spacing = 10
        total_width = (
            len(self.policy_names) * button_width + (len(self.policy_names) - 1) * button_spacing
        )

        # 시나리오 선택기 아래에 배치
        start_x = (self.game_width - total_width) // 2
        start_y = 70  # 시나리오 선택기 아래

        # 배경 그리기
        bg_rect = pygame.Rect(start_x - 10, start_y - 5, total_width + 20, selector_height)
        bg_surface = pygame.Surface((bg_rect.width, bg_rect.height), pygame.SRCALPHA)
        bg_surface.fill((0, 0, 0, 160))
        self.screen.blit(bg_surface, bg_rect)
        pygame.draw.rect(self.screen, (80, 80, 80), bg_rect, 2)

        # 제목 그리기
        title_text = self.small_font.render("AI Policy:", True, (255, 255, 255))
        title_rect = title_text.get_rect()
        title_rect.centerx = start_x + total_width // 2
        title_rect.y = start_y
        self.screen.blit(title_text, title_rect)

        # 정책 버튼들 그리기
        button_y = start_y + 18
        self.policy_button_rects = {}

        for i, policy_name in enumerate(self.policy_names):
            button_x = start_x + i * (button_width + button_spacing)
            button_rect = pygame.Rect(button_x, button_y, button_width, button_height)

            # 선택된 정책은 다른 색상으로 표시
            if i == self.selected_policy_index:
                button_color = (80, 80, 150)  # 선택된 버튼은 파란색
                text_color = (255, 255, 255)
            else:
                button_color = (50, 50, 50)  # 선택되지 않은 버튼은 진한 회색
                text_color = (180, 180, 180)

            # 버튼 배경
            pygame.draw.rect(self.screen, button_color, button_rect)
            pygame.draw.rect(self.screen, (120, 120, 120), button_rect, 1)

            # 버튼 텍스트
            text_surface = self.small_font.render(policy_name, True, text_color)
            text_rect = text_surface.get_rect(center=button_rect.center)
            self.screen.blit(text_surface, text_rect)

            # 클릭 감지를 위해 저장
            self.policy_button_rects[policy_name] = button_rect

    def draw_ui_panel(self):
        """우측 UI 패널 그리기"""
        # 패널 배경
        panel_rect = pygame.Rect(self.panel_x, 0, self.panel_width, self.height)
        pygame.draw.rect(self.screen, (40, 40, 40), panel_rect)
        pygame.draw.rect(self.screen, (80, 80, 80), panel_rect, 2)

        # 제목
        title = self.font.render("Display Options", True, (255, 255, 255))
        self.screen.blit(title, (self.panel_x + 10, 10))

        # 체크박스들
        y_start = 50
        self.checkbox_rects = {}

        options = [
            ("show_sight_range", "Sight Range"),
            ("show_attack_range", "Attack Range"),
            ("show_visible_matrix", "Visibility Matrix"),
            ("show_distance_matrix", "Distance Matrix"),
            ("show_unit_info", "Unit Info"),
            ("show_grid", "Grid"),
            ("show_obs_array", "Obs Array"),
        ]

        for i, (key, label) in enumerate(options):
            y = y_start + i * self.checkbox_spacing
            checkbox_rect = self.draw_checkbox(self.panel_x + 10, y, self.ui_panel[key], label)
            self.checkbox_rects[key] = checkbox_rect

    def handle_ui_click(self, mouse_pos):
        """UI 패널 클릭 처리"""
        for key, rect in self.checkbox_rects.items():
            if rect.collidepoint(mouse_pos):
                self.ui_panel[key] = not self.ui_panel[key]
                return True
        return False

    def handle_scenario_selector_click(self, mouse_pos):
        """시나리오 선택기 클릭 처리"""
        if not hasattr(self, "scenario_button_rects"):
            return False

        for i, (scenario_name, rect) in enumerate(self.scenario_button_rects.items()):
            if rect.collidepoint(mouse_pos):
                # 시나리오 변경
                old_index = self.selected_scenario_index
                self.selected_scenario_index = i

                if old_index != self.selected_scenario_index:
                    print(f"Scenario changed to: {scenario_name}")
                    # 자동으로 선택된 시나리오로 리셋
                    self.reset_requested = True

                return True
        return False

    def handle_policy_selector_click(self, mouse_pos):
        """정책 선택기 클릭 처리"""
        if not hasattr(self, "policy_button_rects"):
            return False

        for i, (policy_name, rect) in enumerate(self.policy_button_rects.items()):
            if rect.collidepoint(mouse_pos):
                # 정책 변경
                old_index = self.selected_policy_index
                self.selected_policy_index = i

                if old_index != self.selected_policy_index:
                    print(f"AI Policy changed to: {policy_name}")

                return True
        return False

    def handle_toggle_button_click(self, mouse_pos):
        """토글 버튼 클릭 처리"""
        button_x = self.width - self.toggle_button_width - 5
        button_y = 5
        button_rect = pygame.Rect(
            button_x, button_y, self.toggle_button_width, self.toggle_button_height
        )

        if button_rect.collidepoint(mouse_pos):
            self.panel_visible = not self.panel_visible
            self.update_game_area()
            return True
        return False

    def get_selected_policy(self):
        """현재 선택된 정책 이름 반환"""
        if 0 <= self.selected_policy_index < len(self.policy_names):
            return self.policy_names[self.selected_policy_index]
        return "Heuristic"  # 기본값

    def check_reset_request(self):
        """환경 리셋 요청 확인 및 처리"""
        if self.reset_requested:
            self.reset_requested = False
            if self.reset_fn and self.available_scenarios:
                try:
                    import jax

                    # 선택된 시나리오 사용
                    if self.scenario_names and 0 <= self.selected_scenario_index < len(
                        self.scenario_names
                    ):
                        selected_scenario_name = self.scenario_names[self.selected_scenario_index]
                        selected_scenario = self.available_scenarios[selected_scenario_name]

                        obs, state = self.reset_fn(jax.random.key(0), selected_scenario)
                        return obs, state
                    else:
                        # 기본 시나리오 사용 (fallback)
                        obs, state = self.reset_fn(jax.random.key(0), self.scenario)
                        return obs, state

                except Exception as e:
                    print(f"Environment reset failed: {e}")
                    return None, None
        return None, None

    def close(self):
        """렌더러 종료"""
        pygame.quit()


# 사용 예시 함수
def render_loop(state, fps=60, show_ranges=True, env=None, scenario=None, reset_fn=None):
    """
    실시간 렌더링 루프

    Args:
        state: 게임 상태
        fps: 프레임 레이트
        show_ranges: 공격/시야 범위 표시 여부
        env: 환경 객체
        scenario: 시나리오 객체
        reset_fn: 리셋 함수
    """
    renderer = PygameRenderer(fps=fps, env=env, scenario=scenario, reset_fn=reset_fn)

    try:
        # 초기 obs 생성

        obs = env.get_obs(state)
        key = jax.random.key(2)

        while renderer.render(state, show_ranges, obs):
            # 리셋 요청 확인 및 처리
            reset_obs, reset_state = renderer.check_reset_request()
            if reset_obs is not None and reset_state is not None:
                obs = reset_obs
                state = reset_state
                # 선택된 유닛 초기화
                renderer.selected_unit = None
                renderer.selected_unit_obs = None
                renderer.obs_scroll_offset = 0
                print("Environment reset completed!")
                continue  # 이번 프레임은 리셋만 하고 다음 프레임으로

            # 액션 준비
            actions = {}

            # 유저가 컨트롤하는 유닛들의 액션 사용
            for unit_name in env.unit_keys:
                if unit_name in renderer.user_controlled_actions:
                    actions[unit_name] = renderer.user_controlled_actions[unit_name]
                else:
                    # 선택된 정책에 따라 액션 결정
                    selected_policy = renderer.get_selected_policy()

                    if selected_policy == "Heuristic":
                        action_key, key = jax.random.split(key)
                        # Heuristic 정책 사용
                        actions[unit_name] = heuristic_policy(action_key, obs[unit_name])
                    elif selected_policy == "Idle":
                        # 아무것도 하지 않음 (IDLE 액션)
                        actions[unit_name] = jnp.array([0.0, UnitAction.IDLE])
                    else:
                        # 기본값: Heuristic 정책
                        action_key, key = jax.random.split(key)
                        actions[unit_name] = heuristic_policy(action_key, obs[unit_name])

            # 게임 스텝 실행
            obs, state, reward, done, info = step(jax.random.key(0), state, actions)
            # print(state["game_manager"].target)

            # 유저 컨트롤 액션 초기화 (한 프레임에만 적용)
            renderer.user_controlled_actions.clear()

    finally:
        renderer.close()


if __name__ == "__main__":
    # 테스트용 더미 데이터
    import jax
    import jax.numpy as jnp
    from src.maenv.physics import Transform, RigidBody, CircleCollider
    from src.maenv.tabs.tabs_battle_simulator.tabs_battle_simulator import BattleSimulator
    from src.maenv.tabs.tabs_unit_comb.tabs_unit_comb import TABSUnitComb
    from src.maenv.tabs.scenarios import default_tabs_conf, generate_scenario
    from src.maenv.tabs.tabs_battle_simulator.heuristic_policy import heuristic_policy
    import functools

    default_tabs_conf = default_tabs_conf.replace(scenario_name="10farmers")
    scenario = generate_scenario(default_tabs_conf)

    # 테스트 유닛 생성
    heuristic_policy = jax.jit(functools.partial(heuristic_policy, num_agents=20, epsilon=0.01))
    env = BattleSimulator(num_agents=20)
    reset = jax.jit(env.reset)
    # reset = env.reset
    obs, state = reset(jax.random.key(0), scenario)
    # state["game_manager"] = state["game_manager"].update_distance_matrix(state)
    step = jax.jit(env.step)
    # step = env.step
    render_loop(state, env=env, scenario=scenario, reset_fn=reset)
