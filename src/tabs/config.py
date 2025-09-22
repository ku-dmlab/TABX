from dataclasses import dataclass
from src.tabs.constants import ALL_UNIT_NAMES


@dataclass(frozen=True)
class TABSConfig:
    scenario_name: str = "2F1K2A1H_low"  # The predefined scenario name
    max_num_units: int = len(ALL_UNIT_NAMES)  # The maximum number of unit types
    max_field_height: int = 4  # The maximum height size of battle field
    max_field_width: int = 5  # The maximum width size of battle field
    max_n_ally: int = 10  # The maximum number of ally agents
    max_n_enemy: int = 10  # The maximum number of enemy agents


@dataclass(frozen=True)
class TABSHeuristicConfig:
    epsilon: float = 0.1
    aggressive_threshold: float = 0.3
    rotate_noise_scale: float = 0.5
    healer_rotate_noise_scale: float = 0.1
    healer_aggressive_threshold: float = 0.85
    assasin_speed: float = 1.4
    ranger_attack_range: float = 10.0
