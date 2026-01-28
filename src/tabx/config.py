from dataclasses import dataclass


@dataclass(frozen=True)
class TABXConfig:
    max_n_ally: int = 10  # The maximum number of ally agents
    max_n_enemy: int = 10  # The maximum number of enemy agents
    max_n_zone: int = 4  # The maximum number of zone
