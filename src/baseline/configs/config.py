from dataclasses import dataclass


@dataclass(frozen=True)
class PPOConfig:
    gamma: float = 0.99  # The discount factor
    lamda: float = 0.95  # The lambda for GAE
    clip_value: float = 0.2  # The clip value for PPO
    clip_ratio: float = 0.2  # The clip ratio for PPO
    entropy_coef: float = 0.0  # The entropy coefficient
    epochs: int = 5
    layer_dim: int = 256
    rollout_step: int = 512
    learning_scheduler: bool = True
    max_grad_norm: float = 0.25
    seed: int = 42
    lr: float = 1e-4
    n_env: int = 128  # the number of environments to run in parallel
    batch_size: int = 128


@dataclass(frozen=True)
class PQNConfig:
    eps_start: float = 1.0
    eps_finish: float = 0.01
    eps_decay: float = 0.2
    batch_size: float = 4
    num_epochs: int = 4
    max_grad_norm: float = 0.25
    lr: float = 1e-4
    gamma: float = 0.99
    n_env: int = 128
    lamda: float = 0.95
    seed: int = 42
    rollout_step: int = 32
    learning_scheduler: bool = True
    reward_scale: float = 50.0
    layer_dim: int = 256
    batch_norm = False
