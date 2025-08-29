from dataclasses import dataclass
import tyro
from tqdm import tqdm
@dataclass
class Config:
    seed: int = 42
    n_env: int = 64 # the number of environments to run in parallel
    rollout_step: int = 1024 # the number of rollouts to run in parallel
    gamma: float = 0.99 # the discount factor
    lamda: float = 0.95 # the lambda for GAE
    clip_value: float = 1.0 # the clip value for PPO
    clip_ratio: float = 0.05 # the clip ratio for PPO
    entropy_coef: float = 0.01 # the entropy coefficient
    ppo_epochs: int = 10 # the number of epochs to update the policy and value function
    max_grad_norm: float = 0.5 # the maximum gradient norm
    lr: float = 1e-3 # the learning rate
    layer_dim = 256

    scenario: str = '8archer_vs_1mammoth_1healer'

    save_path: str = '/save'
    gpu_id: int = 3

    total_env_step: int = int(2e8)
    log_step: int = 100



if __name__ == '__main__':
    config = tyro.cli(Config)

    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = str(config.gpu_id)

    from src.baseline.mappo import MAPPO
    from src.maenv.tabs.wrappers.wrapper import BattleSimulatorAutoResetWrapper, BattleSimulatorLogWrapper
    from src.maenv.tabs.scenarios import default_tabs_conf, generate_scenario
    from src.maenv.tabs.tabs_battle_simulator.tabs_battle_simulator import BattleSimulator

    import jax
    from functools import partial
    import wandb

    config.action_dim = 6
    config.obs_dim = 158
    config.state_dim = 158 * 8

    

    import hashlib
    import json
    
    # Create a hash of the config for unique folder naming
    config_dict = {k: v for k, v in vars(config).items() if not k.startswith('_')}
    config_str = json.dumps(config_dict, sort_keys=True)
    config_hash = hashlib.md5(config_str.encode()).hexdigest()[:8]
    config.save_path = f'/save/{config.scenario}_{config_hash}'

    wandb.init(project="battle_simulator_mappo", config=config)


    default_tabs_conf = default_tabs_conf.replace(scenario_name=config.scenario)
    scenario = generate_scenario(default_tabs_conf)

    env = BattleSimulatorLogWrapper(BattleSimulatorAutoResetWrapper(BattleSimulator(max_n_ally=int(scenario.ally_unit_comp.sum().item()), max_n_enemy=int(scenario.enemy_unit_comp.sum().item())), scenario))

    mappo = MAPPO(config, env)
    train_state = mappo.init_train_state()
    train_fn = jax.jit(partial(mappo.train, step = config.log_step))

    for step in tqdm(range(config.total_env_step // (config.n_env * config.rollout_step * config.log_step))):
        result = train_fn(train_state)
        train_state, train_info = result
        mappo.save_state(train_state, config.save_path + f'/{step}')
        
        train_info["returned_episode_returns"] = train_info["returned_episode_returns"][:, :, 0].mean(axis=1).flatten()
        train_info["returned_episode_lengths"] = train_info["returned_episode_lengths"][:, :, 0].mean(axis=1).flatten()
        train_info["returned_episode_wins"] = train_info["returned_episode_wins"][:, :, 0].mean(axis=1).flatten()
        
        for i in range(config.log_step):
            wandb.log(jax.tree.map(lambda x: x[i], train_info))
        




