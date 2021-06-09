import numpy as np
import torch.nn as nn
from dataclasses import dataclass, field
from typing import Any, Tuple, Dict

@dataclass
class MinAtarConfig():

    #common
    env : str                                           
    obs_shape: Tuple                                            
    action_size: int
    obs_dtype: np.dtype = np.uint8
    action_dtype: np.dtype = np.float32
    train_steps: int = 1000000
    pixel: bool = True
    max_episodes: int = 500
    action_repeat: int = 1
    time_limit: int = 50
    batch_size: int = 50
    seq_len: int = 50
    eval_every: int = 1e4
    eval_episode: int = 1
    eval_render: bool = False
    clip_rewards: str = 'tanh'
    train_every: int = 5
    collect_intervals: int = 50
    seed_episodes: int = 5
    grad_clip: float = 100.0
    eps: float = 1e-5
    wd: float = 1e-6
    lr: Dict = field(default_factory=lambda:{'model':1e-3, 'actor':4e-5, 'critic':1e-4})

    #reprsentation
    deter_size: int = 100
    class_size: int = 10
    category_size: int = 10
    embedding_size: int = 100
    rssm_node_size: int = 100
    rssm_state_type: str = 'discrete'
    obs_encoder: Dict = field(default_factory=lambda:{'layers':3, 'node_size':100, 'dist':None, 'activation':nn.ELU, 'kernels':[4, 4, 4, 4]})
    obs_decoder: Dict = field(default_factory=lambda:{'layers':3, 'node_size':100, 'dist':'normal', 'activation':nn.ELU, 'kernels':[5, 5, 6, 6]})
    reward: Dict = field(default_factory=lambda:{'layers':3, 'node_size':100, 'dist':'normal', 'activation':nn.ELU})
    discount: Dict = field(default_factory=lambda:{'layers':3, 'node_size':100, 'dist':'binary', 'activation':nn.ELU, 'use':True})
    loss_scale: Dict = field(default_factory=lambda:{'kl':0.1, 'reward':1.0, 'discount':5.0})
    kl: Dict = field(default_factory=lambda:{'use_kl_balance':False, 'kl_balance_scale':0.8, 'use_free_nats':False, 'free_nats':0.0})

    #behaviour
    discount_: float = 0.99
    lambda_: float = 0.95
    horizon: int = 8
    actor: Dict = field(default_factory=lambda:{'layers':3, 'node_size':100, 'dist':'one_hot', 'activation':nn.ELU})
    critic: Dict = field(default_factory=lambda:{'layers':3, 'node_size':100, 'dist':None, 'activation':nn.ELU})
    expl: Dict = field(default_factory=lambda:{'train_noise':0.4, 'eval_noise':0.0, 'expl_min':0.05, 'expl_decay':10000.0, 'expl_type':'epsilon_greedy'})
    actor_grad: str ='reinforce'
    actor_grad_mix: int = 0.0
    actor_entropy_scale: float = 1e-3
    use_slow_target: float = True
    slow_target_update: float = 100
    slow_target_fraction: float = 1.0