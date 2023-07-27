import torch

from gym.envs.classic_control.continuous_mountain_car import Continuous_MountainCarEnv

from measures import Measure
from .task import Task, RewardFunction


MagellanSparseContinuousMountainCarFlagPosition = 0.45


class MagellanSparseContinuousMountainCarFlagMeasure(Measure):
    def __call__(self, states, actions, next_states, next_state_means, next_state_vars, model):
        position = next_states[:, 0]
        measure = torch.where(position >= MagellanSparseContinuousMountainCarFlagPosition,
                              100 * torch.ones_like(position), torch.zeros_like(position))
        return measure


class MagellanSparseContinuousMountainCarFlagRewardFunction(RewardFunction):
    def __call__(self, state, action, next_state):
        return 100 if next_state[0] > MagellanSparseContinuousMountainCarFlagPosition else 0


class MagellanSparseContinuousMountainCarEnv(Continuous_MountainCarEnv):
    @property
    def tasks(self):
        t = dict()
        t['capture_flag'] = Task(measure=MagellanSparseContinuousMountainCarFlagMeasure(),
                                 reward_function=MagellanSparseContinuousMountainCarFlagRewardFunction())
        return t
