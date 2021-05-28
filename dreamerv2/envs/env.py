from dreamerv2.utils.wrapper_utils import TimeLimit, ActionRepeat
import gym


class GymEnv():
    def __init__(self, env_buffer):
        self.env_name = env_buffer['env_name']
        self.env_seed = env_buffer['env_seed']
        self.time_limit = env_buffer['time_limit']
        self.action_repeat = env_buffer['action_repeat']

    def _wrap_env(self):
        env = TimeLimit(gym.make(self.env_name), self.time_limit)
        env = ActionRepeat(env)
        env.seed(self.env_seed)
        return env 
    