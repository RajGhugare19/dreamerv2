class RewardFunction:
    def __call__(self, state, action, next_state):
        raise NotImplementedError


class Task:
    def __init__(self, measure, reward_function):
        self.measure = measure
        self.reward_function = reward_function
