import torch 

def lambda_return(reward_seq, value_seq, bootstrap, discount_tensor, _lambda=0.95):
    """
    :param reward_seq: (seq_len, batch_size)
    :param value_seq: (seq_len, batch_size)
    :param bootstrap: (batch_size) : value of bootstrap for last state 
    :param _discount: gamma value
    :param _lambda: lambda value
    :return lambda_returns: (seq_len, batch_size)
    """
    next_values = torch.cat([value_seq[1:], bootstrap[None]], 0)
    inputs = reward_seq + discount_tensor * next_values * (1 - _lambda)
    last = bootstrap
    indices = reversed(range(len(inputs)))
    outputs = []
    for index in indices:
        inp, disc = inputs[index], discount_tensor[index]
        last = inp + disc*_lambda*last
        outputs.append(last)
        
    outputs = list(reversed(outputs))
    outputs = torch.stack(outputs, 0)
    returns = outputs
    return returns


def compute_return(
                reward: torch.Tensor,
                value: torch.Tensor,
                discount: torch.Tensor,
                bootstrap: torch.Tensor,
                lambda_: float):
        """
        Compute the discounted reward for a batch of data.
        reward, value, and discount are all shape [horizon - 1, batch, 1] (last element is cut off)
        Bootstrap is [batch, 1]
        """
        next_values = torch.cat([value[1:], bootstrap[None]], 0)
        target = reward + discount * next_values * (1 - lambda_)
        timesteps = list(range(reward.shape[0] - 1, -1, -1))
        outputs = []
        accumulated_reward = bootstrap
        for t in timesteps:
            inp = target[t]
            discount_factor = discount[t]
            accumulated_reward = inp + discount_factor * lambda_ * accumulated_reward
            outputs.append(accumulated_reward)
        returns = torch.flip(torch.stack(outputs), [0])
        return returns