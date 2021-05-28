import torch 

def lambda_return(reward_seq, value_seq, bootstrap=None, _discount=0.99, _lambda=0.95):
    """
    :param reward_seq: (seq_len, batch_size)
    :param value_seq: (seq_len, batch_size)
    :param bootstrap: (batch_size) : value of bootstrap for last state 
    :param _discount: gamma value
    :param _lambda: lambda value
    :return lambda_returns: (seq_len, batch_size)
    """
    if bootstrap == None:
        bootstrap = torch.zeros_like(value_seq[0]).to(value_seq.device)
    next_values = torch.cat([value_seq[1:], bootstrap[None]], 0)
    discount_tensor = _discount * torch.ones_like(reward_seq) 
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