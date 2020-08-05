import numpy as np

# Can be used to convert rewards into discounted returns:
# ret[i] = sum of t = i to T of gamma^(t-i) * rew[t]
def discount_path(path, gamma):
    '''
    Given a "path" of items x_1, x_2, ... x_n, return the discounted
    path, i.e.
    X_1 = x_1 + h*x_2 + h^2 x_3 + h^3 x_4
    X_2 = x_2 + h*x_3 + h^2 x_4 + h^3 x_5
    etc.
    Can do (more efficiently?) w SciPy. Python here for readability
    Inputs:
    - path, list/tensor of floats
    - h, discount rate
    Outputs:
    - Discounted path, as above
    '''
    curr = 0
    rets = []
    for i in range(len(path)):
        curr = curr*gamma + path[-1-i]
        rets.append(curr)
    rets =  np.stack(list(reversed(rets)), 0)
    return rets


def get_path_indices(not_dones):
    """
    Returns list of tuples of the form:
        (agent index, time index start, time index end + 1)
    For each path seen in the not_dones array of shape (# agents, # time steps)
    E.g. if we have an not_dones of composition:
    tensor([[1, 1, 0, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 0, 1, 1, 0, 1, 1, 0, 1]], dtype=torch.uint8)
    Then we would return:
    [(0, 0, 3), (0, 3, 10), (1, 0, 3), (1, 3, 5), (1, 5, 9), (1, 9, 10)]
    """
    indices = []
    num_timesteps = not_dones.shape[0]
    last_index = 0
    for i in range(num_timesteps):
        if not_dones[i] == 0.:
            indices.append((last_index, i + 1))
            last_index = i + 1
    if last_index != num_timesteps:
        indices.append((last_index, num_timesteps))
    return indices
