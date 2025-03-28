from typing import Union
from stable_baselines3.common.type_aliases import Schedule
import torch.nn as nn
import torch as th
import random
import numpy as np

def get_schedule_fn(initial_value: Union[float, str]) -> Schedule:
    """
    Linear learning rate schedule.

    :param initial_value: (float or str)
    :return: (function)
    """
    # Force conversion to float
    initial_value_ = float(initial_value)

    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0
        :param progress_remaining: (float)
        :return: (float)
        """
        return progress_remaining * initial_value_

    return func


def set_random_seed(seed: int, using_cuda: bool = False) -> None:
    """
    Seed the different random generators.

    :param seed:
    :param using_cuda:
    """
    # Seed python RNG
    random.seed(seed)
    # Seed numpy RNG
    np.random.seed(seed)
    # seed the RNG for all devices (both CPU and CUDA)
    th.manual_seed(seed)
    # force to set the seed for generating random numbers on all GPUs
    th.cuda.manual_seed_all(seed)

    if using_cuda:
        # Deterministic operations for CuDNN, it may impact performances
        th.backends.cudnn.deterministic = True
        th.backends.cudnn.benchmark = False


# Necessary for my KFAC implementation.
class AddBias(nn.Module):
    def __init__(self, bias):
        super(AddBias, self).__init__()
        self._bias = nn.Parameter(bias.unsqueeze(1))

    def forward(self, x):
        if x.dim() == 2:
            bias = self._bias.t().view(1, -1)
        else:
            bias = self._bias.t().view(1, -1, 1, 1)

        return x + bias
    
    
def blockwise_eigh(matrix, block_size, epsilon, device):
    n = matrix.shape[0]
    eigenvalues_list = []
    eigenvectors_list = []

    for i in range(0, n, block_size):
        block = matrix[i : i+block_size, i : i+block_size]
        block_size_actual = block.shape[0]

        d, Q = th.linalg.eigh(block + epsilon * th.eye(block_size_actual, device=device))
        # d, Q = th.linalg.eigh(block)

        eigenvalues_list.append(d)
        eigenvectors_list.append(Q)

    eigenvalues = th.cat(eigenvalues_list)
    eigenvectors = th.block_diag(*eigenvectors_list)

    return eigenvalues, eigenvectors