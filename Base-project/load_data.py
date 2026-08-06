"""Script containg functions to set random seeds and add noise to the data"""

import math
import random

import numpy as np
import torch


def set_random_seeds(seed):
    """Function to seed everything.

    Args:
        seed (int): Random seed to control torch, cuda and np seeds
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def add_noise(Y, stoc, randnum):
    """Function to add noise to the output, Y.

    Args:
        Y (tensor): Output data.
        stoc (float): Proportion of noise to be added.
        randnum (int): Random seed to control which individuals are selected.

    Returns:
        tensor: OUtput data with noise added.
    """

    # Set random seed
    random.seed(randnum)

    # Selects the number of positive values that need to be switched
    num1s = np.sum(Y)
    num10 = math.ceil(stoc * num1s)
    which1 = np.where(Y == 1)[0]
    which0 = np.where(Y == 0)[0]

    # Randomly selects the correct number of 1s/0s that will be switched to 0s/1s
    which10 = random.sample(list(which1), num10)
    which01 = random.sample(list(which0), num10)

    # Switches the 1s to 0s and 0s to 1s
    Y[which10] = 0
    Y[which01] = 1
    return Y
