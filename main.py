import os
import time
import random
from collections import deque
import itertools
from datetime import datetime

from tqdm import tqdm
import numpy as np
import gym
import cv2
import matplotlib.pyplot as plt
import pandas as pd

import utils
from TransitionTable import TransitionTable
from DQNAgent import DQNAgent

if __name__ == '__main__':
    env_name = 'Pong-v0'
    # env_name = 'PongNoFrameskip-v4'
    env = gym.make(env_name)

    n_actions = env.action_space.n
    agent = DQNAgent(
        n_actions=n_actions,
    )
