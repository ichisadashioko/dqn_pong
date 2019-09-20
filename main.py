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

env_name = 'Pong-v0'
# env_name = 'PongNoFrameskip-v4'
env = gym.make(env_name)

n_actions = env.action_space.n
agent = DQNAgent(
    n_actions=n_actions,
)

learn_start = 10_000

num_steps = 100_000

env.seed(0)
state = env.reset()
state = utils.process_raw_image(state)
eps_reward = 0

for i in tqdm(range(num_steps)):
    # epsilon greedy
    eps = agent.calc_eps(i)
    if random.random() < eps or len(agent.memory) == 0:
        action = random.randrange(agent.n_actions)
    else:
        state_batch = np.array([agent.recentState()], dtype=np.uint8)
        action = np.argmax(agent.target_net.predict(state_batch)[0])

    new_state, reward, done, info = env.step(action)
    new_state = utils.process_raw_image(new_state)
    terminal = 1 if done else 0
    agent.memory.append(state, action, reward, terminal, new_state)

    state = new_state
    if done:
        env.seed(0)
        state = env.reset()
        state = utils.process_raw_image(state)

    if i > learn_start:
        agent.train()
