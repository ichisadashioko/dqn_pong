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

if __name__ == '__main__':
    env_name = 'Pong-v0'
    # env_name = 'PongNoFrameskip-v4'
    env = gym.make(env_name)

    n_actions = env.action_space.n
    memory = TransitionTable()

    #=== Populate memory - START ===#
    # num_steps = 5_000

    # env.seed(0)
    # state = env.reset()
    # state = utils.process_raw_image(state)

    # eps_reward = 0
    # for step in tqdm(range(num_steps)):
    #     action = random.randrange(n_actions)
    #     new_state, reward, done, info = env.step(action)
    #     new_state = utils.process_raw_image(new_state)
    #     terminal = 1 if done else 0
    #     memory.append(state, action, reward, terminal, new_state)
    #     state = new_state

    #     if done:
    #         env.seed(0)
    #         state = env.reset()
    #         state = utils.process_raw_image(state)

    #     # frame = cv2.resize(
    #     #     src=state,
    #     #     dsize=None,
    #     #     fx=4,
    #     #     fy=4,
    #     #     interpolation=cv2.INTER_NEAREST,
    #     # )

    #     # env.render()

    #     # cv2.imshow('frame', frame)
    #     # cv2.waitKey(5)

    # # cv2.destroyAllWindows()
    #=== Populate memory - END ===#

    # memory_save_dir = f'{env_name}/memory/{utils.time_now()}'
    memory_save_dir = f'{env_name}/memory/20190914_165410'

    memory.load(memory_save_dir)

    # visualize frames concatenation
    rand = False
    for i in range(1100, len(memory)):
        if rand:
            index = random.randrange(len(memory))
        else:
            index = i
        print(f'Preview index: {index}')
        if memory.t[index]:
            print(f'Terminal index: {index}')
        preview_frame = utils.preview_concat_frames(memory, index)
        frame = cv2.resize(preview_frame, None, fx=4, fy=4, interpolation=cv2.INTER_NEAREST)
        cv2.imshow('frame', frame)
        k = cv2.waitKey(0) & 0xff
        if k == ord('q') or k == 32:
            break
        elif k == ord('r'):
            rand = True
    cv2.destroyAllWindows()
