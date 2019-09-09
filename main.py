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


def time_now():
    return datetime.now().strftime('%Y%m%d_%H%M%S')


def process_raw_image(image):
    image = np.mean(image, axis=2, dtype=np.uint8)
    image = image[::2, ::2]
    return image


class TransitionTable:
    def __init__(self, maxSize=10_000):
        self.s = deque(maxlen=maxSize)
        self.a = deque(maxlen=maxSize)
        self.r = deque(maxlen=maxSize)
        self.t = deque(maxlen=maxSize)
        self.next_state = None

        self.state_fname = 'state'
        self.action_fname = 'actions.npy'
        self.reward_fname = 'rewards.npy'
        self.terminal_fname = 'terminal.npy'

        self.num_meta_bytes = 4

    def append(self, s, a, r, t):
        if self.next_state is None:
            print('Set next state before add record.')
            return

        self.s.append(self.next_state)
        self.next_state = s

        self.a.append(a)
        self.r.append(r)
        self.t.append(t)

    def __len__(self):
        return min(len(self.s), len(self.r), len(self.t))

    def save(self, path):
        table_size = len(self)
        if table_size == 0:
            return

        if not os.path.exists(path):
            os.makedirs(path)

        state_fpath = f'{path}/{self.state_fname}'
        action_fpath = f'{path}/{self.action_fname}'
        reward_fpath = f'{path}/{self.reward_fname}'
        terminal_fpath = f'{path}/{self.terminal_fname}'

        action_np = np.array(list(self.a)[:table_size], dtype=np.uint8)
        np.save(action_fpath, action_np)

        reward_np = np.array(list(self.r)[:table_size], dtype=np.float32)
        np.save(reward_fpath, reward_np)

        terminal_np = np.array(list(self.t)[:table_size])
        np.save(terminal_fpath, terminal_np)

        with open(state_fpath, mode='wb') as out_file:
            num_records = table_size.to_bytes(
                self.num_meta_bytes,
                byteorder='big',
            )
            out_file.write(num_records)
            for i in tqdm(range(table_size)):
                image_bytes = cv2.imencode('.png', self.s[i])[1].tobytes()
                size_of_image = len(image_bytes).to_bytes(
                    self.num_meta_bytes,
                    byteorder='big',
                )
                out_file.write(size_of_image)
                out_file.write(image_bytes)

            image_bytes = cv2.imencode('.png', self.next_state)[1].tobytes()
            size_of_image = len(image_bytes).to_bytes(
                self.num_meta_bytes,
                byteorder='big',
            )

            out_file.write(size_of_image)
            out_file.write(image_bytes)

    def load(self, path):
        state_fpath = f'{path}/{self.state_fname}'
        action_fpath = f'{path}/{self.action_fname}'
        reward_fpath = f'{path}/{self.reward_fname}'
        terminal_fpath = f'{path}/{self.terminal_fname}'

        action_np = np.load(action_fpath)
        reward_np = np.load(reward_fpath)
        terminal_np = np.load(terminal_fpath)

        self.s.clear()
        self.a.clear()
        self.r.clear()
        self.t.clear()

        with open(state_fpath, mode='rb') as inp_file:
            num_records = int.from_bytes(inp_file.read(self.num_meta_bytes), byteorder='big')
            for i in tqdm(range(num_records)):
                size_of_image = int.from_bytes(inp_file.read(self.num_meta_bytes), byteorder='big')
                image_bytes = inp_file.read(size_of_image)
                np_buf = np.frombuffer(image_bytes, dtype=np.uint8)
                image = cv2.imdecode(np_buf, cv2.IMREAD_UNCHANGED)

                self.s.append(image)
                self.a.append(action_np[i])
                self.r.append(reward_np[i])
                self.t.append(terminal_np[i])

            size_of_image = int.from_bytes(inp_file.read(self.num_meta_bytes), byteorder='big')
            image_bytes = inp_file.read(size_of_image)
            np_buf = np.frombuffer(image_bytes, dtype=np.uint8)
            image = cv2.imdecode(np_buf, cv2.IMREAD_UNCHANGED)

            self.next_state = image


if __name__ == '__main__':
    env_name = 'PongNoFrameskip-v4'
    env = gym.make(env_name)

    n_actions = env.action_space.n
    memory = TransitionTable()

    num_steps = 5_000

    env.seed(0)
    state = env.reset()
    # state = process_raw_image(state)
    # memory.next_state = state

    # eps_reward = 0
    # for step in tqdm(range(num_steps)):
    #     action = random.randrange(n_actions)
    #     new_state, reward, done, info = env.step(action)
    #     new_state = process_raw_image(new_state)
    #     terminal = 0 if done else 1
    #     memory.append(new_state, action, reward, terminal)

    #     new_state = cv2.resize(
    #         src=new_state,
    #         dsize=None,
    #         fx=4,
    #         fy=4,
    #         interpolation=cv2.INTER_NEAREST,
    #     )
    #     cv2.imshow('frame', new_state)
    #     cv2.waitKey(5)
    #     env.render()
    #     time.sleep(0.01)
    #     if done:
    #         env.seed(0)
    #         state = env.reset()

    # # cv2.destroyAllWindows()
    # memory_save_dir = f'{env_name}/memory/{time_now()}'
    # memory.save(memory_save_dir)

    print('Before load:', len(memory))

    memory_save_dir = f'{env_name}/memory/20190909_220144'
    memory.load(memory_save_dir)
    print('After load:', len(memory))

    for s in tqdm(memory.s):
        new_state = cv2.resize(
            src=s,
            dsize=None,
            fx=4,
            fy=4,
            interpolation=cv2.INTER_NEAREST,
        )
        cv2.imshow('frame', new_state)
        cv2.waitKey(1)

    cv2.destroyAllWindows()
