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
    def __init__(
            self,
            maxSize=10_000,
            histLen=4,
            state_width=80,
            state_height=105,
    ):
        self.histLen = histLen
        self.state_width = state_width
        self.state_height = state_height

        self.s = deque(maxlen=maxSize)
        self.next_s = deque(maxlen=maxSize)
        self.a = deque(maxlen=maxSize)
        self.r = deque(maxlen=maxSize)
        self.t = deque(maxlen=maxSize)

        self.state_fname = 'state'
        self.next_state_fname = 'next_state'
        self.action_fname = 'actions.npy'
        self.reward_fname = 'rewards.npy'
        self.terminal_fname = 'terminal.npy'

        self.num_meta_bytes = 4

    def print_sizes(self):
        print('self.s:', len(self.s))
        print('self.next_s:', len(self.next_s))
        print('self.a:', len(self.a))
        print('self.r:', len(self.r))
        print('self.t:', len(self.t))

    def append(self, s, a, r, t, next_s):

        self.s.append(s)
        self.next_s.append(next_s)

        self.a.append(a)
        self.r.append(r)
        self.t.append(t)

    def __len__(self):
        return min(
            len(self.s),
            len(self.a),
            len(self.r),
            len(self.t),
            len(self.next_s),
        )

    def save(self, path):
        table_size = len(self)
        if table_size == 0:
            return

        if not os.path.exists(path):
            os.makedirs(path)

        state_fpath = f'{path}/{self.state_fname}'
        next_state_fpath = f'{path}/{self.next_state_fname}'
        action_fpath = f'{path}/{self.action_fname}'
        reward_fpath = f'{path}/{self.reward_fname}'
        terminal_fpath = f'{path}/{self.terminal_fname}'

        action_np = np.array(list(self.a)[:table_size], dtype=np.uint8)
        np.save(action_fpath, action_np)

        reward_np = np.array(list(self.r)[:table_size], dtype=np.float32)
        np.save(reward_fpath, reward_np)

        terminal_np = np.array(list(self.t)[:table_size])
        np.save(terminal_fpath, terminal_np)

        with open(state_fpath, mode='wb') as s_out_file, open(next_state_fpath, mode='wb') as next_s_out_file:
            # write the number of states at the beginning of the file
            num_records = table_size.to_bytes(
                self.num_meta_bytes,
                byteorder='big',
            )
            s_out_file.write(num_records)
            next_s_out_file.write(num_records)
            # write the size of the png image then the image itself
            for i in tqdm(range(table_size)):
                image_bytes = cv2.imencode('.png', self.s[i])[1].tobytes()
                size_of_image = len(image_bytes).to_bytes(
                    self.num_meta_bytes,
                    byteorder='big',
                )
                s_out_file.write(size_of_image)
                s_out_file.write(image_bytes)

                image_bytes = cv2.imencode('.png', self.next_s[i])[1].tobytes()
                size_of_image = len(image_bytes).to_bytes(
                    self.num_meta_bytes,
                    byteorder='big',
                )
                next_s_out_file.write(size_of_image)
                next_s_out_file.write(image_bytes)

    def load(self, path):
        state_fpath = f'{path}/{self.state_fname}'
        next_state_fpath = f'{path}/{self.next_state_fname}'
        action_fpath = f'{path}/{self.action_fname}'
        reward_fpath = f'{path}/{self.reward_fname}'
        terminal_fpath = f'{path}/{self.terminal_fname}'

        action_np = np.load(action_fpath)
        reward_np = np.load(reward_fpath)
        terminal_np = np.load(terminal_fpath)

        self.s.clear()
        self.next_s.clear()

        self.a.clear()
        self.r.clear()
        self.t.clear()

        with open(state_fpath, mode='rb') as s_inp_file, open(next_state_fpath, mode='rb') as next_s_inp_file:
            num_records = int.from_bytes(
                s_inp_file.read(self.num_meta_bytes),
                byteorder='big',
            )
            next_s_inp_file.read(self.num_meta_bytes)

            print('Loading images...')
            for i in tqdm(range(num_records)):
                size_of_image = int.from_bytes(
                    s_inp_file.read(self.num_meta_bytes),
                    byteorder='big',
                )
                image_bytes = s_inp_file.read(size_of_image)
                np_buf = np.frombuffer(image_bytes, dtype=np.uint8)
                s = cv2.imdecode(np_buf, cv2.IMREAD_UNCHANGED)

                size_of_image = int.from_bytes(
                    next_s_inp_file.read(self.num_meta_bytes),
                    byteorder='big',
                )
                image_bytes = next_s_inp_file.read(size_of_image)
                np_buf = np.frombuffer(image_bytes, dtype=np.uint8)
                next_s = cv2.imdecode(np_buf, cv2.IMREAD_UNCHANGED)

                if len(self) > 0:
                    if (self.next_s[-1] == s).all():
                        self.s.append(self.next_s[-1])
                    else:
                        self.s.append(s)
                else:
                    self.s.append(s)
                self.next_s.append(next_s)
                self.a.append(action_np[i])
                self.r.append(reward_np[i])
                self.t.append(terminal_np[i])

    def concatState(self, index):
        state = np.zeros(
            shape=(self.state_height, self.state_width, self.histLen),
            dtype=np.uint8,
        )
        for i in range(self.histLen):
            pos = index - i
            if pose < 0:
                break
            else:
                state[:, :, i] = self.s[pos]
        pass

    def sample(self, index):

        return self.s[index], self.a[index], self.r[index], self.t[index]

    def sample_batch(self, batch_size=32):
        if len(self) < batch_size:
            return


if __name__ == '__main__':
    env_name = 'PongNoFrameskip-v4'
    env = gym.make(env_name)

    n_actions = env.action_space.n
    memory = TransitionTable()

    num_steps = 5_000

    env.seed(0)
    state = env.reset()
    state = process_raw_image(state)

    eps_reward = 0
    for step in tqdm(range(num_steps)):
        action = random.randrange(n_actions)
        new_state, reward, done, info = env.step(action)
        new_state = process_raw_image(new_state)
        terminal = 1 if done else 0
        memory.append(state, action, reward, terminal, new_state)
        state = new_state

        # new_state = cv2.resize(
        #     src=new_state,
        #     dsize=None,
        #     fx=4,
        #     fy=4,
        #     interpolation=cv2.INTER_NEAREST,
        # )
        # cv2.imshow('frame', new_state)
        # cv2.waitKey(5)
        # env.render()
        # time.sleep(0.01)
        if done:
            env.seed(0)
            state = env.reset()
            state = process_raw_image(state)

    # cv2.destroyAllWindows()
    memory_save_dir = f'{env_name}/memory/{time_now()}'
    memory.save(memory_save_dir)
    # before load
    memory.print_sizes()

    org_s = list(memory.s)
    org_next_s = list(memory.next_s)
    org_a = list(memory.a)
    org_r = list(memory.r)
    org_t = list(memory.t)
    # after load
    memory.load(memory_save_dir)
    memory.print_sizes()

    load_s = list(memory.s)
    load_next_s = list(memory.next_s)
    load_a = list(memory.a)
    load_r = list(memory.r)
    load_t = list(memory.t)

    np_org_s = np.array(org_s)
    np_org_next_s = np.array(org_next_s)
    np_load_s = np.array(load_s)
    np_load_next_s = np.array(load_next_s)

    print('s', (np_org_s == np_load_s).all())
    print('next_s', (np_org_next_s == np_load_next_s).all())
