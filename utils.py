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


def sign(x):
    if x == 0:
        return 1
    return int(x / abs(x))


def time_now():
    return datetime.now().strftime('%Y%m%d_%H%M%S')


def process_raw_image(image):
    image = np.mean(image, axis=2, dtype=np.uint8)
    image = image[::2, ::2]
    return image


def preview_concat_frames(memory, index):
    state = memory.concatState(index, memory.s)
    next_state = memory.concatState(index, memory.next_s)

    preview_frame = np.zeros(
        shape=(memory.state_height * 2, memory.state_width * memory.histLen),
        dtype=np.uint8,
    )

    for i in range(memory.histLen):
        preview_frame[
            0:memory.state_height,
            memory.state_width * i:memory.state_width * (i + 1)
        ] = state[:, :, i]
        preview_frame[
            memory.state_height:memory.state_height * 2,
            memory.state_width * i:memory.state_width * (i + 1)
        ] = next_state[:, :, i]
    return preview_frame


def preview_batch_images(image_batch):
    """
    Image batch should have shape like (32, 105, 80, 4).
    """
    num_rows, height, width, num_cols = image_batch.shape
    im = np.zeros(shape=(
        num_rows * height,
        num_cols * width
    ), dtype=np.uint8)

    for row in range(num_rows):
        for col in range(num_cols):
            x1 = col * width
            x2 = (col + 1) * width
            y1 = row * height
            y2 = (row + 1) * height
            im[y1:y2, x1:x2] = image_batch[row, :, :, col]

    return im


def evaluate_agent(agent, env, num_episodes=10):
    logs = []
    for _ in tqdm(range(num_episodes)):
        ep_a = []
        ep_r = []
        ep_t = []
        ep_q = []

        state_memory = deque(maxlen=4)
        s = env.reset()
        s = process_raw_image(s)
        state_memory.append(s)

        done = False
        while not done:
            batch = np.zeros(shape=(1, 105, 80, 4), dtype=np.uint8)
            for i in range(len(state_memory)):
                batch[0, :, :, i] = state_memory[-(i + 1)]
            batch = batch.astype(np.float32) / 255.0
            q = agent.target_net.predict(batch)[0]
            action = np.argmax(q)
            s2, reward, done, info = env.step(action)
            s2 = process_raw_image(s2)
            state_memory.append(s2)

            ep_a.append(action)
            ep_r.append(reward)
            ep_t.append(done)
            ep_q.append(q)
        log = {
            'ep_a': ep_a,
            'ep_r': ep_r,
            'ep_t': ep_t,
            'ep_q': ep_q,
        }
        logs.append(log)

    return logs
