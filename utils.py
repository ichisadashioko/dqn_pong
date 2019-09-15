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
            x2 = (col+1) * width
            y1 = row * height
            y2 = (row+1) * height
            im[y1:y2, x1:x2] = image_batch[row, :, :, col]

    return im