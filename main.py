import os
import time
import random

from tqdm import tqdm
import numpy as np
import gym
import cv2
import matplotlib.pyplot as plt
import pandas as pd

def process_raw_image(image):
    image = np.mean(image, axis=2, dtype=np.uint8)
    image = image[::2, ::2]
    return image

if __name__ == '__main__':
    env_name = 'PongNoFrameskip-v4'
    env = gym.make(env_name)

    n_actions = env.action_space.n

    num_steps = 5_000

    env.seed(0)
    state = env.reset()
    for step in tqdm(range(num_steps)):
        action = random.randrange(n_actions)
        new_state, reward, done, info = env.step(action)
        new_state = process_raw_image(new_state)
        new_state = cv2.resize(new_state, dsize=None, fx=4, fy=4, interpolation=cv2.INTER_NEAREST)
        cv2.imshow('frame', new_state)
        cv2.waitKey(5)
        # env.render()
        # time.sleep(0.01)
        if done:
            env.seed(0)
            state = env.reset()
        
    cv2.destroyAllWindows()