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

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten
from tensorflow.keras.optimizers import RMSprop

import utils
from TransitionTable import TransitionTable


class DQNAgent:
    def __init__(
        self,
        n_actions=4,
        memory_size=100_000,
        state_width=80,
        state_height=105,
        histLen=4,
        learning_rate=0.00001,
        eps_start=1.0,
        eps_end=0.1,
        eps_endt=1_000_000,
    ):
        self.n_actions = n_actions
        self.learning_rate = learning_rate
        self.histLen = histLen
        self.state_width = state_width
        self.state_height = state_height
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_endt = eps_endt

        self.target_net = self.createNetwork(
            name='target',
            input_shape=(self.state_height, self.state_width, self.histLen),
            n_actions=self.n_actions,

        )
        self.policy_net = self.createNetwork(
            name='target',
            input_shape=(self.state_height, self.state_width, self.histLen),
            n_actions=self.n_actions,
        )
        self.policy_net.set_weights(self.target_net.get_weights())

        self.memory = TransitionTable(
            maxSize=memory_size,
            histLen=self.histLen,
            state_width=self.state_width,
            state_height=self.state_height,
        )

    def calc_eps(self, stepNum):
        eps_range = self.eps_start - self.eps_end
        eps_delta = 1 - (stepNum / self.eps_endt)
        eps_delta = max(0, eps_delta)
        eps = eps_end + eps_delta
        return eps

    def createNetwork(
        self,
        name,
        input_shape=(105, 80, 4),
        n_actions=4,
        lr=0.0001,
    ):
        model = Sequential(name=name, layers=[
            Conv2D(
                name=f'{name}_conv2d_1',
                filters=32,
                kernel_size=5,
                strides=3,
                activation='relu',
                # padding='valid',
                input_shape=(*input_shape,),
                data_format='channels_last',
            ),
            Conv2D(
                name=f'{name}_conv2d_2',
                filters=64,
                kernel_size=5,
                strides=3,
                activation='relu',
            ),
            Conv2D(
                name=f'{name}_conv2d_3',
                filters=64,
                kernel_size=3,
                # strides=2,
                activation='relu',
            ),
            # Conv2D(
            #     name=f'{name}_conv2d_4',
            #     filters=64,
            #     kernel_size=3,
            #     strides=2,
            #     activation='relu',
            # ),
            Flatten(name=f'{name}_flatten_1'),
            Dense(
                name=f'{name}_dense_1',
                units=256,
                activation='relu',
            ),
            Dense(
                name=f'{name}_dense_2',
                units=n_actions,
                activation='linear',
            ),
        ])

        optimizer = RMSprop(lr=lr)
        model.compile(
            loss='mse',
            optimizer=optimizer,
            metrics=['accuracy', 'mse'],
        )
        model.summary()

        return model
