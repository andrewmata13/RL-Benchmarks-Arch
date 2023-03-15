import gym
import time
import torch
import numpy as np
from torch import nn
from torch.nn import functional as F
from collections import OrderedDict
from stable_baselines3 import PPO, DQN, A2C

from gym.envs.classic_control import MountainCarEnv

class ExtendedMountainCarEnv(MountainCarEnv):
    def __init__(self, network):
        MountainCarEnv.__init__(self)
        self.network = network

    def get_state(self):
        return self.state

    def reset(self, state):
        self.state = state
        return np.array(self.state, dtype=np.float32)

    def render(self):
        MountainCarEnv.render(self)

    def predict(self, state):
        action, _ = self.network.predict(state, deterministic=True)
        return action

class MountaincarEnv:
    def __init__(self):
        self.bounds = [[-1.2, 0.6], [-0.07, 0.07]]
        
        #Load in network
        if torch.cuda.is_available():
            self.model = DQN.load("../envs/MountainCar/dqn_MountainCar.model", device=0)
        else:
            self.model = DQN.load("../envs/MountainCar/dqn_MountainCar.model", device="cpu")
        self.model.set_random_seed(seed=0)
        
        self.env = ExtendedMountainCarEnv(self.model)
    
        self.actionSpace = [0, 1, 2]
        self.mask = []
        self.continuous = False
        self.safetyCheck = True


'''
env = MountainCarEnv()

#Training model
model = DQN("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=2000000)
model.save("dqn_MountainCar.model")
'''
