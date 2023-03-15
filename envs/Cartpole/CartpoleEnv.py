import gym
import time
import torch
import numpy as np
from torch import nn
from torch.nn import functional as F
from collections import OrderedDict
from stable_baselines3 import PPO, DQN, A2C

from gym.envs.classic_control import CartPoleEnv

class ExtendedCartPoleEnv(CartPoleEnv):
    def __init__(self, network):
        CartPoleEnv.__init__(self)
        self.tau = 0.02
        self.network = network

    def get_state(self):
        return self.state
        
    def reset(self, state):
        self.steps_beyond_done = None
        self.state = state
        return np.array(self.state, dtype=np.float32)

    def render(self):  
        CartPoleEnv.render(self)
    
    def predict(self, state):
        action, _ = self.network.predict(state, deterministic=True)
        return action
        
class CartpoleEnv:
    def __init__(self):
        self.bounds = [[-4.8, 4.8], [-5, 5], [-0.418, 0.418], [-5, 5]]

        #Load in network
        if torch.cuda.is_available():
            self.model = DQN.load("../envs/Cartpole/dqn_Cartpole.model", device=0)
        else:
            self.model = DQN.load("../envs/Cartpole/dqn_Cartpole.model", device="cpu")

        self.model.set_random_seed(seed=0)
        
        self.env = ExtendedCartPoleEnv(self.model)
    
        self.actionSpace = [0, 1]
        self.mask = []
        self.continuous = False
        self.safetyCheck = False
