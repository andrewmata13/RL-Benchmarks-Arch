import gym
import time
import torch
import numpy as np
from torch import nn
from torch.nn import functional as F
from collections import OrderedDict
from stable_baselines3 import PPO, DQN, A2C

from gym.envs.mujoco import HopperEnv

class ExtendedHopper(HopperEnv):
    def __init__(self, network):
        HopperEnv.__init__(self)
        self.network = network
        self.qnetwork = False
        
    def reset(self, state):
        # hack, set initial x-pos to 0, need to be careful when running tree sims
        q1 = np.array([0, state[0], state[1], state[2], state[3], state[4]])
        q2 = np.array([state[5], state[6], state[7], state[8], state[9], state[10]])
        self.set_state(q1, q2)
        return self._get_obs()

    def get_state(self):
        return self._get_obs()

    def predict(self, state):
        action, _ = self.network.predict(state, deterministic=True)
        return action

class Hopper:
    def __init__(self):
        self.bounds = [
            [0, 5],         
            [-2*np.pi, 2*np.pi],         
            [-2*np.pi, 2*np.pi],
            [-2*np.pi, 2*np.pi],         
            [-2*np.pi, 2*np.pi],
            [-10, 10],         
            [-10, 10],     
            [-10, 10],
            [-10, 10],     
            [-10, 10],
            [-10, 10]
        ]

        #Load in network
        if torch.cuda.is_available():
            self.model = PPO.load("../envs/Hopper/ppo_Hopper.model", device=0)
        else:
            self.model = PPO.load("../envs/Hopper/ppo_Hopper.model", device="cpu")

        self.model.set_random_seed(seed=0)
            
        self.env = ExtendedHopper(self.model)

        self.actionBounds = [[-1, 1], [-1, 1], [-1, 1]]
        self.mask = []
        self.continuous = True
        self.safetyCheck = False

'''
env = HopperEnv()

#Training model
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=3000000)
model.save("ppo_Hopper.model")
'''
