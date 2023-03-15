import gym
import time
import torch
import numpy as np
from torch import nn
from torch.nn import functional as F
from collections import OrderedDict
from stable_baselines3 import PPO, DQN, A2C

from gym.envs.mujoco import InvertedDoublePendulumEnv

class ExtendedDoublePendulumEnv(InvertedDoublePendulumEnv):
    def __init__(self, network):
        InvertedDoublePendulumEnv.__init__(self)
        self.network = network
        
    def reset(self, state):
        decons = np.array([state[0], np.arcsin(state[1]), np.arcsin(state[2]), state[7], state[6], state[5]])
        q1 = np.array([decons[0], decons[1], decons[2]])
        q2 = np.array([decons[5], decons[4], decons[3]])
        self.set_state(q1, q2)
        return self._get_obs()

    def get_state(self):
        return self._get_obs()
    
    def predict(self, state):
        action, _ = self.network.predict(state, deterministic=True)
        action = action[0]
        return action

class InvertedDoublePendulum:
    def __init__(self):
        self.bounds = [
            [-5, 5],         #cart pos
            [-1, 1],         #sin of angles
            [-1, 1],
            [-1, 1],         #cos of angles
            [-1, 1],
            [-5, 5],         #cart vel
            [-0.5, 0.5],     #pole ang vels
            [-0.5, 0.5],
            [-100, 100],     #Contraint forces
            [-100, 100],
            [-100, 100]
        ]

        #Load in network
        if torch.cuda.is_available():
            self.model = PPO.load("../envs/DoublePendulum/ppo_DoublePendulum.model", device=0)
        else:
            self.model = PPO.load("../envs/DoublePendulum/ppo_DoublePendulum.model", device="cpu")

        self.model.set_random_seed(seed=0)
            
        self.env = ExtendedDoublePendulumEnv(self.model)

        self.actionBounds = [[-1, 1]]
        self.mask = [3, 4, 6, 7, 8, 9, 10]
        self.continuous = True
        self.safetyCheck = False

'''
env = InvertedDoublePendulumEnv()

#Training model
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=300000)
model.save("ppo_DoublePendulum.model")
'''
