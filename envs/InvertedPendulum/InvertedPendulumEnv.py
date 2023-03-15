import gym
import time
import torch
import numpy as np
from torch import nn
from torch.nn import functional as F
from collections import OrderedDict
from stable_baselines3 import PPO, DQN, A2C

from gym.envs.mujoco import InvertedPendulumEnv

class ExtendedInvertedPendulumEnv(InvertedPendulumEnv):
    def __init__(self):
        InvertedPendulumEnv.__init__(self)
        if torch.cuda.is_available():
            self.network = PPO.load("../envs/InvertedPendulum/ppo_InvertedPendulum.model", device=0)
        else:
            self.network = PPO.load("../envs/InvertedPendulum/ppo_InvertedPendulum.model", device="cpu")
            
        self.network.set_random_seed(seed=0)

    def get_state(self):
        return self._get_obs()
        
    def reset(self, state):
        q1 = np.array([state[0], state[1]])
        q2 = np.array([state[2], state[3]])
        self.set_state(q1, q2)
        return self._get_obs()

    def predict(self, state):
        action, _ = self.network.predict(state, deterministic=True)
        action = action[0]
        return action
    
class InvertedPendulum:
    def __init__(self):
        self.bounds = [[-4.8, 4.8], [-5, 5], [-2, 2], [-5, 5]]

        #Load in network
        if torch.cuda.is_available():
            self.model = PPO.load("../envs/InvertedPendulum/ppo_InvertedPendulum.model", device=0)
        else:
            self.model = PPO.load("../envs/InvertedPendulum/ppo_InvertedPendulum.model", device="cpu")

        self.model.set_random_seed(seed=0)

        self.env = ExtendedInvertedPendulumEnv()

        self.actionBounds = [[-3, 3]]
        self.mask = []
        self.continuous = True
        self.safetyCheck = False
        

'''
#Training model
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=150000)
model.save("ppo_InvertedPendulum.model")
'''

