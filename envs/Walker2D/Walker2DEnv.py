import gym
import time
import torch
import numpy as np
from torch import nn
from torch.nn import functional as F
from collections import OrderedDict
from stable_baselines3 import PPO, DQN, A2C

from gym.envs.mujoco import Walker2dEnv

class ExtendedWalker2D(Walker2dEnv):
    def __init__(self, network):
        Walker2dEnv.__init__(self)
        self.network = network
        
    def reset(self, state):
        q1 = np.array([0]+state[:8])
        q2 = np.array(state[8:])
        self.set_state(q1, q2)
        return self._get_obs()

    def get_state(self):
        return self._get_obs()
    
    def predict(self, state):
        action, _ = self.network.predict(state, deterministic=True)
        return action

class Walker2D:
    def __init__(self):
        self.bounds = [
            [0, 5],         
            [-2*np.pi, 2*np.pi],         
            [-2*np.pi, 2*np.pi],
            [-2*np.pi, 2*np.pi],         
            [-2*np.pi, 2*np.pi],
            [-2*np.pi, 2*np.pi],
            [-2*np.pi, 2*np.pi],         
            [-2*np.pi, 2*np.pi],
            [-10, 10],         
            [-10, 10],     
            [-10, 10],
            [-10, 10],     
            [-10, 10],
            [-10, 10],
            [-10, 10],     
            [-10, 10],
            [-10, 10]
        ]

        #Load in network  
        if torch.cuda.is_available():
            self.model = PPO.load("../envs/Walker2D/ppo_Walker2D.model", device=0)
        else:
            self.model = PPO.load("../envs/Walker2D/ppo_Walker2D.model", device="cpu")

        self.model.set_random_seed(seed=0)
            
        self.env = ExtendedWalker2D(self.model)

        self.actionBounds = [[-1, 1], [-1, 1], [-1, 1], [-1, 1], [-1, 1], [-1, 1]]
        self.mask = []
        self.continuous = True
        self.safetyCheck = False

'''
env = Walker2dEnv()

#Training model
model = PPO("MlpPolicy", env, batch_size=32, n_steps=512, learning_rate=5.05041e-05, ent_coef=0.000585045, clip_range=0.1, n_epochs=20, max_grad_norm=1, vf_coef=0.871923, verbose=1)
model.learn(total_timesteps=1000000)
model.save("ppo_Walker2D.model")
'''
