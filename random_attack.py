import sys
import numpy as np

sys.path.insert(0,'..')
sys.path.insert(1,'../..')

from envs.Cartpole.CartpoleEnv import CartpoleEnv
from envs.InvertedPendulum.InvertedPendulumEnv import InvertedPendulum
from envs.DoublePendulum.DoublePendulumEnv import InvertedDoublePendulum
from envs.MountainCar.MountainCarEnv import MountaincarEnv
from envs.LunarLander.LunarLanderEnv import LunarLanderEnv
from envs.Hopper.HopperEnv import Hopper
from envs.Walker2D.Walker2DEnv import Walker2D

def calculateError(clean, noisy, bounds):
    errs = []
    for i in range(len(clean)):
        errs.append(abs(clean[i] - noisy[i]) / (bounds[i][1] - bounds[i][0] + 1e-8))
    return max(errs)

class RandomAttack:
    def __init__(self, environment):
        self.environment = environment
        self.env = environment.env
        self.bounds = environment.bounds

    def attack(self, current_state, epsilon):
        cleanState = current_state
        ranges = []
        for i, b in enumerate(self.bounds):
            margin = ((b[1] - b[0]) * epsilon * 1e-2)
            ranges.append([cleanState[i] - margin, cleanState[i] + margin])

        perturbed_state = []
        for rang in ranges:
            perturbed_state.append(np.random.uniform(rang[0], rang[1]))

        return perturbed_state
            
    def run_simulations(self, starting_bounds, epsilon, maxDepth, numSim=1000):
        num_unsafe_found = 0
        
        for _ in range(numSim):
            stepNum = 0

            # Pick random starting state
            start_state = []
            for b in starting_bounds:
                start_state.append(np.random.uniform(b[0], b[1]))

            obs = self.env.reset(start_state)
            done = False
            
            while not done and stepNum <= maxDepth:
                action = self.env.predict(obs)
                obs, reward, done, _ = self.env.step(action)

                # Attack the observation
                noisyObs = self.attack(obs, epsilon)
                assert calculateError(obs, noisyObs, self.bounds) <= epsilon*1e-2
                obs = noisyObs

                #environment.env.render()
                stepNum += 1

            # For Mountain Car
            if not done and self.environment.safetyCheck:
                assert stepNum == maxDepth + 1
                num_unsafe_found += 1  
            elif done and not self.environment.safetyCheck:
                num_unsafe_found += 1

            print(stepNum)
                
        return num_unsafe_found / numSim
                
'''
# Cartpole
environment = CartpoleEnv()
#starting_bounds = [[-0.05, 0.05], [-0.05, 0.05], [-0.05, 0.05], [-0.05, 0.05]]
starting_bounds = [[0, 0], [0, 0], [0, 0], [0, 0]]
'''

'''
# Mountain Car
environment = MountaincarEnv()
starting_bounds = [[-0.53, -0.47], [0, 0]]
'''

'''
# Inverted Pendulum
environment = InvertedPendulum()
starting_bounds = [[-0.01, 0.01], [0, 0], [0, 0], [0, 0]]
#starting_bounds = [[0, 0], [0, 0], [0, 0], [0, 0]]
'''

'''
# Double Pendulum
environment = InvertedDoublePendulum()
#starting_bounds = [[-0.05, 0.05], [-0.05, 0.05], [-0.05, 0.05], [-0.05, 0.05], [-0.05, 0.05],
#                   [0, 0], [0, 0], [0, 0],[0, 0],[0, 0],[0, 0]]

starting_bounds = [[-0.05, 0.05], [0, 0], [0, 0], [0, 0], [0, 0],
                   [0, 0], [0, 0], [0, 0],[0, 0],[0, 0],[0, 0]]
'''

'''
# Lunar Lander
environment = LunarLanderEnv()
starting_bounds = [[-0.5, 0.5], [0.4, 0.8], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]]
'''

'''
# Hopper
environment = Hopper()
input_noise = 5e-3
#starting_bounds = [[1.25 - input_noise, 1.25 + input_noise], [-input_noise, input_noise], [-input_noise, input_noise], [-input_noise, input_noise], [-input_noise, input_noise], [-input_noise, input_noise], [-input_noise, input_noise], [-input_noise, input_noise], [-input_noise, input_noise], [-input_noise, input_noise], [-input_noise, input_noise]]
starting_bounds = [[1.25 - input_noise, 1.25 + input_noise],[0, 0],[0, 0],[0, 0],[0, 0],[0, 0],[0, 0],[0, 0],[0, 0],[0, 0],[0, 0]]
'''

'''
# Walker2D
input_noise = 5e-3
environment = Walker2D()
#starting_bounds = [[[1.25 - input_noise, 1.25 + input_noise]], [[-input_noise, input_noise]]*16]
starting_bounds = [[[1.25 - input_noise, 1.25 + input_noise]], [[0, 0]]*16]
starting_bounds = starting_bounds[0] + starting_bounds[1]
#print(starting_bounds)
'''

rand_attack = RandomAttack(environment)
attack_success = rand_attack.run_simulations(starting_bounds, 1, 200)
print("Attack Success:", attack_success)
