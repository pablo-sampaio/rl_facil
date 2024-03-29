import gymnasium as gym
from gymnasium import spaces
import numpy as np


class AccessControlEnv(gym.Env):
    def __init__(self):
        self.num_servers = 10
        self.server_busy_prob = 0.06
        self.priorities = [1, 2, 4, 8]
        
        self.action_space = spaces.Discrete(2)  # accept or reject
        self.observation_space = spaces.Tuple((spaces.Discrete(self.num_servers + 1), spaces.Discrete(len(self.priorities))))
        
        self.reset()
        
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.free_servers = self.num_servers
        self.time_step = 0
        return self._get_state(), None
    
    def step(self, action):
        assert self.action_space.contains(action), "Invalid action"
        
        # Update customer request
        reward = 0
        if action == 0 and self.free_servers > 0:
            self.free_servers -= 1
            reward = self.priorities[self.customer_priority_idx]

        self.time_step += 1

        # Update server states
        if self.free_servers < self.num_servers and self.np_random.random() < self.server_busy_prob:
            self.free_servers += 1       

        return self._get_state(), reward, False, False, None
    
    def _get_state(self):
        self.customer_priority_idx = self.np_random.choice(len(self.priorities))
        return (self.free_servers, self.customer_priority_idx)
    


if __name__=='__main__':
    import sys
    from os import path
    sys.path.append( path.dirname( path.dirname( path.abspath(__file__) ) ) )
    from wrappers import FromDiscreteTupleToDiscreteObs

    env = AccessControlEnv()
    env = FromDiscreteTupleToDiscreteObs(env)

    state, _ = env.reset()
    done = False
    total_reward = 0

    for i in range(30):
        action = 0  # Replace with your RL agent's action selection

        next_state, reward, termi, trunc, _ = env.step(action)
        total_reward += reward

        print(f'State {state=}, Action {action=}, Reward {reward=}')
        state = next_state

    print("Total reward:", total_reward)
