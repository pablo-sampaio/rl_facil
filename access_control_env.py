import gym
from gym import spaces
import numpy as np


class AccessControlEnv(gym.Env):
    def __init__(self):
        self.num_servers = 10
        self.server_busy_prob = 0.06
        self.priorities = [1, 2, 4, 8]
        
        self.action_space = spaces.Discrete(2)  # accept or reject
        self.observation_space = spaces.Tuple((spaces.Discrete(self.num_servers + 1), spaces.Discrete(len(self.priorities))))
        
        self.reset()
        
    def reset(self):
        self.free_servers = self.num_servers
        self.time_step = 0
        return self._get_state()
    
    def step(self, action):
        assert self.action_space.contains(action), "Invalid action"
        
        # Update customer request
        reward = 0
        if action == 0 and self.free_servers > 0:
            self.free_servers -= 1
            reward = self.priorities[self.customer_priority_idx]
        
        self.time_step += 1

        # Update server states
        if self.free_servers < self.num_servers and np.random.rand() < self.server_busy_prob:
            self.free_servers += 1       

        return self._get_state(), reward, False, {}
    
    def _get_state(self):
        self.customer_priority_idx = np.random.choice(len(self.priorities))
        return (self.free_servers, self.customer_priority_idx)
    

if __name__=='__main__':
    from util.wrappers import FromDiscreteTupleToDiscreteObs

    env = AccessControlEnv()
    #env = TupleToDiscreteWrapper(env)

    state = env.reset()
    done = False
    total_reward = 0

    for i in range(30):
        action = 0  # Replace with your RL agent's action selection

        next_state, reward, done, _ = env.step(action)
        total_reward += reward

        print(f'State {state=}, Action {action=}, Reward {reward=}')
        state = next_state

    print("Total reward:", total_reward)
