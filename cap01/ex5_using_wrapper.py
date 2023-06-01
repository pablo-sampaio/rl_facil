
import gym


class PunishEarlyStop(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
    
    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        # if ended because the pole fell down
        #if done and self.env._elapsed_steps < self.env._max_episode_steps:
        if 'TimeLimit.truncated' not in info:
            reward = -100
        return obs, reward, done, info



env = gym.make("CartPole-v1")
env = PunishEarlyStop(env)
gym.__version__
obs = env.reset()
done = False
sum_rewards = 0.0

while not done:
    env.render()
    action = env.action_space.sample()

    next_obs, reward, done, info = env.step(action)
    sum_rewards += reward

    obs = next_obs
    
    #time.sleep(0.1)

env.close()
print("Recompensa total:", sum_rewards)
