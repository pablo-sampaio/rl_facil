
import gymnasium as gym


class PunishEarlyStop(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        # if ended because the pole fell down
        if terminated:
            reward = -100
        return obs, reward, terminated, truncated, info


env = gym.make("CartPole-v1", render_mode="human")
env = PunishEarlyStop(env)

obs, _ = env.reset()
terminated = truncated = False
sum_rewards = 0.0

while not (terminated or truncated):
    env.render()
    action = env.action_space.sample()

    next_obs, reward, terminated, truncated, info = env.step(action)

    sum_rewards += reward

    obs = next_obs
    
    #time.sleep(0.1)

env.close()
print("Recompensa total:", sum_rewards)
