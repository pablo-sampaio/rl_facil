
import gym
import time

env = gym.make("MountainCar-v0")
#env = gym.make("Taxi-v3")
#env = gym.make("CartPole-v1")
#env = gym.make("Pendulum-v1")
#env = gym.make("LunarLander-v2")

obs = env.reset()
done = False
sum_rewards = 0.0

while not done:
    env.render()
    action = env.action_space.sample()
    print(obs)
    print(action)

    next_obs, reward, done, info = env.step(action)
    sum_rewards += reward

    # calcula usando "obs" e "next_obs"

    obs = next_obs
    
    #time.sleep(0.1)

env.close()
print("Recompensa total:", sum_rewards)
