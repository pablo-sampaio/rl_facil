
import gym
import time

env = gym.make("MountainCar-v0")
#env = gym.make("Taxi-v3")
#env = gym.make("CartPole-v1")
#env = gym.make("Pendulum-v1")
#env = gym.make("LunarLander-v2")

TOTAL_STEPS = 5

i = 0
obs = env.reset()
trajectory = []  # ou rollout -> são os detalhes do episódio

done = False

for i in range(0,TOTAL_STEPS):
    #env.render()
    action = env.action_space.sample()

    next_obs, reward, done, info = env.step(action)

    trajectory.append( (obs, action, reward) )

    obs = next_obs
    time.sleep(0.1)

env.close()

print("(STATE, ACTION, REWARD)")
print(trajectory)
