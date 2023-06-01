
import gym

env = gym.make("MountainCar-v0")
#env = gym.make("Taxi-v3")
#env = gym.make("CartPole-v1")
#env = gym.make("Pendulum-v1")
#env = gym.make("LunarLander-v2")

env.reset()
done = False

while not done:
    env.render()
    action = env.action_space.sample()
    _, _, done, _ = env.step(action)

env.close()

