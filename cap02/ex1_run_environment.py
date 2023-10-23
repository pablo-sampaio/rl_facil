
import gymnasium as gym

env = gym.make("MountainCar-v0", render_mode="human")
#env = gym.make("Taxi-v3", render_mode="human")
#env = gym.make("CartPole-v1", render_mode="human")
#env = gym.make("Pendulum-v1", render_mode="human")
#env = gym.make("LunarLander-v2", render_mode="human")

env.reset()
done = False

while not done:
    env.render()
    action = env.action_space.sample()
    _, _, terminated, truncated, _ = env.step(action)
    done = terminated or truncated

env.close()

