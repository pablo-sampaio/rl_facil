
import gym
import time

env = gym.make("MountainCar-v0")

obs = env.reset()
done = False
sum_rewards = 0.0

while not done:
    env.render()

    # se a velocidade for negativa, acelera para a esquerda    
    # senão, acelera para a direita
    if obs[1] < 0:
        action = 0
    else:
        action = 2

    next_obs, reward, done, _ = env.step(action)
    sum_rewards += reward
    obs = next_obs
    
    time.sleep(0.01)

print("Recompensa total:", sum_rewards)
