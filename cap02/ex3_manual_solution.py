
import gymnasium as gym

env = gym.make("MountainCar-v0", render_mode="human")

obs, _ = env.reset()
done = False
sum_rewards = 0.0

while not done:
    env.render()

    posx = obs[0]
    vel = obs[1]

    # uma política determinística criada manualmente
    if vel > 0:
        action = 2  # mover para a direita
    else:
        action = 1  # deixar livre

    next_obs, reward, terminated, truncated, _ = env.step(action)
    done = terminated or truncated

    sum_rewards += reward
    obs = next_obs

print("Recompensa total:", sum_rewards)
env.close()