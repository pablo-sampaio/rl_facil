
import gymnasium as gym
import time

#env = gym.make("MountainCar-v0", render_mode="human")
#env = gym.make("Taxi-v3", render_mode="human")
#env = gym.make("CartPole-v1", render_mode="human")
#env = gym.make("Pendulum-v1", render_mode="human")
env = gym.make("LunarLander-v2", render_mode="human")

TOTAL_STEPS = 5
GAMMA = 0.9

i = 0
obs, _ = env.reset()
trajectory = []  # ou rollout -> são os detalhes do episódio

done = False

# atenção: o correto seria ir até o fim...
for i in range(0,TOTAL_STEPS):
    env.render()
    action = env.action_space.sample()

    next_obs, reward, terminated, truncated, info = env.step(action)

    trajectory.append( (obs, action, reward) )

    obs = next_obs
    time.sleep(0.1)

# poderia adicionar este para guardar o estado final, mas não será útil
#trajectory.append( (obs, None, None) )
env.close()

print("(STATE, ACTION, REWARD)")
print(trajectory)


# calcula o retorn (G) do episódio completo
G = 0.0
for (s, a, r) in reversed(trajectory):   
    G = r + GAMMA*G
    
print(G)


# calcula os retornos a cada passo (G_t, para cada t=0...n) do episódio
Gt = 0.0
all_Gts = [ Gt ]
for (s, a, r) in reversed(trajectory): 
    Gt = r + GAMMA*Gt
    all_Gts.insert(0, Gt)

print(all_Gts)
