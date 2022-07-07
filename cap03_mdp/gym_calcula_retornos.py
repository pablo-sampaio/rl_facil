
import gym
import time

#env = gym.make("MountainCar-v0")
#env = gym.make("Taxi-v3")
env = gym.make("CartPole-v1")
#env = gym.make("Pendulum-v1")
#env = gym.make("LunarLander-v2")

TOTAL_STEPS = 5
GAMMA = 0.9

i = 0
obs = env.reset()
trajectory = []  # ou rollout -> são os detalhes do episódio

done = False

# atenção: o correto seria ir até o fim...
for i in range(0,TOTAL_STEPS):
    #env.render()
    action = env.action_space.sample()

    next_obs, reward, done, info = env.step(action)

    trajectory.append( (obs, action, reward) )

    obs = next_obs
    time.sleep(0.1)

trajectory.append( (obs, None, None) )
env.close()

print("(STATE, ACTION, REWARD)")
print(trajectory)


# calcula o retorn (G) do episódio completo
G = 0.0
for (s, a, r) in reversed(trajectory):   
    G = r + GAMMA*G
    
print(G)


# calcula os retornos a cada passo (G_i, para cada i=0...n) do episódio completo
G = 0.0
all_Gs = [ G ]
for (s, a, r) in reversed(trajectory):   
    G = r + GAMMA*G
    all_Gs.insert(0, G)

print(all_Gs)
