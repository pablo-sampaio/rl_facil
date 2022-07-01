
import gym
import time

# descomente apenas a linha do ambiente que lhe interessa
env = gym.make("MountainCar-v0")
#env = gym.make("Taxi-v3")
#env = gym.make("CartPole-v1")
#env = gym.make("Pendulum-v1")
#env = gym.make("LunarLander-v2")

# reinicia um episodio no ambiente, e retorna a observação inicial
obs = env.reset()

# variável para indicar se o episódio acabou
done = False

while not done:
    # exibe visualmente o estado do ambiente (opcional)
    env.render()
    
    # você pode escolher uma ação qualquer
    action = env.action_space.sample()

    # aplica a ação no ambiente
    (obs, r, done, _) = env.step(action)
    
    # alguns ambientes renderizam muito rapidamente, então vale a pena adicionar uma espera
    #time.sleep(0.05)

# encerra o ambiente, principalmente, se você usou renderização
env.close()
