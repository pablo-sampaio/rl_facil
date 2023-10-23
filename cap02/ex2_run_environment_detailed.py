
import gymnasium as gym
import time

# o gym oferece vários ambientes, criados pela string de identificação
# descomente apenas a linha do ambiente que lhe interessa
#env = gym.make("MountainCar-v0", render_mode="human")
#env = gym.make("Taxi-v3", render_mode="human")
#env = gym.make("CartPole-v1", render_mode="human")
env = gym.make("Pendulum-v1", render_mode="human")
#env = gym.make("LunarLander-v2", render_mode="human")


# inicia um episodio no ambiente e recebe a observação inicial e o dicionário 'info'
obs, _ = env.reset()

# variável para indicar se o episódio acabou
done = False

while not done:
    # exibe visualmente o estado do ambiente (opcional)
    env.render()
    
    # escolhe uma ação aleatória, usando uma função do próprio ambiente
    # neste ponto, você pode usar um algoritmo qualquer para escolher a ação
    action = env.action_space.sample()

    # aplica a ação no ambiente e recebe uma 5-tupla
    #      obs  - a próxima observação
    #      r    - a recompensa deste passo
    #      terminated - indica se o episódio acabou naturalmente por chegar em um estado terminal
    #      truncated  - indica se o episódio acabou de forma não natural (por chegar em um limite de tempo, digamos)
    #      info - dicionário com informações extras (pode ser ignorado)
    (state, r, terminated, truncated, _) = env.step(action)
    done = terminated or truncated
    
    # às vezes, vale a pena adicionar uma espera, para deixar a renderização mais lenta
    time.sleep(0.05)

# aguarda um tempo antes de encerrar
time.sleep(2.0)

# encerra o ambiente
# necessário, principalmente, quando usar renderização
env.close()
