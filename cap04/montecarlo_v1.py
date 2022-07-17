
# The Monte-Carlo dual to Q-learning/SARSA
# References: 
# - Book by Sutton & Barto, chapter 5
# - Lazy programmer's implementation
# - https://www.analyticsvidhya.com/blog/2018/11/reinforcement-learning-introduction-monte-carlo-learning-openai-gym/

import gym
import numpy as np

from .util_plot import save_rewards_plot
from .util_experiments import test_greedy_Q_policy


# Esta é a política. Neste caso, escolhe uma ação com base nos valores
# da tabela Q, usando uma estratégia epsilon-greedy.
def choose_action(Q, state, num_actions, epsilon):
    if np.random.random() < epsilon:
        return np.random.randint(0, num_actions)
    else:
        return np.argmax(Q[state])


# Algoritmo Monte-Carlo de Controle, variante "toda-visita".
# Atenção: os espaços de estados e de ações precisam ser discretos, dados por valores inteiros
def run_montecarlo1(env, episodes, gamma=0.95, epsilon=0.1, render=False):
    # assert?
    num_actions = env.action_space.n
    
    # dicionário com todos os retornos descontados, para cada par (estado,ação)
    returns_history = dict()

    # inicializa a tabela Q toda com zero,
    # usar o estado como índice das linhas e a ação como índice das colunas
    Q = np.zeros(shape = (env.observation_space.n, num_actions))

    # para cada episódio, guarda sua soma de recompensas (retorno não-discontado)
    sum_rewards_per_ep = []

    # loop principal
    for i in range(episodes):
        done = False
        sum_rewards, reward = 0, 0
        ep_trajectory = []
        
        state = env.reset()
    
        # PARTE 1: executa um episódio completo
        while done != True:   
            # exibe/renderiza os passos no ambiente, durante 1 episódio a cada mil e também nos últimos 5 episódios 
            #if render and (i >= (episodes - 5) or (i+1) % 1000 == 0):
            #    env.render()
                
            # escolhe a próxima ação -- usa epsilon-greedy
            action = choose_action(Q, state, num_actions, epsilon)
        
            # realiza a ação, ou seja, dá um passo no ambiente
            next_state, reward, done, _ = env.step(action)
            
            # adiciona a tripla que representa este passo
            ep_trajectory.append( (state, action, reward) )
            
            sum_rewards += reward
            state = next_state
        
        sum_rewards_per_ep.append(sum_rewards)

        # a cada 100 episódios, imprime informação sobre o progresso 
        if (i+1) % 100 == 0:
            avg_reward = np.mean(sum_rewards_per_ep[-100:])
            print(f"Episode {i+1} Average Reward (last 100): {avg_reward:.3f}")

        # PARTE 2: atualiza Q (e a política, implicitamente)
        Gt = 0
        for (s, a, r) in reversed(ep_trajectory):
            Gt = r + gamma*Gt
            
            if returns_history.get((s,a)) is not None:
                returns_history[s,a].append(Gt)
            else:
                returns_history[s,a] = [ Gt ]
            
            # média entre todas as ocorrências de (s,a) encontradas nos episódios
            #Q[s,a] = np.mean(returns_history[s,a]) # LENTO! -> vamos melhorar
            delta = Gt - Q[s,a]
            Q[s,a] = Q[s,a] + (1/len(returns_history[s,a])) * delta

    return sum_rewards_per_ep, Q



if __name__ == "__main__":
    ENV_NAME = "Taxi-v3"
    r_max_plot = 10

    EPISODES = 3000
    GAMMA = 0.95
    EPSILON = 0.1

    env = gym.make(ENV_NAME)
    
    # Roda o algoritmo Monte-Carlo para o problema de controle (ou seja, para achar a política ótima)
    rewards, Qtable = run_montecarlo1(env, EPISODES, GAMMA, EPSILON, render=False)
    print("Últimos resultados: media =", np.mean(rewards[-20:]), ", desvio padrao =", np.std(rewards[-20:]))

    # Salva um arquivo com o gráfico de passos x retornos (não descontados)
    filename = f"results/montecarlo1-{ENV_NAME.lower()[0:8]}-ep{EPISODES}.png"
    save_rewards_plot(rewards, r_max_plot, filename)

    test_greedy_Q_policy(env, Qtable, 10, True)
    env.close()
