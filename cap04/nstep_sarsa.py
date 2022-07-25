# A Q-Learning implementation
# Referencia: 
# - https://towardsdatascience.com/getting-started-with-reinforcement-learning-and-open-ai-gym-c289aca874f

import gym
import numpy as np

from util_plot import plot_result
from util_experiments import test_greedy_Q_policy
from collections import deque


# Esta é a política. Neste caso, escolhe uma ação com base nos valores
# da tabela Q, usando uma estratégia epsilon-greedy.
def choose_action(Q, state, num_actions, epsilon):
    if np.random.random() < epsilon:
        return np.random.randint(0, num_actions)
    else:
        return np.argmax(Q[state])   # alt. para aleatorizar empates: np.random.choice(np.where(b == bmax)[0])


# Algoritmo Q-learning, online learning (TD-learning)
# Atenção: os espaços de estados e de ações precisam ser discretos, dados por valores inteiros
def run_nstep_sarsa(env, episodes, nstep=1, lr=0.1, gamma=0.95, epsilon=0.1, render=False):
    assert isinstance(env.observation_space, gym.spaces.Discrete)
    assert isinstance(env.action_space, gym.spaces.Discrete)

    num_actions = env.action_space.n
    
    # inicializa a tabela Q com valores aleatórios de -1.0 a 0.0
    # usar o estado como índice das linhas e a ação como índice das colunas
    Q = np.random.uniform(low = -1.0, high = 0.0, 
                          size = (env.observation_space.n, num_actions))

    gamma_array = np.array([ gamma**i for i in range(0,nstep)])

    # para cada episódio, guarda sua soma de recompensas (retorno não-discontado)
    sum_rewards_per_ep = []
    
    # loop principal
    for i in range(episodes):
        done = False
        sum_rewards, reward = 0, 0
        
        state = env.reset()
        # escolhe a próxima ação -- usa epsilon-greedy
        action = choose_action(Q, state, num_actions, epsilon)

        # históricos de: estados, ações e recompensas
        hs = deque(maxlen=nstep)
        ha = deque(maxlen=nstep)
        hr = deque(maxlen=nstep)
    
        # executa 1 episódio completo, fazendo atualizações na Q-table
        while not done: 
            # exibe/renderiza os passos no ambiente, durante 1 episódio a cada mil e também nos últimos 5 episódios 
            if render and (i >= (episodes - 5) or (i+1) % 1000 == 0):
                env.render()
                        
            # realiza a ação, ou seja, dá um passo no ambiente
            next_state, reward, done, _ = env.step(action)

            hs.append(state)
            ha.append(action)
            hr.append(reward)
            
            if len(hs) == nstep:
                if done: 
                    # para estados terminais
                    V_next_state = 0
                else:
                    # escolhe (antecipadamente) a ação do próximo estado -- usa epsilon-greedy
                    action = choose_action(Q, next_state, num_actions, epsilon)
                    # para estados não-terminais -- valor máximo (melhor ação)
                    V_next_state = Q[next_state,action]

                # delta = (estimativa usando a nova recompensa) - estimativa antiga
                delta = ( sum(gamma_array*hr) + gamma**nstep * V_next_state ) - Q[hs[0],ha[0]]
                
                # atualiza a Q-table para o par (estado,ação) de n passos atrás
                Q[hs[0],ha[0]] += lr * delta
            
            # sempre executa...
            sum_rewards += reward
            state = next_state

        # ao fim do episódio, atualiza Q dos estados que restaram no histórico
        laststeps = len(hs) # pode ser inferior ao "nstep", em episódios muito curtos
        for j in range(1,laststeps):
            hs.popleft()
            ha.popleft()
            hr.popleft()
            delta = ( sum(gamma_array[0:laststeps-j]*hr) + 0 ) - Q[hs[0],ha[0]]
            Q[hs[0],ha[0]] += lr * delta

        sum_rewards_per_ep.append(sum_rewards)
        
        # a cada 100 episódios, imprime informação sobre o progresso 
        if (i+1) % 100 == 0:
            avg_reward = np.mean(sum_rewards_per_ep[-100:])
            print(f"Episode {i+1} Average Reward (last 100): {avg_reward:.3f}")

    return sum_rewards_per_ep, Q



if __name__ == "__main__":
    ENV_NAME = "Taxi-v3"
    r_max_plot = 10

    EPISODES = 20000
    LR = 0.01
    GAMMA = 0.95
    EPSILON = 0.1

    env = gym.make(ENV_NAME)
    
    # Roda o algoritmo Q-Learning
    rewards, Qtable = run_nstep_sarsa(env, EPISODES, 4, LR, GAMMA, EPSILON, render=False)
    print("Últimos resultados: media =", np.mean(rewards[-20:]), ", desvio padrao =", np.std(rewards[-20:]))

    # Salva um arquivo com o gráfico de episódios x retornos (não descontados)
    filename = f"results/qlearning-{ENV_NAME.lower()[0:8]}-ep{EPISODES}-lr{LR}.png"
    plot_result(rewards, r_max_plot, None)

    test_greedy_Q_policy(env, Qtable, 10, True)
    env.close()
