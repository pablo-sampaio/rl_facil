# A "n-step SARSA" implementation
from collections import deque

import gymnasium as gym
import numpy as np

import sys
from os import path
sys.path.append( path.dirname( path.dirname( path.abspath(__file__) ) ) )

from util.qtable_helper import epsilon_greedy_random_tiebreak


# Algoritmo "n-step SARSA"
# Atenção: os espaços de estados e de ações precisam ser discretos, dados por valores inteiros
def run_nstep_sarsa(env, episodes, nsteps=1, lr=0.1, gamma=0.95, epsilon=0.1, verbose=False):
    assert isinstance(env.observation_space, gym.spaces.Discrete)
    assert isinstance(env.action_space, gym.spaces.Discrete)
    assert isinstance(nsteps, int)

    num_actions = env.action_space.n
    
    # inicializa a tabela Q com zeros
    # usar o estado como índice das linhas e a ação como índice das colunas
    Q = np.zeros(shape=(env.observation_space.n, num_actions))

    gamma_array = np.array([ gamma**i for i in range(0,nsteps)])
    gamma_power_nstep = gamma**nsteps

    # para cada episódio, guarda sua soma de recompensas (retorno não-descontado)
    sum_rewards_per_ep = []
    
    # loop principal
    for i in range(episodes):
        done = False
        sum_rewards, reward = 0, 0
        
        state, _ = env.reset()
        # escolhe a próxima ação
        action = epsilon_greedy_random_tiebreak(Q, state, epsilon)

        # históricos de: estados, ações e recompensas
        hs = deque(maxlen=nsteps)
        ha = deque(maxlen=nsteps)
        hr = deque(maxlen=nsteps)
    
        # executa 1 episódio completo, fazendo atualizações na Q-table
        while not done: 
            # realiza a ação
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            sum_rewards += reward

            # escolhe (antecipadamente) a ação do próximo estado
            next_action = epsilon_greedy_random_tiebreak(Q, next_state, epsilon)

            hs.append(state)
            ha.append(action)
            hr.append(reward)
            
            # se o histórico estiver completo com 'n' passos
            # vai fazer uma atualização no valor Q do estado mais antigo
            if len(hs) == nsteps:
                if terminated: 
                    # para estados terminais
                    V_next_state = 0
                else:
                    # para estados não-terminais -- valor máximo (melhor ação)
                    V_next_state = Q[next_state,next_action]

                # delta = (estimativa usando a nova recompensa) - estimativa antiga
                delta = ( sum(gamma_array * hr) + gamma_power_nstep * V_next_state ) - Q[hs[0],ha[0]]
                
                # atualiza a Q-table para o par (estado,ação) de n passos atrás
                Q[hs[0],ha[0]] += lr * delta

            # preparação para avançar mais um passo
            # lembrar que a ação a ser realizada já está escolhida
            state = next_state
            action = next_action
            # fim do laço por episódio

        # ao fim do episódio, atualiza o Q dos estados que restaram no histórico
        laststeps = len(hs) # pode ser inferior ao "nstep", em episódios muito curtos
        for j in range(laststeps-1,0,-1):
            hs.popleft()
            ha.popleft()
            hr.popleft()
            #delta = ( sum(gamma_array[0:j]*hr) + V_next_state ) - Q[hs[0],ha[0]]
            delta = ( sum(gamma_array[0:j]*hr) + 0 ) - Q[hs[0],ha[0]]   # assumindo que V_next_state == 0, mas isso pode não ser verdade em episódios truncados
            Q[hs[0],ha[0]] += lr * delta

        sum_rewards_per_ep.append(sum_rewards)
        
        # a cada 100 episódios, imprime informação sobre o progresso 
        if verbose and ((i+1) % 100 == 0):
            avg_reward = np.mean(sum_rewards_per_ep[-100:])
            print(f"Episode {i+1} Average Reward (last 100): {avg_reward:.3f}")

    return sum_rewards_per_ep, Q



if __name__ == "__main__":
    from gymnasium.wrappers.time_limit import TimeLimit
    from util.plot import plot_result
    from util.qtable_helper import evaluate_qtable
    from util.envs import RacetrackEnv

    #env = gym.make("FrozenLake-v1")
    #r_max = 1.0
    #env = gym.make("Taxi-v3")
    #r_max = 10.0

    env = TimeLimit(RacetrackEnv(), 300)
    r_max = 0.0

    EPISODES = 10_000
    LR = 0.1  # frozen-lake, use: 0.01
    GAMMA = 0.95
    EPSILON = 0.1
    NSTEPS = 3
  
    # Roda o algoritmo "n-step SARSA"
    rewards, qtable = run_nstep_sarsa(env, EPISODES, NSTEPS, LR, GAMMA, EPSILON, verbose=True)
    print("Últimos resultados: media =", np.mean(rewards[-20:]), ", desvio padrao =", np.std(rewards[-20:]))

    # Exibe um gráfico episódios x retornos 
    plot_result(rewards, r_max, None)

    #render_env = gym.make("FrozenLake-v1", render_mode="human")
    #render_env = gym.make("Taxi-v3", render_mode="human")
    render_env = TimeLimit(RacetrackEnv(render_mode="human"), 300)
    
    evaluate_qtable(render_env, qtable, 10, verbose=True)
    
    render_env.close()
    env.close()
