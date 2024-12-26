# A Q-Learning implementation
# Referencia: 
# - https://towardsdatascience.com/getting-started-with-reinforcement-learning-and-open-ai-gym-c289aca874f

import gymnasium as gym
import numpy as np

import sys
from os import path
sys.path.append( path.dirname( path.dirname( path.abspath(__file__) ) ) )

from util.qtable_helper import epsilon_greedy

# Define se os algoritmos irão ou não imprimir dados parciais na saída de texto
VERBOSE = False


# Esta é a política. Neste caso, escolhe uma ação com base nos valores
# da tabela Q, usando uma estratégia epsilon-greedy.
'''def epsilon_greedy(Q, state, epsilon):
    Q_state = Q[state]
    num_actions = len(Q_state)
    if np.random.random() < epsilon:
        return np.random.randint(0, num_actions)
    else:
        # em caso de empates, retorna sempre o menor índice -- mais eficiente, porém não é bom para alguns ambientes
        return np.argmax(Q_state)
'''

# Algoritmo Q-learning
# Atenção: os espaços de estados e de ações precisam ser discretos, dados por valores inteiros
def run_qlearning(env, episodes, lr=0.1, gamma=0.95, epsilon=0.1):
    assert isinstance(env.observation_space, gym.spaces.Discrete)
    assert isinstance(env.action_space, gym.spaces.Discrete)

    num_actions = env.action_space.n
    
    # inicializa a tabela Q com valores aleatórios pequenos (para evitar empates)
    # usar o estado como índice das linhas e a ação como índice das colunas
    Q = np.random.uniform(low=-0.01, high=+0.01, size=(env.observation_space.n, num_actions))
    #Q = np.zeros(shape = (env.observation_space.n, num_actions)) # ruim, porque inicia com vários empates

    # para cada episódio, guarda sua soma de recompensas (retorno não-descontado)
    all_episode_rewards = []
    
    # loop principal
    for i in range(episodes):
        done = False
        sum_rewards, reward = 0, 0
        
        state, _ = env.reset()
    
        # executa 1 episódio completo, fazendo atualizações na Q-table
        while not done:
            # escolhe a próxima ação -- usa epsilon-greedy
            #action = epsilon_greedy_random_tiebreak(Q, state, epsilon)
            action = epsilon_greedy(Q, state, epsilon)
        
            # realiza a ação, ou seja, dá um passo no ambiente
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            if terminated: 
                # para estados terminais
                V_next_state = 0
            else:
                # para estados não-terminais -- valor máximo (melhor ação)
                V_next_state = np.max(Q[next_state])

            # atualiza a Q-table
            # delta = (estimativa usando a nova recompensa) - estimativa antiga
            delta = (reward + gamma * V_next_state) - Q[state,action]
            Q[state,action] = Q[state,action] + lr * delta
            
            sum_rewards += reward
            state = next_state

        #epsilon = np.exp(-0.005*i)

        all_episode_rewards.append(sum_rewards)
        # a cada 100 episódios, imprime informação sobre o progresso 
        if VERBOSE and ((i+1) % 100 == 0):
            avg_reward = np.mean(all_episode_rewards[-100:])
            print(f"Episode {i+1} Average Reward (last 100): {avg_reward:.3f}")

    return all_episode_rewards, Q


# Algoritmo SARSA
# Atenção: os espaços de estados e de ações precisam ser discretos, dados por valores inteiros
def run_sarsa(env, episodes, lr=0.1, gamma=0.95, epsilon=0.1):
    assert isinstance(env.observation_space, gym.spaces.Discrete)
    assert isinstance(env.action_space, gym.spaces.Discrete)

    num_actions = env.action_space.n
    
    # inicializa a tabela Q com valores aleatórios pequenos (para evitar empates)
    # usar o estado como índice das linhas e a ação como índice das colunas
    Q = np.random.uniform(low=-0.01, high=+0.01, size=(env.observation_space.n, num_actions))
    #Q = np.zeros(shape = (env.observation_space.n, num_actions))
    
    # para cada episódio, guarda sua soma de recompensas (retorno não-descontado)
    all_episode_rewards = []
    
    # loop principal
    for i in range(episodes):
        done = False
        sum_rewards, reward = 0, 0
        
        state, _ = env.reset()
        
        # escolhe a próxima ação
        #action = epsilon_greedy_random_tiebreak(Q, state, epsilon)
        action = epsilon_greedy(Q, state, epsilon)
    
        # executa 1 episódio completo, fazendo atualizações na Q-table
        while not done:
            # realiza a ação, ou seja, dá um passo no ambiente
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # escolhe a próxima ação -- usa epsilon-greedy
            #next_action = epsilon_greedy_random_tiebreak(Q, next_state, epsilon)
            next_action = epsilon_greedy(Q, next_state, epsilon)

            if terminated: 
                # para estados terminais
                V_next_state = 0
            else:
                # para estados não-terminais -- valor da próxima ação (já escolhida)
                V_next_state = Q[next_state,next_action]

            # atualiza a Q-table
            # delta = (estimativa usando a nova recompensa) - estimativa antiga
            delta = (reward + gamma * V_next_state) - Q[state,action]
            Q[state,action] = Q[state,action] + lr * delta
            
            sum_rewards += reward
            state = next_state
            action = next_action

        #epsilon = np.exp(-0.005*i)

        all_episode_rewards.append(sum_rewards)
        # a cada 100 episódios, imprime informação sobre o progresso 
        if VERBOSE and ((i+1) % 100 == 0):
            avg_reward = np.mean(all_episode_rewards[-100:])
            print(f"Episode {i+1} Average Reward (last 100): {avg_reward:.3f}")

    return all_episode_rewards, Q


# function to show the greedy policy for each state of FrozenLake
# actions are letters: U(p), D(own), L(eft), R(ight)
def show_frozenlake_greedy_policy(Q):
    policy = np.array([[' ']*4 for _ in range(4)], dtype='object')
    for state in range(16):
        if state in [5,7,11,12,15]:
            policy[state//4, state%4] = '*'
        else:
            policy[state//4, state%4] = ['L','D','R','U'][np.argmax(Q[state])]
    print(policy)


if __name__ == "__main__":
    from util.plot import plot_result
    from util.qtable_helper import evaluate_qtable_policy

    ENV_NAME, r_max = "FrozenLake-v1", 1.0
    #ENV_NAME, r_max = "Taxi-v3", 10.0

    EPISODES = 15_000
    LR = 0.05
    GAMMA = 0.95
    EPSILON = 0.1

    env = gym.make(ENV_NAME)
    rendering_env = gym.make(ENV_NAME, render_mode='human')
    #'''
    # Roda o algoritmo Q-Learning
    returns1, Qtable1 = run_qlearning(env, EPISODES, LR, GAMMA, EPSILON)
    print("Q-Learning - Treinamento - últimos retornos: media =", np.mean(returns1[-20:]), ", desvio padrao =", np.std(returns1[-20:]))

    # Mostra um gráfico de episódios x retornos não descontados. Se quiser salvar, passe o nome do arquivo como o 3o parâmetro.
    plot_result(returns1, r_max, window=30)

    if ENV_NAME == "FrozenLake-v1":
        show_frozenlake_greedy_policy(Qtable1)

    print("Q-Learning - Executando depois de treinado:")
    evaluate_qtable_policy(rendering_env, Qtable1, 10)

    #'''  
    print(" ######### ")
    
    # Roda o algoritmo SARSA
    returns2, Qtable2 = run_sarsa(env, EPISODES, LR, GAMMA, EPSILON)
    print("SARSA - Treinamento - últimos retornos: media =", np.mean(returns2[-20:]), ", desvio padrao =", np.std(returns2[-20:]))

    # Mostra um gráfico de episódios x retornos não descontados. Se quiser salvar, passe o nome do arquivo como 3o parâmetro.
    plot_result(returns2, r_max, window=30)

    if ENV_NAME == "FrozenLake-v1":
        show_frozenlake_greedy_policy(Qtable2)
    
    print("SARSA - Executando depois de treinado:")
    evaluate_qtable_policy(rendering_env, Qtable2, 10)
    #'''

    rendering_env.close()