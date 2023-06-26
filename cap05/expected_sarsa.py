import gym
import numpy as np


# esta função pode ser usada para converter um array "x" de valores
# numéricos quaisquer em probabilidades
def softmax_probs(x):
    x = x - np.max(x)
    x = np.exp(x)
    x = x / np.sum(x)
    return x

# escolhe uma ação da Q-table usando uma estratégia softmax
def softmax_choice(Q, state):
    probs = softmax_probs(Q[state])
    return np.random.choice(len(probs), p=probs)

# define as probabilidades de escolher uma ação usando uma estratégia epsilon-greedy
# um pouco mais detalhada (que considera os empates no valor máximo de Q)
def epsilon_greedy_probs(Q, state, num_actions, epsilon):
    # probabilidade que todas as ações têm de ser escolhidas nas decisões exploratórias (não-gulosas)
    non_greedy_action_probability = epsilon / num_actions

    # conta quantas ações estão empatadas com o valor máximo de Q neste estado
    q_max = np.max(Q[state, :])
    greedy_actions = 0
    for i in range(num_actions):
        if Q[state][i] == q_max:
            greedy_actions += 1
    
    # probabilidade de cada ação empatada com Q máximo: 
    # probabilidade de ser escolhida de forma gulosa (greedy) + probabilidade de ser escolhida de forma exploratória
    greedy_action_probability = ((1 - epsilon) / greedy_actions) + non_greedy_action_probability

    # prepara a lista de probabilidades: cada índice tem a probabilidade da ação daquele índice
    probs = []
    for i in range(num_actions):
        if Q[state][i] == q_max:
            probs.append(greedy_action_probability)
        else:
            probs.append(non_greedy_action_probability)
    return probs

def epsilon_greedy_choice(Q, state, epsilon=0.1):
    num_actions = len(Q[state])
    probs = epsilon_greedy_probs(Q, state, num_actions, epsilon)
    return np.random.choice(num_actions, p=probs)


# Algoritmo Expected-SARSA
# Atenção: os espaços de estados e de ações precisam ser discretos, dados por valores inteiros
def run_expected_sarsa(env, episodes, lr=0.1, gamma=0.95, epsilon=0.1, render=False):
    assert isinstance(env.observation_space, gym.spaces.Discrete)
    assert isinstance(env.action_space, gym.spaces.Discrete)

    num_actions = env.action_space.n
    
    # inicializa a tabela Q com valores aleatórios de -1.0 a 0.0
    # usar o estado como índice das linhas e a ação como índice das colunas
    Q = np.random.uniform(low = -1.0, high = 0.0, 
                          size = (env.observation_space.n, num_actions))

    # para cada episódio, guarda sua soma de recompensas (retorno não-descontado)
    sum_rewards_per_ep = []
    
    # loop principal
    for i in range(episodes):
        done = False
        sum_rewards, reward = 0, 0
        
        state = env.reset()
    
        # executa 1 episódio completo, fazendo atualizações na Q-table
        while not done:
            # exibe/renderiza os passos no ambiente, durante 1 episódio a cada mil e também nos últimos 5 episódios 
            if render and (i >= (episodes - 5) or (i+1) % 1000 == 0):
                env.render()
            
            # escolhe a próxima ação -- usa epsilon-greedy
            action = epsilon_greedy_choice(Q, state, epsilon)
            #action = softmax_choice(Q, state)  # bad results!

            # realiza a ação, ou seja, dá um passo no ambiente
            next_state, reward, done, _ = env.step(action)

            if done: 
                # para estados terminais
                V_next_state = 0
            else:
                # para estados não-terminais -- valor esperado
                p_next_actions = epsilon_greedy_probs(Q, next_state, num_actions, epsilon)
                #p_next_actions = softmax_probs(Q[next_state_num])
                V_next_state = np.sum( p_next_actions * Q[next_state] ) 

            # atualiza a Q-table
            # delta = (estimativa usando a nova recompensa) - estimativa antiga
            delta = (reward + gamma * V_next_state) - Q[state,action]
            Q[state,action] = Q[state,action] + lr * delta
            
            sum_rewards += reward
            state = next_state

        # salva o retorno do episódio que encerrou
        sum_rewards_per_ep.append(sum_rewards)
        
        # a cada 100 episódios, imprime informação sobre o progresso 
        if (i+1) % 100 == 0:
            avg_reward = np.mean(sum_rewards_per_ep[-100:])
            print(f"Episode {i+1} Average Reward (last 100): {avg_reward:.3f}")

    return sum_rewards_per_ep, Q



if __name__ == "__main__":
    import sys
    from os import path
    sys.path.append( path.dirname( path.dirname( path.abspath(__file__) ) ) )

    from util.plot import plot_result
    from util.experiments import test_greedy_Q_policy

    ENV_NAME, r_max = "FrozenLake-v1", 1.0
    #ENV_NAME, r_max = "Taxi-v3", 10.0

    EPISODES = 10000
    LR = 0.2
    GAMMA = 0.95
    EPSILON = 0.1

    env = gym.make(ENV_NAME)
    
    # Roda o algoritmo Expected-SARSA
    returns, Qtable = run_expected_sarsa(env, EPISODES, LR, GAMMA, EPSILON, render=False)
    print("Últimos resultados: media =", np.mean(returns[-20:]), ", desvio padrao =", np.std(returns[-20:]))

    # Mostra um gráfico de episódios x retornos (não descontados)
    plot_result(returns, r_max, None, window=50)

    test_greedy_Q_policy(env, Qtable, 5, True)
    env.close()
