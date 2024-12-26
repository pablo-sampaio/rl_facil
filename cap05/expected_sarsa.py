import gymnasium as gym
import numpy as np


# esta função pode ser usada para converter um array "x" de valores
# numéricos quaisquer em probabilidades
def softmax_probs(Q, state):
    values = Q[state]
    values = values - np.max(values)
    values = np.exp(values)
    values = values / np.sum(values)
    return values

# escolhe uma ação da Q-table usando uma estratégia softmax
def softmax_choice(Q, state):
    probs = softmax_probs(Q[state])
    return np.random.choice(len(probs), p=probs)

# define as probabilidades de escolher uma ação usando uma estratégia epsilon-greedy
# um pouco mais detalhada (que considera os empates no valor máximo de Q)
def epsilon_greedy_probs(Q, state, epsilon):
    Q_state = Q[state]
    num_actions = len(Q_state)
    q_max = np.max(Q_state)
    
    non_greedy_action_probability = epsilon / num_actions
    greedy_actions = np.sum(Q_state == q_max)
    
    greedy_action_probability = ((1 - epsilon) / greedy_actions) + non_greedy_action_probability
    
    probs = np.where(Q_state == q_max, greedy_action_probability, non_greedy_action_probability)
    #probs = np.as_numpy([ greedy_action_probability if Q_state[i]==q_max else non_greedy_action_probability 
    #                        for i in range(num_actions) ])
    
    return probs

def epsilon_greedy_choice(Q, state, epsilon=0.1):
    num_actions = len(Q[state])
    probs = epsilon_greedy_probs(Q, state, epsilon)
    return np.random.choice(num_actions, p=probs)


# Algoritmo Expected-SARSA
# Atenção: os espaços de estados e de ações precisam ser discretos, dados por valores inteiros
def run_expected_sarsa(env, episodes, lr=0.1, gamma=0.95, epsilon=0.1):
    assert isinstance(env.observation_space, gym.spaces.Discrete)
    assert isinstance(env.action_space, gym.spaces.Discrete)

    num_actions = env.action_space.n
    
    # inicializa a tabela Q com valores aleatórios pequenos (para evitar empates)
    # usar o estado como índice das linhas e a ação como índice das colunas
    Q = np.random.uniform(low=-0.01, high=+0.01, size=(env.observation_space.n, num_actions))

    # para cada episódio, guarda sua soma de recompensas (retorno não-descontado)
    all_episode_rewards = []
    
    # loop principal
    for i in range(episodes):
        done = False
        sum_rewards, reward = 0, 0
        
        state, _ = env.reset()
    
        # executa 1 episódio completo, fazendo atualizações na Q-table
        while not done:
            # escolhe a próxima ação com a 'behavior policy'
            action = epsilon_greedy_choice(Q, state, epsilon)
            #action = softmax_choice(Q, state)  # bad results!

            # realiza a ação, ou seja, dá um passo no ambiente
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            if terminated: 
                # para estados terminais
                V_next_state = 0
            else:
                # para estados não-terminais -- valor esperado
                # atualiza conforme a 'target policy'
                p_next_actions = epsilon_greedy_probs(Q, next_state, epsilon)
                #p_next_actions = softmax_probs(Q[next_state_num])
                V_next_state = np.sum( p_next_actions * Q[next_state] ) 

            # atualiza a Q-table
            # delta = (estimativa usando a nova recompensa) - estimativa antiga
            delta = (reward + gamma * V_next_state) - Q[state,action]
            Q[state,action] = Q[state,action] + lr * delta
            
            sum_rewards += reward
            state = next_state

        # salva o retorno do episódio que encerrou
        all_episode_rewards.append(sum_rewards)
        
        # a cada 100 episódios, imprime informação sobre o progresso 
        if (i+1) % 100 == 0:
            avg_reward = np.mean(all_episode_rewards[-100:])
            print(f"Episode {i+1} Average Reward (last 100): {avg_reward:.3f}")

    return all_episode_rewards, Q



if __name__ == "__main__":
    import sys
    from os import path
    sys.path.append( path.dirname( path.dirname( path.abspath(__file__) ) ) )

    from util.plot import plot_result
    from util.qtable_helper import evaluate_qtable_policy

    ENV_NAME, r_max = "FrozenLake-v1", 1.0
    #ENV_NAME, r_max = "Taxi-v3", 10.0

    EPISODES = 10000
    LR = 0.2
    GAMMA = 0.95
    EPSILON = 0.1

    env = gym.make(ENV_NAME)
    
    # Roda o algoritmo Expected-SARSA
    returns, Qtable = run_expected_sarsa(env, EPISODES, LR, GAMMA, EPSILON)
    print("Últimos resultados: media =", np.mean(returns[-20:]), ", desvio padrao =", np.std(returns[-20:]))

    # Mostra um gráfico de episódios x retornos (não descontados)
    plot_result(returns, r_max, window=50)

    rendering_env = gym.make(ENV_NAME, render_mode='human')
    evaluate_qtable_policy(rendering_env, Qtable, 5, verbose=True)
    env.close()
