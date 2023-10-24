# A Q-Learning implementation
# Referencia: 
# - https://towardsdatascience.com/getting-started-with-reinforcement-learning-and-open-ai-gym-c289aca874f

import gymnasium as gym
import numpy as np

# Define se os algoritmos irão ou não imprimir dados parciais na saída de texto
VERBOSE = False


# Esta é a política. Neste caso, escolhe uma ação com base nos valores
# da tabela Q, usando uma estratégia epsilon-greedy.
def epsilon_greedy(Q, state, epsilon):
    Q_state = Q[state]
    num_actions = len(Q_state)
    if np.random.random() < epsilon:
        return np.random.randint(0, num_actions)
    else:
        return np.argmax(Q_state)   # em caso de empates, retorna sempre o menor índice --> mais eficiente, porém não é bom para alguns ambientes (como o FrozenLake)
        #return np.random.choice(np.where(Q_state == Q_state.max())[0]) # aleatoriza em caso de empates


# Algoritmo Q-learning
# Atenção: os espaços de estados e de ações precisam ser discretos, dados por valores inteiros
def run_qlearning(env, episodes, lr=0.1, gamma=0.95, epsilon=0.1):
    assert isinstance(env.observation_space, gym.spaces.Discrete)
    assert isinstance(env.action_space, gym.spaces.Discrete)

    num_actions = env.action_space.n
    
    # inicializa a tabela Q toda com zeros
    # usar o estado como índice das linhas e a ação como índice das colunas
    Q = np.zeros(shape = (env.observation_space.n, num_actions))

    # para cada episódio, guarda sua soma de recompensas (retorno não-descontado)
    all_episode_rewards = []
    
    # loop principal
    for i in range(episodes):
        done = False
        sum_rewards, reward = 0, 0
        
        state, _ = env.reset()
    
        # executa 1 episódio completo, fazendo atualizações na Q-table
        while not done:
            # exibe/renderiza os passos no ambiente, durante 1 episódio a cada mil e também nos últimos 5 episódios 
            #if render and ((i >= (episodes - 5) or (i+1) % 1000 == 0)):
            #    env.render()
            
            # escolhe a próxima ação -- usa epsilon-greedy
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
    
    # inicializa a tabela Q toda com zeros
    # usar o estado como índice das linhas e a ação como índice das colunas
    Q = np.zeros(shape = (env.observation_space.n, num_actions))

    # para cada episódio, guarda sua soma de recompensas (retorno não-descontado)
    all_episode_rewards = []
    
    # loop principal
    for i in range(episodes):
        done = False
        sum_rewards, reward = 0, 0
        
        state, _ = env.reset()
        
        # escolhe a próxima ação
        action = epsilon_greedy(Q, state, epsilon)
    
        # executa 1 episódio completo, fazendo atualizações na Q-table
        while not done:
            # exibe/renderiza os passos no ambiente, durante 1 episódio a cada mil e também nos últimos 5 episódios 
            #if render and ((i >= (episodes - 5) or (i+1) % 1000 == 0)):
            #    env.render()
                  
            # realiza a ação, ou seja, dá um passo no ambiente
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # escolhe a próxima ação -- usa epsilon-greedy
            next_action = epsilon_greedy(Q, state, epsilon)

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


if __name__ == "__main__":
    import sys
    from os import path
    sys.path.append( path.dirname( path.dirname( path.abspath(__file__) ) ) )

    from util.plot import plot_result
    from util.experiments import test_greedy_Q_policy

    #ENV_NAME, r_max = "FrozenLake-v1", 1.0
    ENV_NAME, r_max = "Taxi-v3", 10.0

    EPISODES = 15_000
    LR = 0.05
    GAMMA = 0.95
    EPSILON = 0.1

    env = gym.make(ENV_NAME)
    rendering_env = gym.make(ENV_NAME, render_mode='human')
    #'''
    # Roda o algoritmo Q-Learning
    returns, Qtable = run_qlearning(env, EPISODES, LR, GAMMA, EPSILON)
    print("Q-Learning - Treinamento - últimos retornos: media =", np.mean(returns[-20:]), ", desvio padrao =", np.std(returns[-20:]))

    # Mostra um gráfico de episódios x retornos não descontados. Se quiser salvar, passe o nome do arquivo como o 3o parâmetro.
    plot_result(returns, r_max, None)

    print("Q-Learning - Executando depois de treinado:")
    test_greedy_Q_policy(rendering_env, Qtable, 5)

    print(" ######### ")
    #'''  
    # Roda o algoritmo SARSA
    returns, Qtable = run_sarsa(env, EPISODES, LR, GAMMA, EPSILON, render=False)
    print("SARSA - Treinamento - últimos retornos: media =", np.mean(returns[-20:]), ", desvio padrao =", np.std(returns[-20:]))

    # Mostra um gráfico de episódios x retornos não descontados. Se quiser salvar, passe o nome do arquivo como 3o parâmetro.
    plot_result(returns, r_max, None)

    print("SARSA - Executando depois de treinado:")
    test_greedy_Q_policy(rendering_env, Qtable, 5)
    rendering_env.close()
    #'''