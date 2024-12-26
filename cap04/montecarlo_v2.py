
# The Monte-Carlo dual to Q-learning/SARSA
# References: 
# - Book by Sutton & Barto, chapter 5
# - Lazy programmer's implementation
# - https://www.analyticsvidhya.com/blog/2018/11/reinforcement-learning-introduction-monte-carlo-learning-openai-gym/

import gymnasium as gym
import numpy as np


# Esta é a política. Neste caso, escolhe uma ação com base nos valores
# da tabela Q, usando uma estratégia epsilon-greedy.
def choose_action(Q, state, num_actions, epsilon):
    if np.random.random() < epsilon:
        return np.random.randint(0, num_actions)
    else:
        return np.argmax(Q[state])


# Algoritmo Monte-Carlo de Controle, variante "toda-visita".
# Atenção: os espaços de estados e de ações precisam ser discretos, dados por valores inteiros
def run_montecarlo2(env, episodes, lr=0.1, gamma=0.95, epsilon=0.1, render_env=None):
    assert isinstance(env.observation_space, gym.spaces.Discrete)
    assert isinstance(env.action_space, gym.spaces.Discrete)

    num_actions = env.action_space.n
    
    # inicializa a tabela Q toda com zero,
    # usar o estado como índice das linhas e a ação como índice das colunas
    Q = np.zeros(shape = (env.observation_space.n, num_actions))

    # para cada episódio, guarda sua soma de recompensas (retorno não-descontado)
    sum_rewards_per_ep = []

    train_env = None

    # loop principal
    for i in range(episodes):
        done = False
        sum_rewards, reward = 0, 0
        ep_trajectory = []
        
        # exibe/renderiza os passos no ambiente, durante 1 episódio a cada mil e também nos últimos 5 episódios 
        if (render_env is not None) and (i >= (episodes - 5) or (i+1) % 1000 == 0):
            print("Rendering episode", i+1)
            train_env = render_env
        else:
            train_env = env

        state, _ = train_env.reset()
    
        # PARTE 1: executa um episódio completo
        while not done:
            # escolhe a próxima ação -- usa epsilon-greedy
            action = choose_action(Q, state, num_actions, epsilon)
        
            # realiza a ação, ou seja, dá um passo no ambiente
            next_state, reward, terminated, truncated, _ = train_env.step(action)
            done = terminated or truncated
            
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
            delta = Gt - Q[s,a]
            Q[s,a] = Q[s,a] + lr * delta

    return sum_rewards_per_ep, Q



if __name__ == "__main__":
    import sys
    from os import path
    sys.path.append( path.dirname( path.dirname( path.abspath(__file__) ) ) )

    from util.plot import plot_result
    from util.qtable_helper import evaluate_qtable_policy

    ENV_NAME = "Taxi-v3"
    #ENV_NAME = "FrozenLake-v1"
    #ENV_NAME = "CliffWalking-v0"
    r_max_plot = 10

    EPISODES = 20_000
    LR = 0.01
    GAMMA = 0.95
    EPSILON = 0.1

    env = gym.make(ENV_NAME)
    render_env = gym.make(ENV_NAME, render_mode="human", max_episode_steps=100)
    
    # Roda o algoritmo Monte-Carlo para o problema de controle (ou seja, para achar a política ótima)
    rewards, Qtable = run_montecarlo2(env, EPISODES, LR, GAMMA, EPSILON, render_env=render_env)
    print("Últimos resultados: media =", np.mean(rewards[-20:]), ", desvio padrao =", np.std(rewards[-20:]))

    # Mostra um gráfico de episódios x retornos não descontados
    # Passe o caminho para salvar o gráfico em arquivo; ou passe None para exibir em uma janela
    #filename = f"results/montecarlo2-{ENV_NAME.lower()[0:8]}-ep{EPISODES}.png"
    filename = None
    plot_result(rewards, r_max_plot, filename=filename)

    evaluate_qtable_policy(env, Qtable, 10, verbose=True)
    env.close()
