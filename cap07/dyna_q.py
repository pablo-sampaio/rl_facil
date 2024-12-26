# Referencia: 
# - https://towardsdatascience.com/getting-started-with-reinforcement-learning-and-open-ai-gym-c289aca874f

import random as rand

import gymnasium as gym
import numpy as np

import sys
from os import path
sys.path.append( path.dirname( path.dirname( path.abspath(__file__) ) ) )

from util.qtable_helper import epsilon_greedy


def planning(model, planning_steps, Q, lr, gamma):
    all_s_a = list(model.keys())
    if len(all_s_a) < planning_steps:
        samples = rand.choices(all_s_a, k=planning_steps)
    else:
        samples = rand.sample(all_s_a, k=planning_steps)

    for s, a in samples:
        r, next_s, is_terminal = model[(s,a)]
        if is_terminal:
            V_next_s = 0
        else:
            V_next_s = np.max(Q[next_s])
        delta = (r + gamma * V_next_s) - Q[s,a]
        Q[s,a] = Q[s,a] + lr * delta


# Algoritmo Dyna Q
def run_dyna_q(env, episodes, lr=0.1, gamma=0.95, epsilon=0.1, planning_steps=5, verbose=True):
    assert isinstance(env.observation_space, gym.spaces.Discrete)
    assert isinstance(env.action_space, gym.spaces.Discrete)

    num_actions = env.action_space.n

    # inicializa a tabela Q
    Q = np.random.uniform(low=-0.01, high=+0.01, size=(env.observation_space.n, num_actions))

    model = dict({})

    # para cada episódio, guarda sua soma de recompensas (retorno não-descontado)
    sum_rewards_per_ep = []

    # loop principal
    for i in range(episodes):
        done = False
        sum_rewards, reward = 0, 0

        state, _ = env.reset()

        # executa 1 episódio completo
        while not done:

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

            # atualiza a Q-table / direct RL
            delta = (reward + gamma * V_next_state) - Q[state,action]
            Q[state,action] = Q[state,action] + lr * delta

            # atualiza o modelo
            model[state,action] = (reward, next_state, done)

            # planejamento / indirect RL
            planning(model, planning_steps, Q, lr, gamma)

            sum_rewards += reward
            state = next_state

        sum_rewards_per_ep.append(sum_rewards)

        # a cada 1000 passos, imprime informação sobre o progresso
        if verbose and ((i+1) % 1000 == 0):
            avg_reward = np.mean(sum_rewards_per_ep[-100:])
            print(f"Step {i+1} Average Reward (last 100): {avg_reward:.3f}")

    state = env.reset()
    reward = 0

    return sum_rewards_per_ep, Q


if __name__ == "__main__":
    from util.plot import plot_result
    from util.qtable_helper import evaluate_qtable_policy

    #ENV_NAME, r_max = "FrozenLake-v1", 1.0
    ENV_NAME, r_max = "Taxi-v3", 10.0

    EPISODES = 1_000
    LR = 0.05
    GAMMA = 0.95
    EPSILON = 0.1

    env = gym.make(ENV_NAME)
    rendering_env = gym.make(ENV_NAME, render_mode='human')

    # Roda o algoritmo Q-Learning
    returns, qtable = run_dyna_q(env, EPISODES, LR, GAMMA, EPSILON, planning_steps=5, verbose=True)
    print("Dyna-Q - Treinamento - últimos retornos: media =", np.mean(returns[-20:]), ", desvio padrao =", np.std(returns[-20:]))

    # Mostra um gráfico de episódios x retornos não descontados. Se quiser salvar, passe o nome do arquivo como o 3o parâmetro.
    plot_result(returns, r_max)

    # Cria um ambiente com renderização em modo gráfico e avalia o agente nele
    render_env = gym.make(ENV_NAME, render_mode="human")
    evaluate_qtable_policy(render_env, qtable, 10, epsilon=0.0, verbose=True)
