import numpy as np

import sys
from os import path

sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

from util.bandit_envs import MultiArmedBanditEnv
from cap02.baseline_algorithms import run_random


def run_decaying_epsilon_greedy(env, total_steps,minimal_epsilon, target_step, initial_epsilon = float(1.0)):
    num_actions = env.get_num_actions()

    # estatisticas por ação
    Q = [0.0 for i in range(num_actions)]  # recompensa média (esperada) por ação
    action_cnt = [0 for i in range(num_actions)]  # quantas vezes cada ação foi realizada

    env.reset()

    #Decaimento e epsilon inicial
    decay = float((initial_epsilon - minimal_epsilon)/target_step)
    epsilon = float(initial_epsilon)

    reward_per_step = []  # recompensas recebidas a cada passo


    for _ in range(total_steps):
        # Atualiza o valor de epsilon

        if epsilon <= minimal_epsilon:
            epsilon = float(minimal_epsilon)
        else:
            epsilon = float(epsilon - decay)
        if (np.random.random() <= epsilon):
            a = np.random.randint(num_actions)
        else:
            a = np.argmax(Q)

        r = env.step(a)

        reward_per_step.append(r)

        # atualiza estatísticas
        action_cnt[a] += 1

        # atualiza a recompensa média da ação
        delta = r - Q[a]
        Q[a] += (1 / action_cnt[a]) * delta
        # alternativa equivalente: Q[a] = ((action_cnt[a]-1)*Q[a] + r) / action_cnt[a]

    return reward_per_step, Q



if __name__ == '__main__':
    BANDIT_PROBABILITIES = [0.2, 0.5, 0.75]
    env = MultiArmedBanditEnv(BANDIT_PROBABILITIES)

    rewards, _ = run_decaying_epsilon_greedy(env, 10_000, 0.01, 1000)
    print(f"Decaying Eps-greedy - soma de recompensas:", sum(rewards))

    rewards, _ = run_decaying_epsilon_greedy(env, 10_000, 0.1, 10_000, 0.1)
    print(f"Eps-greedy - soma de recompensas:", sum(rewards))
