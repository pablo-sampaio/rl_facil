
import numpy as np

import sys
from os import path
sys.path.append( path.dirname( path.dirname( path.abspath(__file__) ) ) )

from envs.bandits import MultiArmedBanditEnv


def run_epsilon_greedy(env, total_steps, epsilon):
    num_actions = env.get_num_actions()

    # estatisticas por ação
    Q = [0.0 for i in range(num_actions)]          # recompensa média (esperada) por ação
    action_cnt  = [0 for i in range(num_actions)]  # quantas vezes cada ação foi realizada

    env.reset()

    reward_per_step = []    # recompensas recebidas a cada passo

    for _ in range(total_steps):
        # gera um valor aleatório entre 0 e 1
        # se ele ficar abaixo de "epsilon", faz ação aleatória
        if (np.random.random() <= epsilon):
            a = np.random.randint(num_actions)
        else:
            a = np.argmax(Q)
        
        r = env.step(a)

        reward_per_step.append(r)
        
        # atualiza estatísticas
        action_cnt[a] += 1

        # atualiza a recompensa média da ação
        Q[a] += (1/action_cnt[a]) * (r - Q[a])
        # alternativa equivalente: Q[a] = ((action_cnt[a]-1)*Q[a] + r) / action_cnt[a]

    return reward_per_step, Q



if __name__ == '__main__':
    from cap01.baseline_algorithms import run_random

    BANDIT_PROBABILITIES = [0.2, 0.5, 0.75]
    env = MultiArmedBanditEnv(BANDIT_PROBABILITIES)

    rewards, _ = run_epsilon_greedy(env, total_steps=10000, epsilon=0.3)
    print(f"Eps-greedy (0.3) - soma de recompensas:", sum(rewards))

    rewards, _ = run_epsilon_greedy(env, total_steps=10000, epsilon=0.1)
    print(f"Eps-greedy (0.1) - soma de recompensas:", sum(rewards))

    rewards, _ = run_epsilon_greedy(env, total_steps=10000, epsilon=0.02)
    print(f"Eps-greedy (0.02) - soma de recompensas:", sum(rewards))

    rewards, _ = run_random(env, total_steps=10000)
    print("Random - soma de recompensas:", sum(rewards))
