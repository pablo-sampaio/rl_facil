
import numpy as np

import sys
from os import path
sys.path.append( path.dirname( path.dirname( path.abspath(__file__) ) ) )


def run_ucb(env, total_steps, c=2.0):
    num_actions = env.get_num_actions()

    # estatisticas por ação
    Q = [0.0 for i in range(num_actions)]            # recompensa média (esperada) por ação
    action_cnt  = [0.0 for i in range(num_actions)]  # quantas vezes cada ação foi realizada

    env.reset()

    reward_per_step = []    # recompensa recebida a cada passo
    steps = 0
    done = False

    # PARTE 1: realiza um passo para cada ação
    # para não ter nenhuma ação com "action_cnt" zero
    for a in range(num_actions):
        r = env.step(a)
        steps += 1
        reward_per_step.append(r)

        action_cnt[a] += 1
        Q[a] = r
  
    # PARTE 2: realiza os passos restantes, escolhendo
    # a ação de melhor valor na fórmula do UCB
    for _ in range(total_steps-num_actions):
        # no parametro do argmax: operações aritméticas em arrays, que resultam em um array de tamanho "num_actions"
        a = np.argmax( Q + c * np.sqrt(np.log(steps) / action_cnt) )
        
        r = env.step(a)
        steps += 1
        reward_per_step.append(r)

        # atualiza estatísticas da ação
        action_cnt[a] += 1
        Q[a] = Q[a] + (1/action_cnt[a]) * (r - Q[a]) 

    return reward_per_step, Q



if __name__ == '__main__':
    from envs.bandits import MultiArmedBanditEnv
    from cap01.baseline_algorithms import run_random

    BANDIT_PROBABILITIES = [0.2, 0.5, 0.75]
    env = MultiArmedBanditEnv(BANDIT_PROBABILITIES)

    rewards, _ = run_ucb(env, 10000, c=2.0)
    print(f"UCB(c=2.0) - soma de recompensas:", sum(rewards))

    rewards, _ = run_ucb(env, 10000, c=1.0)
    print(f"UCB(c=1.0) - soma de recompensas:", sum(rewards))

    rewards, _ = run_ucb(env, 10000, c=0.5)
    print(f"UCB(c=0.5) - soma de recompensas:", sum(rewards))

    rewards, _ = run_random(env, total_steps=10000)
    print("Random - soma de recompensas:", sum(rewards))

