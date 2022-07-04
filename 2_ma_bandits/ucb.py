
import numpy as np

from bandit_envs import SimpleMultiArmedBandit
from epsilon_greedy import run_epsilon_greedy


def run_ucb(problem):
    num_actions = problem.get_num_actions()

    # estatisticas por ação
    Q = [0.0] * num_actions          # recompensa média (esperada) por ação
    action_cnt  = [0] * num_actions  # quantas vezes cada ação foi realizada

    problem.reset()

    reward_per_step = []    # recompensa recebida a cada passo
    total_steps = 0
    done = False

    # PARTE 1: realiza um passo para cada ação
    # para cada ação "a" guarda a recompensa obtida em "Q[a]"
    for a in range(num_actions):
        r, done = problem.step(a)
        total_steps += 1
        reward_per_step.append(r)

        action_cnt[a] += 1
        Q[a] = r
        #alt.: Q[a] = Q[a] + (1/action_cnt[a]) * (r - Q[a])
  
    # PARTE 2: realiza os passos restantes de maneira similar ao epsilon-greedy
    # mas escolhe a ação de forma diferente
    while not done:
        # argumento do argmax: operações aritméticas em arrays, que resultam em um array de tamanho "num_actions"
        a = np.argmax( Q + np.sqrt(2.0 * np.log(total_steps) / action_cnt) )
        
        r, done = problem.step(a)
        total_steps += 1
        reward_per_step.append(r)

        # atualiza estatísticas da ação
        action_cnt[a] += 1
        Q[a] = Q[a] + (1/action_cnt[a]) * (r - Q[a]) 

    return reward_per_step, Q



if __name__ == '__main__':
    BANDIT_PROBABILITIES = [0.2, 0.5, 0.75]
    mab_problem = SimpleMultiArmedBandit(BANDIT_PROBABILITIES, max_steps=10000)

    rewards, _ = run_ucb(mab_problem)
    print(f"UCB - soma de recompensas:", sum(rewards))

    rewards, _ = run_epsilon_greedy(mab_problem, 0.1)
    print(f"Eps-greedy (0.1) - soma de recompensas:", sum(rewards))

