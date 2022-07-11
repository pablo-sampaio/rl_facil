
import numpy as np

from bandit_envs import SimpleMultiArmedBandit


def run_epsilon_greedy(env, epsilon):
    num_actions = env.get_num_actions()

    # estatisticas por ação
    Q = [0.0 for i in range(num_actions)]          # recompensa média (esperada) por ação
    action_cnt  = [0 for i in range(num_actions)]  # quantas vezes cada ação foi realizada

    env.reset()

    reward_per_step = []    # recompensas recebidas a cada passo
    done = False

    while not done:
        # gera um valor aleatório entre 0 e 1
        # se ele ficar abaixo de "epsilon", faz ação aleatória
        if (np.random.random() <= epsilon):
            a = np.random.randint(num_actions)
            #a = np.random.choice(num_actions)
        else:
            a = np.argmax(Q)
        
        r, done = env.step(a)

        reward_per_step.append(r)
        
        # atualiza estatísticas
        action_cnt[a] += 1

        # atualiza a recompensa média da ação
        delta = r - Q[a]
        Q[a] += (1/action_cnt[a]) * delta
        # alternativa equivalente: Q[a] = ((action_cnt[a]-1)*Q[a] + r) / action_cnt[a]

    return reward_per_step, Q



if __name__ == '__main__':
    BANDIT_PROBABILITIES = [0.2, 0.5, 0.75]
    mab_problem = SimpleMultiArmedBandit(BANDIT_PROBABILITIES, max_steps=10000)

    rewards, _ = run_epsilon_greedy(mab_problem, 0.1)
    print(f"Eps-greedy (0.1) - soma de recompensas:", sum(rewards))

    rewards, _ = run_epsilon_greedy(mab_problem, 0.4)
    print(f"Eps-greedy (0.4) - soma de recompensas:", sum(rewards))

    rewards, _ = run_epsilon_greedy(mab_problem, 0.01)
    print(f"Eps-greedy (0.01) - soma de recompensas:", sum(rewards))
