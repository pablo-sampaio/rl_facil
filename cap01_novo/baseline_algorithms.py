
import numpy as np
import matplotlib.pyplot as plt

import sys
from os import path
sys.path.append( path.dirname( path.dirname( path.abspath(__file__) ) ) )

from util.bandit_envs import MultiArmedBanditEnv


def run_random(env, total_steps):
    num_actions = env.get_num_actions()
 
    env.reset()
    reward_per_step = []    # recompensa recebida a cada passo

    # realiza todos os passos escolhendo ações aleatórias
    for _ in range(total_steps):
        action = np.random.choice(num_actions)
        reward = env.step(action)
        reward_per_step.append(reward)

    return (reward_per_step, None)


def run_greedy(env, total_steps):
    num_actions = env.get_num_actions()

    # estimativa da recompensa por ação
    Q = [0.0 for i in range(num_actions)]

    env.reset()   
    reward_per_step = []    # recompensa recebida a cada passo

    # PARTE 1: realiza um passo para cada ação
    # para cada ação "a" guarda a recompensa obtida em "Q[a]"
    for action in range(num_actions):
        reward = env.step(action)
        reward_per_step.append(reward)
        Q[action] = reward

    # PARTE 2: realiza os passos restantes repetindo apenas a ação que tem maior Q
    best_action = np.argmax(Q)
    
    for _ in range(total_steps - num_actions):
        reward = env.step(best_action)
        reward_per_step.append(reward)

    return (reward_per_step, Q)


if __name__ == '__main__':
    BANDIT_PROBABILITIES = [0.2, 0.5, 0.75]
    env = MultiArmedBanditEnv(BANDIT_PROBABILITIES)

    rewards, _ = run_greedy(env, total_steps=10000)
    print("Greedy - soma de recompensas:", sum(rewards))

    rewards, _ = run_random(env, total_steps=10000)
    print("Random - soma de recompensas:", sum(rewards))
