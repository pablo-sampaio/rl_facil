
from cProfile import run
from time import time
from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt

from bandit_envs import SimpleMultiArmedBandit


def run_random(env):
    num_actions = env.get_num_actions()
 
    env.reset()
    
    # guarda cada recompensa recebida, a cada passo
    reward_per_step = []   
    done = False

    # realiza todos os passos escolhendo ações aleatórias
    while not done:
        a = np.random.choice(num_actions)
        r, done = env.step(a)
        reward_per_step.append(r)

    return (reward_per_step, None)


def run_greedy(env):
    num_actions = env.get_num_actions()

    # estimativa da recompensa por ação
    Q = [0.0 for i in range(num_actions)]

    env.reset()
    
    reward_per_step = []    # recompensa recebida a cada passo
    done = False

    # PARTE 1: realiza um passo para cada ação
    # para cada ação "a" guarda a recompensa obtida em "Q[a]"
    for a in range(num_actions):
        r, done = env.step(a)
        reward_per_step.append(r)
        Q[a] = r

    # PARTE 2: realiza os passos restantes
    # realizando apenas a ação que tem maior Q
    while not done:
        a = np.argmax(Q)  # indice onde está o maior valor
        r, done = env.step(a)
        reward_per_step.append(r)

    return (reward_per_step, Q)


if __name__ == '__main__':
    BANDIT_PROBABILITIES = [0.2, 0.5, 0.75]
    env = SimpleMultiArmedBandit(BANDIT_PROBABILITIES, max_steps=10000)

    rewards, _ = run_greedy(env)
    print("Greedy - soma de recompensas:", sum(rewards))

    rewards, _ = run_random(env)
    print("Random - soma de recompensas:", sum(rewards))
