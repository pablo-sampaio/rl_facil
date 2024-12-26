
import gymnasium as gym
import numpy as np

import sys
from os import path
sys.path.append( path.dirname( path.dirname( path.abspath(__file__) ) ) )

from util.qtable_helper import evaluate_qtable_policy
from util.plot import plot_result

from nstep_sarsa import run_nstep_sarsa

from envs.wrappers import ObservationDiscretizerWrapper


if __name__ == "__main__":

    # 1. Cria o ambiente e o "encapsula" no wrapper
    ENV_NAME = "CartPole-v1"
    r_max_plot = 500

    env = gym.make(ENV_NAME)
    # usando o wrapper para discretizar o ambiente
    env = ObservationDiscretizerWrapper(env, [5,50,70,50])

    # 2. Roda um algoritmo de treinamento
    EPISODES = 20_000
    NSTEPS = 3
    LR = 0.02
    GAMMA = 0.95
    EPSILON = 0.02

    rewards, Qtable = run_nstep_sarsa(env, EPISODES, NSTEPS, LR, GAMMA, EPSILON, verbose=True)

    print("Últimos resultados: media =", np.mean(rewards[-20:]), ", desvio padrao =", np.std(rewards[-20:]))

    # 3. Salva um arquivo com o gráfico de episódios x retornos (não descontados)
    filename = f"results/expected_sarsa-{ENV_NAME.lower()[0:8]}-ep{EPISODES}-lr{LR}.png"
    plot_result(rewards, r_max_plot)

    # 4. Faz alguns testes, usando a tabela de forma greedy
    evaluate_qtable_policy(env, Qtable, 10, verbose=True)
    env.close()
