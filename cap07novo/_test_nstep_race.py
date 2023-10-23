
import numpy as np

import sys
from os import path
sys.path.append( path.dirname( path.dirname( path.abspath(__file__) ) ) )

from util.envs import RacetrackEnv


if __name__=='__main__':
    from cap05.nstep_sarsa import run_nstep_sarsa
    from util.plot import plot_result
    from util.experiments import test_greedy_Q_policy

    EPISODES = 10_000
    LR = 0.1
    GAMMA = 0.95
    EPSILON = 0.1
    NSTEPS = 3

    env = RacetrackEnv()
    
    # Roda o algoritmo "n-step SARSA"
    rewards, qtable = run_nstep_sarsa(env, EPISODES, NSTEPS, LR, GAMMA, EPSILON, render=True)
    print("Últimos resultados: media =", np.mean(rewards[-20:]), ", desvio padrao =", np.std(rewards[-20:]))

    # Exibe um gráfico episódios x retornos (não descontados)
    plot_result(rewards, window=50)

    test_greedy_Q_policy(env, qtable, 10, True, render_wait=0.00)
    env.close()
