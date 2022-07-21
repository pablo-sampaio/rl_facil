
import gym
import numpy as np

from util_experiments import test_greedy_Q_policy
from util_plot import plot_results

from expected_sarsa import run_expected_sarsa
from qlearning import run_qlearning

from wrappers import DiscreteObservationWrapper, PunishEarlyStop


if __name__ == "__main__":

    # 1. Cria o ambiente e o "encapsula" no wrapper
    ENV_NAME = "MountainCar-v0"
    r_max_plot = 500

    env = gym.make(ENV_NAME)
    # usando o wrapper para discretizar o ambiente
    env = DiscreteObservationWrapper(env, [30,90])

    # 2. Roda um algoritmo de treinamento
    EPISODES = 1000
    LR = 0.63
    GAMMA = 0.95
    EPSILON = 0.10

    rewards, Qtable = run_expected_sarsa(env, EPISODES, LR, GAMMA, EPSILON, render=True)

    print("Últimos resultados: media =", np.mean(rewards[-20:]), ", desvio padrao =", np.std(rewards[-20:]))

    # 3. Salva um arquivo com o gráfico de episódios x retornos (não descontados)
    filename = f"results/expected_sarsa-{ENV_NAME.lower()[0:8]}-ep{EPISODES}-lr{LR}.png"
    plot_results(rewards, r_max_plot, None)

    # 4. Faz alguns testes, usando a tabela de forma greedy
    test_greedy_Q_policy(env, Qtable, 10, True)
    env.close()
