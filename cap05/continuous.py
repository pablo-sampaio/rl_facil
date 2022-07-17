
import gym
import numpy as np

from util_experiments import test_greedy_Q_policy
from util_plot import save_rewards_plot

from expected_sarsa import run_expected_sarsa
from qlearning import run_qlearning

from wrappers import DiscreteObservationWrapper, PunishEarlyStop


if __name__ == "__main__":
    ENV_NAME = "CartPole-v1"
    r_max_plot = 500

    EPISODES = 10000
    LR = 0.5
    GAMMA = 0.95
    EPSILON = 0.1

    env = gym.make(ENV_NAME)
    env = DiscreteObservationWrapper(env, [20,50,10,20])
    
    # Roda o algoritmo Expected-SARSA
    rewards, Qtable = run_expected_sarsa(env, EPISODES, LR, GAMMA, EPSILON, render=True)
    print("Últimos resultados: media =", np.mean(rewards[-20:]), ", desvio padrao =", np.std(rewards[-20:]))

    # Salva um arquivo com o gráfico de episódios x retornos (não descontados)
    filename = f"results/expected_sarsa-{ENV_NAME.lower()[0:8]}-ep{EPISODES}-lr{LR}.png"
    save_rewards_plot(rewards, r_max_plot, filename)

    test_greedy_Q_policy(env, Qtable, 10, True)
    env.close()
