import gymnasium as gym

import sys
from os import path
sys.path.append( path.dirname( path.dirname( path.abspath(__file__) ) ) )

from util.experiments import repeated_exec
from util.plot import plot_multiple_results

from qlearning_sarsa import run_qlearning, run_sarsa
from expected_sarsa import run_expected_sarsa

enviroment = gym.make("Taxi-v3")
#environment = gym.make("FrozenLake-v1")
#environment = gym.make("CliffWalking-v0")

RUNS = 3
NUM_EPISODES = 10000
AUTO_LOAD = True

results = []

for learning_rate in [0.1, 0.5, 1.0]:
    results.append( repeated_exec(RUNS, f"Q-Learning (LR={learning_rate})", run_qlearning, enviroment, NUM_EPISODES, learning_rate, auto_load=AUTO_LOAD) )

for learning_rate in [0.1, 0.5, 1.0]:
    results.append( repeated_exec(RUNS, f"SARSA (LR={learning_rate})", run_sarsa, enviroment, NUM_EPISODES, learning_rate, auto_load=AUTO_LOAD) )

for learning_rate in [0.1, 0.5, 1.0]:
    results.append( repeated_exec(RUNS, f"Exp-SARSA (LR={learning_rate})", run_expected_sarsa, enviroment, NUM_EPISODES, learning_rate, auto_load=AUTO_LOAD) )

plot_multiple_results(results, cumulative=False, x_log_scale=True, window=50)
