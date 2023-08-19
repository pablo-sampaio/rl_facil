import gym

import sys
from os import path
sys.path.append( path.dirname( path.dirname( path.abspath(__file__) ) ) )

from util.experiments import repeated_exec
from util.plot import plot_multiple_results

from qlearning import run_qlearning
from expected_sarsa import run_expected_sarsa
from nstep_sarsa import run_nstep_sarsa

enviroment = gym.make("Taxi-v3")
#environment = gym.make("FrozenLake-v1")
#environment = gym.make("CliffWalking-v0")

RUNS = 3
NUM_EPISODES = 10000

results = []

'''
for learning_rate in [0.1, 0.5, 1.0]:
    results.append( repeated_exec(RUNS, f"Q-Learning (LR={learning_rate})", run_qlearning, enviroment, NUM_EPISODES, learning_rate) )

for learning_rate in [0.1, 0.5, 1.0]:
    results.append( repeated_exec(RUNS, f"Exp-SARSA (LR={learning_rate})", run_expected_sarsa, enviroment, NUM_EPISODES, learning_rate) )
'''
for lr in [0.1, 0.5]:
    results = []
    for nsteps in [1, 2, 4, 8]:
        results.append( repeated_exec(RUNS, f"{nsteps}-step SARSA (LR={lr})", run_nstep_sarsa, enviroment, NUM_EPISODES, nsteps, lr) )
    plot_multiple_results(results, cumulative=False, x_log_scale=True)
#'''

plot_multiple_results(results, cumulative=False, x_log_scale=True, window=50)
