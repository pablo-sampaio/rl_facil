import gymnasium as gym

from montecarlo_v1 import run_montecarlo1
from montecarlo_v2 import run_montecarlo2

import sys
from os import path
sys.path.append( path.dirname( path.dirname( path.abspath(__file__) ) ) )

from util.experiments import repeated_exec
from util.plot import plot_multiple_results


NUM_EPISODES = 4_000 #12_000

enviroment = gym.make("Taxi-v3")
#enviroment = gym.make("FrozenLake-v1")

results = []

# muito lento, se usar o histórico completo!
#results.append( repeated_exec(1, "Monte-Carlo1", run_montecarlo1, enviroment, NUM_EPISODES) )

for learning_rate in [0.01, 0.1, 0.5]:
    results.append( repeated_exec(1, f"Monte-Carlo2 (LR={learning_rate})", run_montecarlo2, enviroment, NUM_EPISODES, learning_rate) )


plot_multiple_results(results, x_log_scale=False)
