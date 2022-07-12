import gym

from util import repeated_exec, plot_results
from montecarlo_v1 import run_montecarlo1
from montecarlo_v2 import run_montecarlo2

EXECUTIONS = 1
NUM_EPISODES = 40000

enviroment = gym.make("Taxi-v3")

results = []

results.append( repeated_exec(1, "Monte-Carlo1", run_montecarlo1, enviroment, NUM_EPISODES) )

for learning_rate in [0.01, 0.05, 0.1]:
    results.append( repeated_exec(1, f"Monte-Carlo2(LR={learning_rate})", run_montecarlo2, enviroment, NUM_EPISODES, learning_rate) )

plot_results(results, cumulative=False)
