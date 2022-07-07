import gym

from util import repeated_exec, plot_results
from montecarlo_v0 import run_montecarlo

EXECUTIONS = 1

enviroment = gym.make("Taxi-v3")

results = []

results.append( repeated_exec(1, "Monte-Carlo", run_montecarlo, enviroment, 1000) )


plot_results(results, cumulative=False)
