import gym

from util_experiments import repeated_exec
from util_plot import plot_results

from montecarlo_v1 import run_montecarlo1
from montecarlo_v2 import run_montecarlo2


NUM_EPISODES = 20000

enviroment = gym.make("Taxi-v3")
#enviroment = gym.make("FrozenLake-v1")

results = []

#results.append( repeated_exec(1, "Monte-Carlo1", run_montecarlo1, enviroment, NUM_EPISODES) )
for learning_rate in [0.01, 0.05, 0.1]:
    results.append( repeated_exec(10, f"Monte-Carlo2 (LR={learning_rate})", run_montecarlo2, enviroment, NUM_EPISODES, learning_rate) )

plot_results(results, cumulative=False, x_log_scale=True)

