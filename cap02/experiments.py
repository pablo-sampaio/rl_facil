
from bandit_envs import MultiArmedBandit, GaussianMultiArmedBandit
from util import repeated_exec, plot_results

from baseline_algorithms import run_greedy, run_random
from epsilon_greedy import run_epsilon_greedy
from ucb import run_ucb



NUM_EXECUTIONS = 30
NUM_STEPS = 10000

BANDITS_PROBABILITIES = [0.2, 0.5, 0.75]
enviroment = MultiArmedBandit(BANDITS_PROBABILITIES)
#enviroment = GaussianMultiArmedBandit(BANDITS_PROBABILITIES, max_steps=10000)

results = []

results.append( repeated_exec(NUM_EXECUTIONS, "RANDOM", run_random, enviroment, NUM_STEPS, False) )
results.append( repeated_exec(NUM_EXECUTIONS, "GREEDY", run_greedy, enviroment, NUM_STEPS, False) )

#for epsilon in [0.1, 0.4, 0.01]:
    #results.append( repeated_exec(NUM_EXECUTIONS, f"EPS({epsilon})-GREEDY", run_epsilon_greedy, enviroment, epsilon) )

#results.append( repeated_exec(NUM_EXECUTIONS, "UCB", run_ucb, enviroment) )


plot_results(results, enviroment.get_max_mean_reward())
