
from bandit_envs import SimpleMultiArmedBandit, GaussianMultiArmedBandit
from util import repeated_exec, plot_results

from bad_algorithms import run_greedy, run_random
from epsilon_greedy import run_epsilon_greedy
from ucb import run_ucb



EXECUTIONS = 50

BANDIT_PROBABILITIES = [0.2, 0.5, 0.75]
enviroment = SimpleMultiArmedBandit(BANDIT_PROBABILITIES, max_steps=10000)
#enviroment = GaussianMultiArmedBandit(BANDIT_PROBABILITIES, max_steps=10000)

results = []

results.append( repeated_exec(EXECUTIONS, "RANDOM", run_random, enviroment) )
results.append( repeated_exec(EXECUTIONS, "GREEDY", run_greedy, enviroment) )

for epsilon in [0.1, 0.4, 0.01]:
    results.append( repeated_exec(EXECUTIONS, f"EPS({epsilon})-GREEDY", run_epsilon_greedy, enviroment, epsilon) )

results.append( repeated_exec(EXECUTIONS, "UCB", run_ucb, enviroment) )


plot_results(results, enviroment.get_max_mean_reward())
