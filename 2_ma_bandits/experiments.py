
from bandit_envs import SimpleMultiArmedBandit
from util import repeated_exec, plot_results

from bad_algorithms import run_greedy, run_random
from epsilon_greedy import run_epsilon_greedy
from ucb import run_ucb



EXECUTIONS = 50

BANDIT_PROBABILITIES = [0.2, 0.5, 0.75]
mab_problem = SimpleMultiArmedBandit(BANDIT_PROBABILITIES, max_steps=10000)

results = []

results.append( repeated_exec(EXECUTIONS, "GREEDY", run_greedy, mab_problem) )
results.append( repeated_exec(EXECUTIONS, "RANDOM", run_random, mab_problem) )
results.append( repeated_exec(EXECUTIONS, "UCB", run_ucb, mab_problem) )
epsilon = 0.1
results.append( repeated_exec(EXECUTIONS, "EPS({epsilon})-GREEDY", run_epsilon_greedy, mab_problem, epsilon) )


plot_results(results, mab_problem.get_max_mean_reward())
