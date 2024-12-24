from baseline_algorithms import run_greedy, run_random
from epsilon_greedy import run_epsilon_greedy
from ucb import run_ucb

from envs.bandits import MultiArmedBanditEnv, GaussianMultiArmedBanditEnv

from util.experiments import repeated_exec
from util.plot import plot_multiple_results


RUNS  = 100
STEPS = 10000

BANDITS_PROBABILITIES = [0.2, 0.5, 0.75]
enviroment = MultiArmedBanditEnv(BANDITS_PROBABILITIES)
#enviroment = GaussianMultiArmedBanditEnv(BANDITS_PROBABILITIES)

results = []

results.append( repeated_exec(RUNS, "RANDOM", run_random, enviroment, STEPS) )
#results.append( repeated_exec(RUNS, "GREEDY", run_greedy, enviroment, STEPS) )

epsilon = 0.02
results.append( repeated_exec(RUNS, f"EPS({epsilon})-GREEDY", run_epsilon_greedy, enviroment, STEPS, epsilon) )

c = 0.5
results.append( repeated_exec(RUNS, f"UCB(c=c)", run_ucb, enviroment, STEPS, c) )

for (alg_name, rewards) in results:
    print("Summary for " + alg_name)
    print(" - total reward:", rewards.sum())
    print(" - avg reward (win rate):", rewards.sum() / STEPS)
    print()

plot_multiple_results(results, cumulative='avg', x_log_scale=True, yreference=enviroment.get_max_mean_reward(), plot_stddev=True)
