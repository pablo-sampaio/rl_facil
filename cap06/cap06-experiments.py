import gym

from util_experiments import repeated_exec
from util_plot import plot_multiple_results

from crossentropy_method_v1 import run_crossentropy_method
from crossentropy_method_v2 import run_crossentropy_method_x


NUM_EPISODES = 1000

enviroment = gym.make("CartPole-v1")

results = []

for batch_size in [5, 10, 20]:
    results = []
    for proportion in [1/5.0, 2/5.0, 3/5.0]:
        results.append( repeated_exec(5, f"CrossEntropy ({batch_size},{proportion:.2f})", run_crossentropy_method, enviroment, NUM_EPISODES, batch_size, proportion) )
    plot_multiple_results(results, cumulative=False, x_log_scale=False)

#plot_multiple_results(results, cumulative=False, x_log_scale=False)
