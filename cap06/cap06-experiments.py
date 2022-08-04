import gym
from models_torch import PolicyModelCrossentropy

from util_experiments import repeated_exec
from util_plot import plot_multiple_results

from crossentropy_method_v1 import run_crossentropy_method1
from crossentropy_method_v2 import run_crossentropy_method2


NUM_EPISODES = 500

#enviroment = gym.make("CartPole-v1")
enviroment = gym.make("Acrobot-v1")

results = []

'''for batch_size in [5, 10, 20]:
    results = []
    for proportion in [1/5.0, 2/5.0, 3/5.0]:
        results.append( repeated_exec(5, f"CrossEntropy ({batch_size},{proportion:.2f})", run_crossentropy_method, enviroment, NUM_EPISODES, batch_size, proportion) )
    plot_multiple_results(results, cumulative=False, x_log_scale=False)
'''

policy = PolicyModelCrossentropy(enviroment.observation_space.shape[0], [64,256], enviroment.action_space.n)

results.append( repeated_exec(1, f"CrossEntropy-2 (20; 0.1)", run_crossentropy_method2, enviroment, NUM_EPISODES, 20, 0.1, policy) )
results.append( repeated_exec(1, f"CrossEntropy-1 (20; 0.1)", run_crossentropy_method1, enviroment, NUM_EPISODES, 20, 0.1, policy) )

plot_multiple_results(results, cumulative=False, x_log_scale=False)
plot_multiple_results(results, cumulative=True, x_log_scale=False)
