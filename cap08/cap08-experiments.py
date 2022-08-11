import gym
from models_torch import PolicyModelPG

from util_experiments import repeated_exec
from util_plot import plot_multiple_results

from reinforce import run_reinforce
from reinforce_advantage import run_advantage_reinforce


NUM_EPISODES = 1000
GAMMA = 0.99

enviroment = gym.make("CartPole-v1")
#enviroment = gym.make("Acrobot-v1")

results = []

for lr in [0.001, 0.0001]:
    initial_policy = PolicyModelPG(enviroment.observation_space.shape[0], [128,512], enviroment.action_space.n, lr=lr)
    
    results.append( repeated_exec(2, f"Reinforce (lr={lr})", run_reinforce, enviroment, NUM_EPISODES, GAMMA, initial_policy) )
    results.append( repeated_exec(2, f"Reinforce+Advtg (lr={lr})", run_advantage_reinforce, enviroment, NUM_EPISODES, GAMMA, initial_policy) )

plot_multiple_results(results, cumulative=False, x_log_scale=False)
plot_multiple_results(results, cumulative=True, x_log_scale=False)
