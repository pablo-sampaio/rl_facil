import gymnasium as gym

import sys
from os import path
sys.path.append( path.dirname( path.dirname( path.abspath(__file__) ) ) )

from models_torch_pg import PolicyModelPG 

from util.experiments import repeated_exec
from util.plot import plot_multiple_results

from reinforce import run_reinforce
from reinforce_advantage import run_reinforce_advantage

GAMMA = 0.99

ENV = gym.make("CartPole-v1")
#ENV = gym.make("Acrobot-v1")
#ENV = gym.make("LunarLander-v2")

inputs = ENV.observation_space.shape[0]
outputs = ENV.action_space.n


NUM_EPISODES = 1000
results = []
for lr in [0.0001, 0.0005, 0.001]:
    initial_policy = PolicyModelPG(inputs, [128,512], outputs, lr=lr)
    results.append( repeated_exec(2, f"Reinforce (lr={lr})", run_reinforce, ENV, NUM_EPISODES, GAMMA, initial_policy) )
    results.append( repeated_exec(2, f"Reinforce+Adv (lr={lr})", run_reinforce_advantage, ENV, NUM_EPISODES, GAMMA, initial_policy) )

plot_multiple_results(results, cumulative=False, x_log_scale=False)
plot_multiple_results(results, cumulative=True, x_log_scale=False)

