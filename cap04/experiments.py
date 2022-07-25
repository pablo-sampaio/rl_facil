import gym

from util_experiments import repeated_exec
from util_plot import plot_multiple_results

from montecarlo_v1 import run_montecarlo1
from montecarlo_v2 import run_montecarlo2
from qlearning import run_qlearning
from expected_sarsa import run_expected_sarsa
from nstep_sarsa import run_nstep_sarsa


NUM_EPISODES = 12000

#enviroment = gym.make("Taxi-v3")
enviroment = gym.make("FrozenLake-v1")

results = []

#results.append( repeated_exec(1, "Monte-Carlo1", run_montecarlo1, enviroment, NUM_EPISODES) )

#for learning_rate in [0.01, 0.05, 0.1]:
#    results.append( repeated_exec(1, f"Monte-Carlo2 (LR={learning_rate})", run_montecarlo2, enviroment, NUM_EPISODES, learning_rate) )

#for learning_rate in [0.05, 0.1, 0.5, 1.0]:
#    results.append( repeated_exec(1, f"Q-Learning (LR={learning_rate})", run_qlearning, enviroment, NUM_EPISODES, learning_rate) )

#for learning_rate in [0.1, 0.5, 1.0]:
#    results.append( repeated_exec(1, f"Exp-SARSA (LR={learning_rate})", run_expected_sarsa, enviroment, NUM_EPISODES, learning_rate) )

for lr in [0.5]:
    for nstep in [1, 2, 3, 4]:
        results.append( repeated_exec(10, f"{nstep}-step SARSA (LR={lr})", run_nstep_sarsa, enviroment, NUM_EPISODES, nstep, lr) )
    
plot_multiple_results(results, cumulative=False, x_log_scale=False)
