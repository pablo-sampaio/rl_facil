import gymnasium as gym

import sys
from os import path
sys.path.append( path.dirname( path.dirname( path.abspath(__file__) ) ) )

from cap09.models_torch_pg import PolicyModelPG, ValueModel

from util.experiments import repeated_exec, repeated_exec_steps
from util.plot import plot_multiple_results

from actor_critic_nstep import run_vanilla_actor_critic_nstep

GAMMA = 0.99

ENV = gym.make("CartPole-v1")
#enviroment = gym.make("Acrobot-v1")
inputs = ENV.observation_space.shape[0]
outputs = ENV.action_space.n


def experiments_max_steps():
    NUM_STEPS = 5000
    all_results = []
    for v_lr in [1e-4, 5e-5]:
        for p_lr in [5e-5, 1e-5]:
            policy_model = PolicyModelPG(inputs, [256,256], outputs, lr=p_lr)
            Vmodel = ValueModel(inputs, [256,32], lr=v_lr)
            
            #result = repeated_exec_steps(10, f"Actor-critic ({p_lr},{v_lr})", run_vanilla_actor_critic_nstep, ENV, NUM_STEPS, GAMMA, 32, policy_model, Vmodel, verbose=False)
            result = repeated_exec(10, f"Actor-critic ({p_lr},{v_lr})", run_vanilla_actor_critic_nstep, ENV, NUM_STEPS, GAMMA, 32, policy_model, Vmodel, verbose=False)
            all_results.append( result )
    
    plot_multiple_results(all_results, cumulative=False, x_log_scale=True, plot_stddev=True, return_type='steps')


# comando principal
experiments_max_steps()
