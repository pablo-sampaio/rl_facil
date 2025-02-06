import gymnasium as gym

import sys
from os import path
sys.path.append( path.dirname( path.dirname( path.abspath(__file__) ) ) )

import models_torch_pg as models

from util.experiments import repeated_exec, repeated_exec_parallel
from util.plot import plot_multiple_results

from reinforce import run_reinforce
from reinforce_baseline import run_reinforce_baseline
from reinforce_advantage import run_reinforce_advantage


if __name__ == "__main__":
    GAMMA = 0.99

    #ENV_NAME = "CartPole-v1"
    #ENV_NAME = "Acrobot-v1"
    ENV_NAME = "LunarLander-v2"

    env = gym.make(ENV_NAME)
    inputs = env.observation_space.shape[0]
    outputs = env.action_space.n

    RUNS     = 20
    EPISODES = 1_200
    
    # código para rodar em 1 CPU
    '''
    results = []
    #for lr in [0.0001, 0.0005, 0.0010, 0.0020]:
    for lr in [0.0001, 0.0010]:
        initial_policy = PolicyModelPG(inputs, [128,512], outputs, lr=lr)
        results.append( repeated_exec(RUNS, f"Reinforce (lr={lr})"    , run_reinforce          , env, EPISODES, GAMMA, initial_policy, auto_load=True) )
        results.append( repeated_exec(RUNS, f"Reinforce+Bas (lr={lr})", run_reinforce_baseline , env, EPISODES, GAMMA, initial_policy, auto_load=True) )
        results.append( repeated_exec(RUNS, f"Reinforce+Adv (lr={lr})", run_reinforce_advantage, env, EPISODES, GAMMA, initial_policy, auto_load=True) )
    '''

    # código para rodar em múltiplas CPUs
    # atenção: no momento, não funciona bem se usar CUDA
    
    models.DEFAULT_DEVICE = "cpu"
    env_factory = lambda: gym.make(ENV_NAME)
    CPUS = 2

    results = []
    # piora para lrs maiores ou menores
    for lr in [0.0002]: #[0.0001, 0.0005]: 
        initial_policy = models.PolicyModelPG(inputs, [128,512], outputs, lr=lr)
        #results.append( repeated_exec_parallel(RUNS, CPUS, f"Reinforce (lr={lr})"      , run_reinforce          , env_factory, EPISODES, args=(GAMMA, initial_policy), auto_save_load=True) )
        results.append( repeated_exec_parallel(RUNS, CPUS, f"Reinforce+Base (lr={lr})" , run_reinforce_baseline , env_factory, EPISODES, args=(GAMMA, initial_policy), auto_save_load=True) )
        results.append( repeated_exec_parallel(RUNS, CPUS, f"Reinforce+Advtg (lr={lr})", run_reinforce_advantage, env_factory, EPISODES, args=(GAMMA, initial_policy), auto_save_load=True) )

    plot_multiple_results(results, cumulative='no', window=20, x_log_scale=False)
    plot_multiple_results(results, cumulative='avg', x_log_scale=False)

