import gymnasium as gym

import sys
from os import path
sys.path.append( path.dirname( path.dirname( path.abspath(__file__) ) ) )

import cap09.models_torch_pg as models 

from util.experiments import repeated_exec_parallel
from util.plot import plot_multiple_results

from actor_critic_nstep import run_vanilla_actor_critic_nstep


if __name__=='__main__':
    GAMMA = 0.99
    N_STEPS = 16

    #ENV_NAME = "CartPole-v1"
    #ENV_NAME = "Acrobot-v1"
    ENV_NAME = "LunarLander-v2"

    env = gym.make(ENV_NAME)
    inputs = env.observation_space.shape[0]
    outputs = env.action_space.n

    RUNS        = 8
    TOTAL_STEPS = 100_000

    # código para rodar em múltiplas CPUs
    # atenção: no momento, não funciona bem se usar CUDA
    
    models.DEFAULT_DEVICE = "cpu"
    env_factory = lambda: gym.make(ENV_NAME)
    CPUS = 4

    results = []

    for p_lr in [3e-5, 5e-5]: # worse: 1e-5 
        for relative_v_lr in [5.0]: # [2.0, 5.0]:
            initial_policy = models.PolicyModelPG(inputs, [256,256], outputs, lr=p_lr)
            
            v_lr = relative_v_lr * p_lr
            initial_Vmodel = models.ValueModel(inputs, [256,256], lr=v_lr)
            
            output = repeated_exec_parallel(RUNS, CPUS, 
                                            f"VAC-{N_STEPS} ({p_lr:.1e},{v_lr:.1e})", 
                                            run_vanilla_actor_critic_nstep, 
                                            env_factory, TOTAL_STEPS, 
                                            args=(GAMMA, N_STEPS, initial_policy, initial_Vmodel), 
                                            kwargs={'verbose':True}, 
                                            auto_save_load=True) 
            results.append(output)

    plot_multiple_results(results, cumulative=False, plot_stddev=False, x_axis='steps')
    plot_multiple_results(results, cumulative=True, plot_stddev=False, x_axis='steps')
