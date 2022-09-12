import gym

from models_torch_pg import PolicyModelPG, ValueModel

from util_experiments import repeated_exec, repeated_exec_steps
from util_plot import plot_multiple_results

from reinforce import run_reinforce
from reinforce_advantage import run_advantage_reinforce
from actor_critic_nstep import run_actor_critic_nstep


GAMMA = 0.99

ENV = gym.make("CartPole-v1")
#enviroment = gym.make("Acrobot-v1")
inputs = ENV.observation_space.shape[0]
outputs = ENV.action_space.n


def experiments_num_episodes():
    NUM_EPISODES = 1000
    results = []
    for lr in [0.0001, 0.0005, 0.001]:
        initial_policy = PolicyModelPG(inputs, [128,512], outputs, lr=lr)
        results.append( repeated_exec(2, f"Reinforce (lr={lr})", run_reinforce, ENV, NUM_EPISODES, GAMMA, initial_policy) )
        results.append( repeated_exec(2, f"Reinforce+Advtg (lr={lr})", run_advantage_reinforce, ENV, NUM_EPISODES, GAMMA, initial_policy) )
    
    plot_multiple_results(results, cumulative=False, x_log_scale=False)
    plot_multiple_results(results, cumulative=True, x_log_scale=False)


def experiments_max_steps():
    NUM_STEPS = 15000
    results = []
    for v_lr in [1e-4, 5e-5]:
        for p_lr in [5e-5, 1e-4]:
            policy_model = PolicyModelPG(inputs, [256,256], outputs, lr=p_lr)
            Vmodel = ValueModel(inputs, [256,32], lr=v_lr)
            results.append( repeated_exec_steps(20, f"Actor-critic ({p_lr},{v_lr})", run_actor_critic_nstep, ENV, NUM_STEPS, GAMMA, 32, policy_model, Vmodel, verbose=False) )
    
    plot_multiple_results(results, cumulative=False, x_log_scale=False)


# comando principal
experiments_max_steps()
