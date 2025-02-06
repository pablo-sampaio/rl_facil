
import time
import os
from tqdm import tqdm

import numpy as np
import gymnasium as gym


def process_returns_linear_interpolation(step_return_list, total_time):
    #assert total_time == step_return_list[-1][0], "The algorithm did not return a final (partial) return at the last time step!"
    partial_returns = [0] * total_time
    X = 0
    t1 = 0
    for i in range(len(step_return_list)):
        t2, Y = step_return_list[i]
        
        # if t1+1 > total_time, it wont't enter the loop
        # if t2 > total_time, it will calculate up to total_time
        for t in range(t1+1, min(t2, total_time) + 1):
            partial_returns[t - 1] = X + ( (Y - X) * (t - t1) / (t2 - t1) )
            #alt.: partial_returns[t - 1] = ((t2 - t) * X + (t - t1) * Y) / (t2 - t1)
        
        t1 = t2
        X = Y
    
    return partial_returns


def repeated_exec(executions, alg_name, algorithm, env, num_iterations, *args, **kwargs):
    '''
    This file runs repeatedly the given algorithm with the given parameters and returns
    results compatible with the functions in 'util.plot'.

    Parameters:
    - executions: number of times that the 'algorithm' will be run from the start
    - alg_name: a string to represent this setting of algorithm (with the given parameters)
    - algorithm: must be a function that receives 'env' then and integer (corresponding to the 'num_iterations') then the list of arguments (given by'*args' and/or **kwargs)
    - num_iterations: number of steps or episodes
    - *args: list of arguments for 'algorithm'
    - **kwargs: named arguments for 'algorithm'
    - auto_load: to save the results and reload automatically when re-executed with exactly the same parameters (including the number of executions)
    '''
    # gets a string to identify the environment
    if isinstance(env, gym.Env):
        env_name = str(env).replace('<', '_').replace('>', '')
    else:
        env_name = type(env).__name__ 
    auto_load = False
    if ('auto_load' in kwargs):
        auto_load = kwargs['auto_load']
        kwargs.pop('auto_load')
    result_file_name = f"results/{env_name}-{alg_name}-episodes{num_iterations}-execs{executions}.npy"
    if auto_load and os.path.exists(result_file_name):
        print("Loading results from", result_file_name)
        RESULTS = np.load(result_file_name, allow_pickle=True)
        return RESULTS
    rewards = np.zeros(shape=(executions, num_iterations))
    t = time.time()
    print(f"Executing {alg_name} ({algorithm}):")
    for i in tqdm(range(executions)):
        alg_output = algorithm(env, num_iterations, *args, **kwargs)
        temp_returns = alg_output[0]
        if isinstance(temp_returns[0], tuple):
            # when the algorithm outputs a list of pairs (time, return)
            rewards[i] = process_returns_linear_interpolation(temp_returns, num_iterations)
        else:
            # when the algoritm outputs a simple list of returns
            rewards[i] = temp_returns
    t = time.time() - t
    print(f"  ({executions} executions of {alg_name} finished in {t:.2f} secs)")
    RESULTS = np.array([alg_name, rewards], dtype=object)
    directory = os.path.dirname(result_file_name)
    if auto_load:
        if not os.path.exists(directory):
            os.makedirs(directory)
        np.save(result_file_name, RESULTS, allow_pickle=True)
    return alg_name, rewards


import multiprocessing

def worker(args):
    i, algorithm, env, num_iterations, alg_args, alg_kwargs = args
    try:
        temp_returns, _ = algorithm(env, num_iterations, *alg_args, **alg_kwargs)
        if isinstance(temp_returns[0], tuple):
            # when the algorithm outputs a list of pairs (time, return)
            return process_returns_linear_interpolation(temp_returns, num_iterations)
        else:
            # when the algoritm outputs a simple list of returns
            return temp_returns
    except Exception as e:
        print(f"Error in execution {i} of {algorithm}: {str(e)}")
        return None

def repeated_exec_parallel(executions, num_cpus, alg_name, algorithm, env_factory, num_iterations, args=(), kwargs=dict(), auto_save_load=False):
    env = env_factory()
    assert isinstance(env, gym.Env)
    env_name = str(env).replace('<', '_').replace('>', '')
    result_file_name = f"results/{env_name}-{alg_name}-episodes{num_iterations}-execs{executions}.npy"
    if auto_save_load and os.path.exists(result_file_name):
        print("Loading results from", result_file_name)
        RESULTS = np.load(result_file_name, allow_pickle=True)
        return RESULTS

    rewards = None  # expected final shape: (executions, num_iterations)
    t = time.time()
    print(f"Executing {alg_name} ({algorithm}) in {num_cpus} cpus:")

    with multiprocessing.Pool(num_cpus) as p:
        args_for_worker = [(i, algorithm, env_factory(), num_iterations, args, kwargs) for i in range(executions)]
        rewards_list = p.map(worker, args_for_worker)
        # catches any excetion raised for invalid rewards list
        try:
            rewards = np.array(rewards_list)
        except:
            print("ERROR: invalid rewards list returned by the algorithm!")
            print("rewards_list =", rewards_list)
            raise

    t = time.time() - t
    print(f"  ({executions} executions of {alg_name} finished in {t:.2f} secs)")

    RESULTS = np.array([alg_name, rewards], dtype=object)
    directory = os.path.dirname(result_file_name)
    if auto_save_load:
        if not os.path.exists(directory):
            os.makedirs(directory)
        np.save(result_file_name, RESULTS, allow_pickle=True)
    
    return alg_name, rewards

