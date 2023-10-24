
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
    print(f"Executing {algorithm}:")
    for i in tqdm(range(executions)):
        temp_returns, _ = algorithm(env, num_iterations, *args, **kwargs)
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


# TODO: REMOVE (o repeated_exec já funciona para saídas com pares (passo, retorno))
# for algorithms that return a list of pairs (timestep, return)
# fazer: descartar o alg_info
def repeated_exec_steps(executions, alg_name, algorithm, env, num_steps, *args, **kwargs):
    env_name = str(env.unwrapped).replace('<','_').replace('>','_')
    result_file_name = f"results/{env_name}-{alg_name}-steps{num_steps}-execs{executions}.npy"
    if os.path.exists(result_file_name):
        print("Loading results from", result_file_name)
        RESULTS = np.load(result_file_name, allow_pickle=True)
        return RESULTS
    rewards = np.zeros(shape=(executions, num_steps))
    #alg_infos = np.empty(shape=(executions,), dtype=object)
    t = time.time()
    print(f"Executing {algorithm}:")
    for i in tqdm(range(executions)):
        # executa o algoritmo
        list_pairs, _ = algorithm(env, num_steps, *args, **kwargs)
        final_steps_i, returns_i = list(zip(*list_pairs))
        final_steps_i, returns_i = list(final_steps_i), list(returns_i)
        prev_return = 0
        prev_final_step = 0
        for step in range(num_steps):
            if not final_steps_i:
                # lista vazia antes do fim
                rewards[i,step] = None
            elif step == final_steps_i[0]:
                # passo final de um episodio
                rewards[i,step] = returns_i[0]
                prev_return = returns_i[0]
                prev_final_step = step
                final_steps_i.pop(0)
                returns_i.pop(0)
            else:
                # passo intermediário - faz uma interpolação
                next_return = returns_i[0]
                next_final_step = final_steps_i[0]
                rewards[i, step] = prev_return + (next_return - prev_return)*(step - prev_final_step) / (next_final_step - prev_final_step)
    t = time.time() - t
    print(f"  ({executions} executions of {alg_name} finished in {t:.2f} secs)")
    rew_mean, rew_std = rewards.mean(axis=0), rewards.std(axis=0)
    RESULTS = np.array([alg_name, rew_mean, rew_std], dtype=object)
    directory = os.path.dirname(result_file_name)
    if not os.path.exists(directory):
        os.makedirs(directory)
    np.save(result_file_name, RESULTS, allow_pickle=True)
    return alg_name, rew_mean, rew_std

