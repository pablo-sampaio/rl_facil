
# TODO: remove this file
#        and use the same functions used for full-RL algorithms

import time
import os
from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt


def repeated_exec(executions, alg_name, algorithm, env, num_episodes, reload=False, *args):
    result_file_name = f"results/{env}-{alg_name}-execs{executions}.npy"
    if reload and os.path.exists(result_file_name):
        print("Loading results from", result_file_name)
        RESULTS = np.load(result_file_name, allow_pickle=True)
        return RESULTS
    rewards = np.zeros(shape=(executions, num_episodes))
    #alg_infos = np.empty(shape=(executions,), dtype=object)
    t = time.time()
    print(f"Executing {algorithm}:")
    for i in tqdm(range(executions)):
        rewards[i], _ = algorithm(env, num_episodes, *args)
    t = time.time() - t
    print(f"  ({executions} executions of {alg_name} finished in {t:.2f} secs)")
    rewards_mean, rewards_std = rewards.mean(axis=0), rewards.std(axis=0)
    RESULTS = np.array([alg_name, rewards_mean, rewards_std], dtype=object)
    np.save(result_file_name, RESULTS, allow_pickle=True)
    return alg_name, rewards_mean, rewards_std



def plot_results(results, max_value):
    total_steps = len(results[0][1])  # no primeiro resultado (#0), pega o tamanho do array de recompensas, na segunda posição (#1)
    bestline = np.ones(total_steps) * max_value
    bestline_style = "--" 
    bestline_color = "gray"
    
    # plot moving average ctr, with x in log-scale
    for (alg_name, rewards, _) in results:
        cumulative_average = np.cumsum(rewards) / (np.arange(1, total_steps+1))
        plt.plot(cumulative_average, label=alg_name)
    plt.plot(bestline, linestyle=bestline_style, color=bestline_color)
    plt.xscale('log')
    plt.title(f"Cumulative average - x in log scale")
    plt.legend()
    plt.show()

    # plot moving average ctr, with x in linear scale
    for (alg_name, rewards, _) in results:
        cumulative_average = np.cumsum(rewards) / (np.arange(1, total_steps+1))
        plt.plot(cumulative_average, label=alg_name)
    plt.plot(bestline, linestyle=bestline_style, color=bestline_color)
    plt.title(f"Cumulative average")
    plt.legend()
    plt.show()

    for (alg_name, rewards, exec_info) in results:
        print("Summary for " + alg_name)
        print(" - total reward:", rewards.sum())
        print(" - avg reward (win rate):", rewards.sum() / total_steps)
        #print(" - extra info (algorithm-dependent):")
        #print(exec_info)
        print()

    return