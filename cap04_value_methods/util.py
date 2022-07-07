
from time import time
from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt

import os

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

def plot_results(results, cumulative=False, x_log_scale=False):
    total_steps = len(results[0][1])  # no primeiro resultado (#0), pega o tamanho do array de recompensas, que fica na segunda posição (#1)

    if not cumulative:
        # plot all the rewards, with x linear
        for (alg_name, rewards, _) in results:
            plt.plot(rewards, label=alg_name)
        if x_log_scale:
            plt.xscale('log')
        plt.title(f"Raw Rewards")
        plt.legend()
        plt.show()

        # plot the rewards smoothed by a moving average with window 50, with x linear
        for (alg_name, rewards, _) in results:
            plt.plot(moving_average(rewards,50), label=alg_name+"-smoothed")
        if x_log_scale:
            plt.xscale('log')
        plt.title(f"Smoothed 50-reward")
        plt.legend()
        plt.show()

    else:
        # plot moving average ctr
        for (alg_name, rewards, _) in results:
            cumulative_average = np.cumsum(rewards) / (np.arange(1, total_steps+1))
            plt.plot(cumulative_average, label=alg_name)
        if x_log_scale:
            plt.xscale('log')
        plt.title(f"Cumulative Average")
        plt.legend()
        plt.show()

    for (alg_name, rewards, exec_info) in results:
        print("Summary for " + alg_name)
        print(" - sum rewards (all episodes):", rewards.sum())
        print(" - extra info (algorithm-dependent):")
        print(exec_info)
        print()

    return


def repeated_exec(executions, alg_name, algorithm, *args):
    result_file_name = f"results/{alg_name}-execs{executions}.npy"
    if os.path.exists(result_file_name):
        print("Loading results from", result_file_name)
        RESULTS = np.load(result_file_name, allow_pickle=True)
        return RESULTS
    #env = args[0]
    num_steps = args[1]
    rewards = np.zeros(shape=(executions, num_steps))
    alg_infos = np.empty(shape=(executions,), dtype=object)
    t = time()
    print(f"Executing {algorithm}:")
    for i in tqdm(range(executions)):
        rewards[i], alg_infos[i] = algorithm(*args)
    t = time() - t
    print(f"  ({executions} executions of {alg_name} finished in {t:.2f} secs)")
    RESULTS = np.array([alg_name, rewards.mean(axis=0), alg_infos], dtype=object)
    np.save(result_file_name, RESULTS, allow_pickle=True)
    return alg_name, rewards.mean(axis=0), alg_infos
