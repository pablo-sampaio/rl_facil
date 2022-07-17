
import time
import os
from tqdm import tqdm

import numpy as np


def repeated_exec(executions, alg_name, algorithm, env, num_steps, *args):
    env_name = str(env.unwrapped).replace('<','_').replace('>','_')
    result_file_name = f"results/{env_name}-{alg_name}-execs{executions}.npy"
    if os.path.exists(result_file_name):
        print("Loading results from", result_file_name)
        RESULTS = np.load(result_file_name, allow_pickle=True)
        return RESULTS
    #num_steps = args[1]
    rewards = np.zeros(shape=(executions, num_steps))
    alg_infos = np.empty(shape=(executions,), dtype=object)
    t = time.time()
    print(f"Executing {algorithm}:")
    for i in tqdm(range(executions)):
        rewards[i], alg_infos[i] = algorithm(env, num_steps, *args)
    t = time.time() - t
    print(f"  ({executions} executions of {alg_name} finished in {t:.2f} secs)")
    RESULTS = np.array([alg_name, rewards.mean(axis=0), alg_infos], dtype=object)
    np.save(result_file_name, RESULTS, allow_pickle=True)
    return alg_name, rewards.mean(axis=0), alg_infos


def test_greedy_Q_policy(env, Q, num_episodes=100, render=False):
    """
    Avalia a política gulosa (greedy) definida implicitamente por uma Q-table.
    Ou seja, executa, em todo estado s, a ação "a = argmax Q(s,a)".
    - env: o ambiente
    - Q: a Q-table (tabela Q) que será usada
    - num_episodes: quantidade de episódios a serem executados
    - render: defina como True se deseja chamar env.render() a cada passo
    
    Retorna:
    - um par contendo o valor escalar do retorno médio por episódio e 
       a lista de retornos de todos os episódios
    """
    episode_returns = []
    total_steps = 0
    for i in range(num_episodes):
        obs = env.reset()
        if render:
            env.render()
            time.sleep(0.02)
        done = False
        episode_returns.append(0.0)
        while not done:
            action = np.argmax(Q[obs])
            obs, reward, done, _ = env.step(action)
            if render:
                env.render(mode="ansi")
            total_steps += 1
            episode_returns[-1] += reward

    mean_return = round(np.mean(episode_returns), 1)
    print("Retorno médio (por episódio):", mean_return, end="")
    print(", episódios:", len(episode_returns), end="")
    print(", total de passos:", total_steps)
    return mean_return, episode_returns
