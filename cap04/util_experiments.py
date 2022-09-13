
import time
import os
from tqdm import tqdm

import numpy as np


def repeated_exec(executions, alg_name, algorithm, env, num_episodes, *args, **kwargs):
    env_name = str(env.unwrapped).replace('<','_').replace('>','_')
    result_file_name = f"results/{env_name}-{alg_name}-episodes{num_episodes}-execs{executions}.npy"
    if os.path.exists(result_file_name):
        print("Loading results from", result_file_name)
        RESULTS = np.load(result_file_name, allow_pickle=True)
        return RESULTS
    rewards = np.zeros(shape=(executions, num_episodes))
    #alg_infos = np.empty(shape=(executions,), dtype=object)
    t = time.time()
    print(f"Executing {algorithm}:")
    for i in tqdm(range(executions)):
        rewards[i], _ = algorithm(env, num_episodes, *args, **kwargs)
    t = time.time() - t
    print(f"  ({executions} executions of {alg_name} finished in {t:.2f} secs)")
    rew_mean, rew_std = rewards.mean(axis=0), rewards.std(axis=0)
    RESULTS = np.array([alg_name, rew_mean, rew_std], dtype=object)
    np.save(result_file_name, RESULTS, allow_pickle=True)
    return alg_name, rew_mean, rew_std


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
    np.save(result_file_name, RESULTS, allow_pickle=True)
    return alg_name, rew_mean, rew_std


def test_greedy_Q_policy(env, Q, num_episodes=100, render=False, render_wait=0.01):
    """
    Avalia a política gulosa (greedy) definida implicitamente por uma Q-table.
    Ou seja, executa, em todo estado s, a ação "a = argmax Q(s,a)".
    - env: o ambiente
    - Q: a Q-table (tabela Q) que será usada
    - num_episodes: quantidade de episódios a serem executados
    - render: defina como True se deseja chamar `env.render()` a cada passo
    - render_wait: intervalo de tempo entre as chamadas a `env.render()`
    
    Retorna:
    - um par contendo o valor escalar do retorno médio por episódio e 
       a lista de retornos de todos os episódios
    """
    episode_returns = []
    total_steps = 0
    for i in range(num_episodes):
        print(f"Episode {i+1}")
        obs = env.reset()
        if render:
            env.render()
            time.sleep(render_wait)
        done = False
        episode_returns.append(0.0)
        while not done:
            action = np.argmax(Q[obs])
            obs, reward, done, _ = env.step(action)
            if render:
                env.render()
                time.sleep(render_wait)
            total_steps += 1
            episode_returns[-1] += reward
        print("- retorno:", episode_returns[-1])
    mean_return = round(np.mean(episode_returns), 1)
    print("Retorno médio (por episódio):", mean_return, end="")
    print(", episódios:", len(episode_returns), end="")
    print(", total de passos:", total_steps)
    return mean_return, episode_returns
