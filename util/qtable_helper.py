
import gymnasium as gym
import numpy as np


# Esta função define uma política, em função da tabela Q e do epsilon
# Escolhe a ação gulosa (greedy) com probabilidade 1-epsilon e uma ação aleatória com probabilidade epsilon.
def epsilon_greedy(Q, state, epsilon):
    Q_state = Q[state]
    num_actions = len(Q_state)
    if np.random.random() < epsilon:
        return np.random.randint(0, num_actions)
    else:
        # em caso de empates, retorna sempre o menor índice -- mais eficiente, porém não é bom para alguns ambientes
        return np.argmax(Q_state)


# Esta função define uma política, em função da tabela Q e do epsilon.
# Como a anterior, mas aleatoriza a escolha em caso de haver mais de uma opção gulosa empatada.
def epsilon_greedy_random_tiebreak(qtable, state, epsilon):
    q_state = qtable[state]
    num_actions = len(q_state)
    if np.random.random() < epsilon:
        return np.random.randint(0, num_actions)
    else:
        # retorna uma ação de valor máximo -- aleatoriza em caso de empates
        return np.random.choice(np.where(q_state == q_state.max())[0])


def _delete_files(folder, prefix, suffix):
    import os
    # check if folder exists
    if not os.path.exists(folder):
        return
    # list files and delete all files with the given prefix and the given suffix
    for file in os.listdir(folder):
        if file.startswith(prefix) and file.endswith(suffix):
            os.remove(os.path.join(folder, file))


def record_video_qtable(env_name, qtable, episodes=2, folder='videos/', prefix='rl-video', epsilon=0.0, max_episode_length=500):
    """
    Grava um vídeo a partir de uma política epsilon-greedy definida pela 'qtable' e pelo valor de 'epsilon'.
    - env_name: A string do ambiente cadastrada no gymnasium ou uma instância da classe. Ao final, o ambiente é fechado (função `close()`).
    - qtable: A tabela Q (Q-table) na forma de array bidimensional, com linha representando estados e colunas representando ações.
    - length: Número de passos do ambiente usados no vídeo.
    - prefiz: Prefixo do nome dos arquivos de vídeo.
    - folder: Pasta onde os arquivos de vídeo serão salvos.
    - epsilon: Valor do parâmetro da política "epsilon-greedy" usada para escolher as ações.
    """
    if isinstance(env_name, str):
        env = gym.make(env_name, render_mode="rgb_array")
    else:
        env = env_name
    
    # delete .mp4 files with the given prefix from the folder
    _delete_files(folder, prefix, ".mp4")
    
    rec_env = gym.wrappers.RecordVideo(env, folder, episode_trigger=lambda i : True, video_length=max_episode_length, name_prefix=prefix)
    for _ in range(episodes):
        state, _ = rec_env.reset()
        ep_steps = 0
        done = False
        while (not done) and (ep_steps < max_episode_length-1):  # porque o reset conta no tamanho do vídeo
            action = epsilon_greedy_random_tiebreak(qtable, state, epsilon)
            state, _, termi, trunc, _ = rec_env.step(action)
            done = termi or trunc
            ep_steps += 1
    rec_env.close()
    env.close()


def evaluate_qtable_policy(env, qtable, num_episodes=100, epsilon=0.0, verbose=False):
    """
    Avalia a política epsilon-greedy definida implicitamente por uma Q-table.
    Por padrão, executa com epsilon=0.0; ou seja, executa, em todo estado s, escolhe a ação "a = argmax Q(s,_)".
    - env: O ambiente instanciado. Ele não é fechado ao final.
    - qtable: A Q-table (tabela Q) que será usada.
    - num_episodes: Quantidade de episódios a serem executados.
    - epsilon: Valor do parâmetro para a escolha epsilon-greedy da ação.
    
    Retorna:
    - um par contendo:
       -  o valor escalar do retorno médio por episódio 
       -  e a lista de retornos de todos os episódios
    """
    episode_returns = []
    total_steps = 0

    for i in range(num_episodes):
        if verbose:
            print(f"Episódio {i+1}: ", end="")
        state, _ = env.reset()
        done = False
        episode_step = 0
        episode_returns.append(0.0)
        
        while not done:
            action = epsilon_greedy_random_tiebreak(qtable, state, epsilon)
            state, reward, termi, trunc, _ = env.step(action)
            done = termi or trunc
            episode_step += 1
            total_steps += 1
            episode_returns[-1] += reward
            if episode_step == 1500:
                print(f"Too long episode, truncating at step {episode_step}.")
                break
        if verbose:
            print(episode_returns[-1])
    
    mean_return = np.mean(episode_returns)
    print(f"Retorno médio (por episódio): {mean_return:.2f}, episódios: {len(episode_returns)}, total de passos: {total_steps}")

    return mean_return, episode_returns


def repeated_exec_qtable_policy(executions, alg_name, qtable, env, num_iterations, epsilon=0.0):
    """
    Executa várias vezes uma política epsilon-greedy definida com a qtable dada.
    Internamente usa o `util.experiments.repeated_exec()`.
    """
    from util.experiments import repeated_exec

    def run_q_greedy(env, num_steps):
        state, _ = env.reset()
        rewards = []
        for i in range(num_steps):
            a = epsilon_greedy_random_tiebreak(qtable, state, epsilon)
            state, r, terminated, truncated, _ = env.step(a)
            done = terminated or truncated
            rewards.append(r)
            if done:
                state, _ = env.reset()
        return rewards, None
    
    return repeated_exec(executions, alg_name, run_q_greedy, env, num_iterations)
