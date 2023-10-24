
import gymnasium as gym
import numpy as np

from .experiments import repeated_exec


# Esta é a política. Neste caso, escolhe uma ação com base nos valores
# da tabela Q, usando uma estratégia epsilon-greedy.
def epsilon_greedy_random_tiebreak(qtable, state, epsilon):
    q_state = qtable[state]
    num_actions = len(q_state)
    if np.random.random() < epsilon:
        return np.random.randint(0, num_actions)
    else:
        return np.random.choice(np.where(q_state == q_state.max())[0])


def record_video_qtable(env_name, qtable, length=500, folder='videos/', prefix='rl-video', epsilon=0.0):
    """
    - env_name: a string do ambiente cadastrada no gymnasium ou a classe do ambiente ou função que o instancia
    - qtable: a tabela Q (Q-table) na forma de array bidimensional
    - length: número de passos do ambiente usados no vídeo
    - prefiz: prefixo do nome dos arquivos de vídeo
    - folder: pasta dos arquivos de vídeo
    - epsilon:  valor do parâmetro para a escolha epsilon-greedy da ação
    """
    if isinstance(env_name, str):
        env = gym.make(env_name, render_mode="rgb_array")
    else:
        env = env_name()
    rec_env = gym.wrappers.RecordVideo(env, folder, episode_trigger=lambda i : True, video_length=length, name_prefix=prefix)
    num_steps = 0
    while num_steps < length:
        state, _ = rec_env.reset()
        num_steps += 1
        done = False
        while (not done) and (num_steps < length):
            action = epsilon_greedy_random_tiebreak(qtable, state, epsilon)
            state, r, termi, trunc, _ = rec_env.step(action)
            done = termi or trunc
            num_steps += 1
    rec_env.close()


def evaluate_qtable(env, qtable, num_episodes=100, epsilon=0.0, verbose=False):
    """
    Avalia a política epsilon-greedy definida implicitamente por uma Q-table.
    Por padrão, executa com epsilon=0.0; ou seja, executa, em todo estado s, a ação "a = argmax Q(s,a)".
    - env: o ambiente
    - qtable: a Q-table (tabela Q) que será usada
    - num_episodes: quantidade de episódios a serem executados
    - epsilon: valor do parâmetro para a escolha epsilon-greedy da ação
    
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
            if episode_step == 1500:
                print(f"Too long episode, truncating at step {episode_step}.")
                break
            episode_returns[-1] += reward
        print(episode_returns[-1])
    
    mean_return = round(np.mean(episode_returns), 1)
    print("Retorno médio (por episódio):", mean_return, end="")
    print(", episódios:", len(episode_returns), end="")
    print(", total de passos:", total_steps)

    return mean_return, episode_returns


def repeated_exec_epsilon_greedy_qtable(executions, alg_name, qtable, env, num_iterations, epsilon=0.0):
    def run_q_greedy(env, num_steps):
        state = env.reset()
        rewards = []
        for i in range(num_steps):
            a = epsilon_greedy_random_tiebreak(qtable, state, epsilon)
            state, r, done, _ = env.step(a)
            rewards.append(r)
            if done:
                state = env.reset()
        return rewards, None
    
    return repeated_exec(executions, alg_name, run_q_greedy, env, num_iterations)
