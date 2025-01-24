
import numpy as np
import gymnasium as gym
import torch


# Faz uma escolha epsilon-greedy
def epsilon_greedy_qnet(qnet, env, state, epsilon):
    if np.random.random() < epsilon:
        action = env.action_space.sample()
    else:
        state_a = np.array([state], copy=False)
        state_v = torch.tensor(state_a, dtype=torch.float32)
        q_vals_v = qnet(state_v)
        _, act_v = torch.max(q_vals_v, dim=1)
        action = int(act_v.item())
    return action


def record_video_qnet(env_name, qnet, episodes=3, folder='videos/', prefix='rl-video', epsilon=0.0):
    """
    Grava um vídeo a partir de uma política epsilon-greedy definida pela 'qtable' e pelo valor de 'epsilon'.
    - env_name: A string do ambiente cadastrada no gymnasium ou uma instância da classe. Ao final, o ambiente é fechado (função `close()`).
    - qnet: A rede neural que representa a função Q.
    - episodes: Número de episódios completos que serão executados.
    - prefiz: Prefixo do nome dos arquivos de vídeo.
    - folder: Pasta onde os arquivos de vídeo serão salvos.
    - epsilon: Valor do parâmetro da política "epsilon-greedy" usada para escolher as ações.
    """
    if isinstance(env_name, str):
        env = gym.make(env_name, render_mode="rgb_array")
    else:
        env = env_name
    rec_env = gym.wrappers.RecordVideo(env, folder, episode_trigger=lambda i : True, name_prefix=prefix)
    num_steps = 0
    for epi in range(episodes):
        state, _ = rec_env.reset()
        num_steps += 1
        epi_reward = 0.0
        done = False
        while not done:
            action = epsilon_greedy_qnet(qnet, env, state, epsilon)
            state, r, termi, trunc, _ = rec_env.step(action)
            done = termi or trunc
            num_steps += 1
            epi_reward += r
        print(f"Episode {epi}: {num_steps} steps / return {epi_reward:.2f}")
    rec_env.close()
    env.close()


def evaluate_qnet_policy(env, q_network, num_episodes=5, epsilon=0.0, verbose=False):
    episode_returns = []
    total_steps = 0

    for epi in range(num_episodes):
        if verbose:
            print(f"Episódio {epi+1}: ", end="")

        state, _ = env.reset()
        episode_step = 0
        total_reward = 0
        done = False

        while not done:
            action = epsilon_greedy_qnet(q_network, env, state, epsilon)

            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode_step += 1
            total_steps += 1
            total_reward += reward

            if episode_step == 1500:
                print(f"Too long episode, truncating at step {episode_step}.")
                break

        episode_returns.append(total_reward)
        if verbose:
            print(total_reward)

    mean_return = np.mean(episode_returns)
    print(f"Retorno médio (por episódio): {mean_return:.2f}, episódios: {len(episode_returns)}, total de passos: {total_steps}")

    return mean_return, episode_returns


'''
# TODO: remover (pois já tem a função acima) ou aproveitar a forma diferente de capturar video
def test_qnetwork_policy(env, Qpolicy, epsilon=0.0, num_episodes=5, videorec=None):
    """
    Avalia a política `Qpolicy` escolhendo de forma epsilon-greedy.
    - env: o ambiente
    - Qpolicy: um rede que representa a função Q(s,a) para ser usada como política epsilon-greedy
    - epsilon: probabilidade de ser feita uma escolha aleatória das ações
    - num_episodes: quantidade de episódios a serem executados
    - video: passe uma instância de VideoRecorder (do gym), se desejar gravar
    Retorna:
    - um par contendo o valor escalar do retorno médio por episódio e 
       a lista de retornos de todos os episódios
    """
    episodes_returns = []
    total_steps = 0
    num_actions = env.action_space.n
    for i in range(num_episodes):
        obs = env.reset()
        if videorec is not None:
            videorec.capture_frame()
        done = False
        steps = 0
        episodes_returns.append(0.0)
        while not done:
            if epsilon > 0.0 and np.random.rand() < epsilon:
                action = np.random.choice(num_actions)
            else:
                state_a = np.array([obs], copy=False)
                state_v = torch.tensor(state_a)
                q_vals_v = Qpolicy(state_v)
                _, act_v = torch.max(q_vals_v, dim=1)
                action = int(act_v.item())
            obs, reward, done, _ = env.step(action)
            if videorec is not None:
                videorec.capture_frame()
            total_steps += 1
            episodes_returns[-1] += reward
            steps += 1
        print(f"EPISODE {i+1}")
        print("- steps:", steps)
        print("- return:", episodes_returns[-1])
    mean_return = round(np.mean(episodes_returns), 1)
    print("RESULTADO FINAL: média (por episódio):", mean_return, end="")
    print(", episódios:", len(episodes_returns), end="")
    print(", total de passos:", total_steps)
    if videorec is not None:
        videorec.close()
    return mean_return, episodes_returns
'''
