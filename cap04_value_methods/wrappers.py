import time

import numpy as np
import gym
import matplotlib.pyplot as plt


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
                time.sleep(0.02)
            total_steps += 1
            episode_returns[-1] += reward

    mean_return = round(np.mean(episode_returns), 1)
    print("Retorno médio (por episódio):", mean_return, end="")
    print(", episódios:", len(episode_returns), end="")
    print(", total de passos:", total_steps)
    return mean_return, episode_returns


def save_rewards_plot(rewards, ymax_suggested, filename):
    # alternative: a moving average
    avg_every100 = [np.mean(rewards[i:i+100])
                    for i in range(0, len(rewards), 100)]
    plt.plot(range(1, len(rewards)+1, 100), avg_every100)
    plt.xlabel('Episodes')
    plt.ylabel('Average Reward')
    ymax = np.max([ymax_suggested, np.max(avg_every100)])
    plt.ylim(top=ymax)
    plt.title('Average Reward vs Episodes')
    plt.savefig(filename)
    print("Arquivo salvo:", filename)
    plt.close()


class GeneralDiscretizer:
    def __init__(self, env, bins_per_dimension):
        self.bins_per_dim = bins_per_dimension.copy()
        self.intervals_per_dim = []
        self.total_bins = 1
        for i, bins in enumerate(bins_per_dimension):
            self.intervals_per_dim.append(
                np.linspace(env.observation_space.low[i], env.observation_space.high[i], bins+1) )
            self.total_bins *= bins

    def to_single_bin(self, state):
        bin_vector = [(np.digitize(x=state[i], bins=intervals) - 1)
                      for i, intervals in enumerate(self.intervals_per_dim)]
        # print(bin_vector)
        return self._bin_vector_to_single_bin(bin_vector, len(bin_vector)-1)

    def _bin_vector_to_single_bin(self, vector, index):
        if index < 0:
            return 0
        return vector[index] + self.bins_per_dim[index] * self._bin_vector_to_single_bin(vector, index-1)

    def get_total_bins(self):
        return self.total_bins


class DiscreteObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env, BINS_PER_DIMENSION):
        super().__init__(env)
        # Create a state discretizer, to turn float/real values into natural values
        self.discretizer = GeneralDiscretizer(env, BINS_PER_DIMENSION)
        self._observation_space = gym.spaces.Discrete(self.discretizer.get_total_bins())

    def observation(self, obs):
        return self.discretizer.to_single_bin(obs)
