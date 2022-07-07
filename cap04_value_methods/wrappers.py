import time

import numpy as np
import gym
import matplotlib.pyplot as plt


def test_Q_policy(env, Q, num_episodes=100, render=False):
    """
    Evaluate a policy (typically, a RL trained agent)
    :param env: o ambiente
    :param Q: a Q-table (tabela Q) que será usada como uma política gulosa (com a = argmax(Q[obs]))
    :param num_episodes: (int) quantidade de episódios
    :param render: defina como True se deseja chamar env.render() a cada passo
    :return: (tuple) recompensa média por episódio e a lista de recompensas acumuladas de todos os episódios
    """
    episode_rewards = []
    total_steps = 0
    for i in range(num_episodes):
        obs = env.reset()
        if render:
            env.render()
            time.sleep(0.02)
        done = False
        episode_rewards.append(0.0)
        while not done:
            action = np.argmax(Q[obs])
            obs, reward, done, _ = env.step(action)
            if render:
                env.render(mode="ansi")
                time.sleep(0.02)
            total_steps += 1
            episode_rewards[-1] += reward

    mean_reward = round(np.mean(episode_rewards), 1)
    print("Mean reward:", mean_reward, end="")
    print(", Num episodes:", len(episode_rewards), end="")
    print(", Num steps:", total_steps)
    return mean_reward, episode_rewards


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
