import numpy as np
import gym


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
