import numpy as np
import gymnasium as gym


def convert_to_flattened_index(indices, dimensions):
    if len(indices) != len(dimensions):
        raise ValueError("Number of indices must match the number of dimensions")

    flattened_index = 0
    for i in range(len(indices)):
        if indices[i] < 0 or indices[i] >= dimensions[i]:
            raise ValueError(f"Value out of bounds at index {i}: {indices[i]}")
        flattened_index = flattened_index * dimensions[i] + indices[i]

    return flattened_index

def convert_from_flattened_index(flattened_index, dimensions):
    indices = [0] * len(dimensions)
    for i in range(len(dimensions)-1, -1, -1):
        indices[i] = flattened_index % dimensions[i]
        flattened_index = flattened_index // dimensions[i]
    return indices


# Converte um espaço contínuo (de ações ou observações) em um espaço discreto.
class BoxSpaceDiscretizer:
    def __init__(self, env_space, bins_per_dimension):
        assert isinstance(env_space, gym.spaces.Box)
        assert len(env_space.shape) == 1, "Only 1-D observations are supported"
        assert env_space.shape[0] == len(bins_per_dimension), "Number of bins must match the dimensions of the space"

        self.bins_per_dim = bins_per_dimension.copy()
        self.full_intervals_per_dim = []
        self.intervals_per_dim = []
        self.total_bins = 1
        
        for i, bins in enumerate(bins_per_dimension):
            min_value = env_space.low[i] if not np.isneginf(env_space.low[i]) else 2*np.finfo(np.float64).min
            max_value = env_space.high[i] if not np.isposinf(env_space.high[i]) else np.finfo(np.float64).max/2

            # cria o 'linspace' do valor inicial ao final
            full_linspace = np.linspace(min_value, max_value, bins+1, endpoint=True)
            #print(f">> Dim {i}: full_linspace: {full_linspace}")
            
            # adiciona o 'linspace' com o valor inicial e o final removidos, por conta do funcionamento do np.digitize():
            #  - valor anterior ao "novo" inicial -> índice "0"
            #  - valor posterior ao "novo" final -> índice "bins-1"
            self.full_intervals_per_dim.append( full_linspace )
            self.intervals_per_dim.append( full_linspace[1:-1] )
            
            self.total_bins *= bins

    def to_bins(self, original_value):
        bin_vector = [np.digitize(x=original_value[i], bins=intervals)
                      for i, intervals in enumerate(self.intervals_per_dim)]
        return bin_vector

    def to_single_bin(self, original_value):
        bin_vector = [np.digitize(x=original_value[i], bins=intervals)
                      for i, intervals in enumerate(self.intervals_per_dim)]
        return convert_to_flattened_index(bin_vector, self.bins_per_dim)

    def from_bins(self, bin_vector):
        original_value = [ (intervals[bin_vector[dim]] + intervals[bin_vector[dim]+1])/2 
                           for dim, intervals in enumerate(self.full_intervals_per_dim) ]
        return original_value

    # it will be useful to convert an entire action into indices of each dimension
    # and then, for the average values of each interval
    def from_single_bin(self, bin_index):
        bin_vector = convert_from_flattened_index(bin_index, self.bins_per_dim)
        return self.from_bins(bin_vector)

    def get_total_bins(self):
        return self.total_bins


class ObservationDiscretizerWrapper(gym.ObservationWrapper):
    '''Classe para converter espaços contínuos em espaços discretos.

    Esta classe converte ambientes de observações (estados) contínuos em ambientes de estados
    discretos. Especificamente, ele converte representações dadas na forma de array de valores float
    em um único inteiro $\geq$ não-negativo (>=0).
    
    Precisa passar para o construtor uma lista que informa em quantos "bins" vai ser discretizada 
    cada dimensão (ou seja, cada valor float) do espaço de estados original.
    '''
    
    def __init__(self, env : gym.Env, BINS_PER_DIMENSION):
        super().__init__(env)
        # cria um BoxSpaceDiscretizer para converter um array de valores float em um único inteiro >= 0
        # precisa dizer em quantos "bins" vai ser discretizada cada dimensão
        self.discretizer = BoxSpaceDiscretizer(env.observation_space, BINS_PER_DIMENSION)
        self.observation_space = gym.spaces.Discrete(self.discretizer.get_total_bins())

    def observation(self, obs):
        return self.discretizer.to_single_bin(obs)


class FromDiscreteTupleToDiscreteObs(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = gym.spaces.Discrete(self._calculate_discrete_size(env.observation_space))

    def _calculate_discrete_size(self, observation_space):
        size = 1
        assert isinstance(observation_space, gym.spaces.Tuple)
        self.dimensions = []
        for space in observation_space:
            size *= space.n
            self.dimensions.append(space.n)
        return size

    def observation(self, observation):
        return convert_to_flattened_index(observation, self.dimensions)
