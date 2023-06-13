import numpy as np
from scipy.special import expit

import matplotlib.pyplot as plt


def exponential_decay(minimum_epsilon, target_step, initial_epsilon=1.0, plateau_at_min=True):
    decay_rate = (minimum_epsilon / initial_epsilon) ** (1 / target_step)
    
    def get_epsilon(current_step):
        if plateau_at_min and (current_step > target_step):
            return minimum_epsilon
        epsilon = initial_epsilon * (decay_rate ** current_step)
        return epsilon
    
    return get_epsilon

def linear_decay(minimum_epsilon, target_step, initial_epsilon=1.0, plateau_at_min=True):
    decay_rate = (initial_epsilon - minimum_epsilon) / target_step
    
    def get_epsilon(current_step):
        if plateau_at_min and (current_step > target_step):
            return minimum_epsilon
        epsilon = initial_epsilon - (decay_rate * current_step)
        return epsilon
    
    return get_epsilon

# igual ao exponencial
def logarithmic_decay(minimum_epsilon, target_step, initial_epsilon=1.0, plateau_at_min=True):
    decay_factor = (np.log(minimum_epsilon) - np.log(initial_epsilon)) / target_step

    def get_epsilon(current_step):
        if plateau_at_min and (current_step > target_step):
            return minimum_epsilon
        epsilon = initial_epsilon * np.exp(decay_factor * current_step)
        return epsilon

    return get_epsilon


def sigmoidal_decay(steepness=4.0):
    # a função sigmóide (representada por "expit") é usada para indicar o fator de "decay"
    # o steepness representa o maior valor de x a ser passado para a sigmóide, i.e. ela será usada no domínio [0; steepness]
    # o max_decay representa o maior valor de decay que seria obtido com o cálculo abaixo, sem correção
    # o máximo deveria ser 1.0, mas este só é atingido para x -> +inf
    max_decay = 2.0 * expit(steepness) - 1.0

    def sigmoidal_decay_fixed_steepness(minimum_epsilon, target_step, initial_epsilon=1.0, plateau_at_min=True):
        def get_epsilon(current_step):
            # não usei >= para calcular o minimo com o método e verificar a suavidade
            if plateau_at_min and (current_step > target_step):
                return minimum_epsilon

            # ela inicia em 0.5 (para x=0) e atinge perto de 1.0 para valores positivos altos de x
            step_progress = (current_step-1) / (target_step-1)
            decay_factor = 2.0 * expit(steepness * step_progress) - 1.0

            # fator de correção: adiciona, proporcionalmente aos passos, o que falta para atingir o máximo desejado (que é o decay=1.0)
            decay_factor += step_progress * (1.0 - max_decay)

            # atinge o mínimo para decay_factor==1.0
            epsilon = initial_epsilon + (minimum_epsilon - initial_epsilon) * decay_factor
            
            return epsilon

        return get_epsilon

    return sigmoidal_decay_fixed_steepness


if __name__ == "__main__":
    # Example usage
    minimum_epsilon = 0.1
    target_step = 7000

    # CHOOSE the type of decay
    #get_epsilon_fn = exponential_decay(minimum_epsilon, target_step, initial_epsilon=1.0)
    #get_epsilon_fn = linear_decay(minimum_epsilon, target_step, initial_epsilon=1.0)
    #get_epsilon_fn = logarithmic_decay(minimum_epsilon, target_step, initial_epsilon=1.0)
    get_epsilon_fn = sigmoidal_decay(5.0)(minimum_epsilon, target_step, initial_epsilon=1.0)

    # Get epsilon at step 500
    epsilon_500 = get_epsilon_fn(500)
    print(f"Epsilon at step 500: {epsilon_500}")

    # Get epsilon at the target step (should be the same as the desired minimum)
    epsilon_1000 = get_epsilon_fn(target_step)
    print(f"Epsilon at step {target_step}: {epsilon_1000}")

    # Plot a graph for 10k steps, showing different decay schemes
    step_sequence = range(1, 10000 + 1)

    #for label, scheme_builder in [("exponential", exponential_decay), ("linear", linear_decay),  ("sigmoidal-1", sigmoidal_decay(1.0)), ("sigmoidal-3", sigmoidal_decay(3.0)), ("sigmoidal-4", sigmoidal_decay(4.0)), ("sigmoidal-8", sigmoidal_decay(8.0))]: #, ("logarithmic", logarithmic_decay)]:
    for label, scheme_builder in [("exponential", exponential_decay), ("linear", linear_decay),  ("sigmoidal", sigmoidal_decay(3.0))]:
        get_epsilon_fn = scheme_builder(minimum_epsilon, target_step)
        plt.plot(step_sequence, [get_epsilon_fn(t) for t in step_sequence], label=label)

    plt.legend()
    plt.show()
