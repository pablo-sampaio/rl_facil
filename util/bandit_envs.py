
import numpy as np


class MultiArmedBanditEnv :
    '''
    Implementação do chamado "Binary multi-armed bandit" or "Bernoulli multi-armed bandit".
    Cada ação tem uma probabilidade distinta de dar uma recompensa 1.0.
    A recompensa é sempre unitária.
    '''

    def __init__(self, actions_reward_probs=[0.1, 0.5]):
        self.num_arms = len(actions_reward_probs)
        self.arms_prob = tuple(actions_reward_probs)  # probabilidade de cada braço dar uma recompensa
        self.reset()

    def reset(self):
        self.step_count = 0
    
    def step(self, action):
        self.step_count += 1
        if (np.random.random() <= self.arms_prob[action]):
            return 1.0
        else:
            return 0.0
    
    def get_num_actions(self):
        return self.num_arms

    def get_max_mean_reward(self):
        # método para dar a informação, não deve ser usado em soluções!
        return np.max(self.arms_prob)
    
    def __repr__(self):
        return f"MultiArmedBanditEnv{self.arms_prob}"


class GaussianMultiArmedBanditEnv :
    '''
    Implementação do problema que considera uma distribuição normal (gaussiana) das recompensas.
    Cada ação dá (quase) sempre alguma recompensa não-nula.
    As recompensas seguem um distribuição de probabilidade normal/gaussina, 
    com uma média específica para cada ação e com variância 1,0.
    '''
    
    def __init__(self, actions_mean_reward=[0.1, 0.5]):
        self.num_arms = len(actions_mean_reward)
        self.arms_means = tuple(actions_mean_reward)
        self.arms_variance = 1.0  # assuming the same for every arm/action
        self.reset()

    def reset(self):
        self.step_count = 0
    
    def step(self, action):
        self.step_count += 1
        chosen_arm_mean = self.arms_means[action]
        r = np.random.randn() / np.sqrt(self.arms_variance) + chosen_arm_mean
        return r

    def get_num_actions(self):
        return self.num_arms
        
    def get_max_mean_reward(self):
        return np.max(self.arms_means)
    
    def __repr__(self):
        return f"GaussianMultiArmedBanditEnv{self.arms_means}"



if __name__ == '__main__':
    max_steps = 50

    print("Versão simples")
    env1 = MultiArmedBanditEnv()
    env1.reset()
    actions = [1, 0] * (max_steps // 2)
    for _ in range(12):
        a = actions.pop()
        r = env1.step(a)
        print(" - ação", a, ", recompensa", r)


    print("Versão gaussiana")
    env2 = GaussianMultiArmedBanditEnv()
    env2.reset()
    actions = [1, 0] * (max_steps // 2)
    for _ in range(12):
        a = actions.pop()
        r = env2.step(a)
        print(" - ação", a, ", recompensa", r)

