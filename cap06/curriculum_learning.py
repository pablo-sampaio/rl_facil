
import gym
import numpy as np

from crossentropy_method_v2 import run_crossentropy_method_x, test_policy
from models_torch import PolicyModelCrossentropy

from util_plot import plot_result
from wrappers import DiscreteObservationWrapper


class SimplifiedMCar(gym.Wrapper):
    def __init__(self, env, goal_x):
        super().__init__(env)
        self.desired_x = goal_x

    def reset(self):
        obs = self.env.reset()
        self.dist = 0
        return obs
    
    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        if obs[0] >= self.desired_x and obs[1] > 0:
            reward = obs[1]
            done = True
        return obs, reward, done, info


if __name__ == "__main__":
    ENV_NAME, rmax = "MountainCar-v0", -100  # resultados ruins
    ENV = gym.make(ENV_NAME)

    BATCH_SIZE = 20        # quantidade de episódios executados por época de treinamento
    PERCENT_BEST = 0.2     # percentual dos episódios (do batch) que serão selecionados

    policy = PolicyModelCrossentropy(ENV.observation_space.shape[0], [128, 256], ENV.action_space.n, lr=0.01)

    all_returns = []

    for goal_x, episodes in [(0.05, 500), (0.15, 1000), (10.0, 2000)]: #(0.15, 1500), (0.20, 1500), (10.0, 3000)]:
        print(f"TREINANDO COM goal_x = {goal_x}:")
        wrapped_env = SimplifiedMCar(ENV, goal_x)
        
        returns, policy = run_crossentropy_method_x(wrapped_env, episodes, BATCH_SIZE, PERCENT_BEST, policy_model=policy, render=True)
        all_returns.extend(returns)

        print("Últimos resultados: media =", np.mean(returns[-20:]), ", desvio padrao =", np.std(returns[-20:]))        
        #plot_result(returns, rmax, None)

    plot_result(all_returns, rmax, None)

    # Executa alguns episódios de forma NÃO-determinística e imprime um sumário
    #test_policy(ENV, policy, False, 5, render=True)

    # Expandindo aqui a execução de alguns episódios de forma DETERMINÍSTICA, para fins didáticos
    for i in range(5):
        print(f"TEST EPISODE {i+1}")
        obs = ENV.reset()
        done = False
        reward = 0.0
        steps = 0
        while not done:
            ENV.render()
            action = policy.best_action(obs) # faz so a acao de maior probabilidade
            obs, r, done, _ = ENV.step(action)
            #print(obs)
            reward += r
            steps += 1
        ENV.render()
        print("- steps:", steps)
        print("- return:", reward)

    ENV.close()
