
import gym
import numpy as np

from crossentropy_method_v2 import run_crossentropy_method2, test_policy
from models_torch import PolicyModelCrossentropy

from util_plot import plot_result


class SimplifiedMCar(gym.Wrapper):
    def __init__(self, env, goal_x):
        super().__init__(env)
        self.goal_x = goal_x

    def reset(self):
        obs = self.env.reset()
        return obs
    
    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        if obs[0] >= self.goal_x and obs[1] > 0:
            reward = obs[1]
            done = True
        return obs, reward, done, info


if __name__ == "__main__":
    ENV_NAME, rmax = "MountainCar-v0", -100  # resultados ruins
    ENV = gym.make(ENV_NAME)

    BATCH_SIZE   = 50      # quantidade de episódios executados por época de treinamento
    PERCENT_BEST = 0.06    # percentual dos episódios (do batch) que serão selecionados

    policy = PolicyModelCrossentropy(ENV.observation_space.shape[0], [128, 256], ENV.action_space.n, lr=0.01)

    all_returns = []

    #este curriculo deu certo no pc da ufrpe
    #for goal_x, episodes in [(0.05, 500), (0.15, 1000), (10.0, 2000)]:
    
    #for goal_x, episodes, target_ret in [(-0.30, 5000, -120.0), (-0.15, 5000, -120.0), (-0.05, 5000, -110.0), (0.05, 5000, -100.0), (0.15, 5000, -90.0), (1.0, 5000, -80.0)]:
    for goal_x, episodes, target_ret in [(-0.30, 10000, -100.0), (-0.25, 10000, -100.0), (-0.20, 10000, -120.0), (-0.15, 10000, -110.0), (-0.10, 10000, -110.0), (-0.05, 10000, -120.0), (0.00, 10000, -120.0), (1.0, 5000, -80.0)]:
        print(f"TREINANDO COM goal_x = {goal_x}:")
        wrapped_env = SimplifiedMCar(ENV, goal_x)
        
        returns, policy = run_crossentropy_method2(wrapped_env, episodes, BATCH_SIZE, PERCENT_BEST, target_return=target_ret, initial_policy=policy, render=False)
        print("Últimos resultados: media =", np.mean(returns[-20:]), ", desvio padrao =", np.std(returns[-20:]))        
        plot_result(returns, rmax, None)
        all_returns.extend(returns)

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
            print(obs)
            reward += r
            steps += 1
        ENV.render()
        print("- steps:", steps)
        print("- return:", reward)

    ENV.close()
