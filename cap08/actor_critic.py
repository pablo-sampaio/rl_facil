
################
# Algoritmo mais simples da família ACTOR-CRITIC (subfamília da policy-gradient).
# Baseado em códigos de "Lazy Programmer" do curso do Udemy e codigos do livro de M. Lapan.
# 
# Para monitorar com Tensorboard, rodar a partir do diretorio deste projeto:
# tensorboard --logdir runs --host localhost
################

import gym
import numpy as np

from models_torch_pg import PolicyModelPG, PolicyModelPGWithExploration, ValueModel, test_policy
from util_plot import plot_result


# Alguns wrappers para o CartPole
class PositiveRewardForCenteredCartPole(gym.Wrapper):
    def __init__(self, env):
        assert env.unwrapped.spec.id == "CartPole-v1"
        super().__init__(env)
    
    def step(self, action):
        observation, reward, d, i = self.env.step(action)
        if reward > 0:
            reward += reward - min(2.4,abs(observation[0])) / 2.4
            reward = reward / 2.0
        return observation, reward, d, i

class NegativeRewardWhenPoleFalls(gym.Wrapper):
    def __init__(self, env, fall_reward=-200):
        assert env.unwrapped.spec.id == "CartPole-v1"
        super().__init__(env)
        self.fall_reward = fall_reward
    
    def step(self, action):
        observation, reward, done, i = self.env.step(action)
        # if ended because the pole fell down
        if done and self.env._elapsed_steps < self.env._max_episode_steps:
            reward = self.fall_reward
        return observation, reward, done, i


# Algoritmo actor-critic básico
def run_basic_actor_critic(env, total_episodes, gamma, initial_policy=None, initial_vmodel=None, target_return=None):
    obs_size = env.observation_space.shape[0]
    n_actions = env.action_space.n

    if initial_policy is None:
        policy_model = PolicyModelPG(obs_size, [256], n_actions, lr=0.0001)
    else:
        policy_model = initial_policy.clone()

    if initial_vmodel is None:
        Vmodel = ValueModel(obs_size, [128], lr=0.008)
    else:
        Vmodel = initial_vmodel

    if target_return is None:
        target_return = float("inf")

    all_returns = []
    steps = 0

    state = env.reset()
    ep_return = 0.0
    episodes = 0
    while episodes < total_episodes:
        # 1. Faz 1 passo
        action = policy_model.sample_action(state)
        next_state, r, done, info = env.step(action)
        ep_return += r
        steps += 1

        # 2. Treina a política
        G = r + gamma * Vmodel.predict(next_state)
        advantage = G - Vmodel.predict(state)
        policy_model.partial_fit([state], [action], [advantage])

        # 3. Treina o modelo de V(.),
        Vmodel.partial_fit([state], [G])
        
        if done:
            all_returns.append(ep_return)
            episodes += 1 
            reward_m = np.mean(all_returns[-50:])

            if target_return is not None and reward_m >= target_return:
                print("- episode %d (step %d): return_mean_50=%.2f, target reached!" % (episodes, steps, reward_m))
                break

            print("step %d \ ep %d: reward=%.2f, reward_mean=%.2f, episodes=%d" % (steps, episodes, ep_return, reward_m, episodes))

            state = env.reset()
            ep_return = 0.0
    
    return all_returns, policy_model


if __name__ == "__main__":
    ENV_NAME, rmax = "CartPole-v1", 500
    #ENV_NAME, rmax = "Acrobot-v1", 0
    ENV = gym.make(ENV_NAME)

    EPISODES = 1000
    GAMMA    = 0.95
    
    inputs = ENV.observation_space.shape[0]
    outputs = ENV.action_space.n

    policy_model = PolicyModelPG(inputs, [256, 256], outputs, lr=0.001)
    Vmodel = ValueModel(inputs, [128], lr=0.002)

    returns, policy = run_basic_actor_critic(ENV, EPISODES, GAMMA, initial_policy=policy_model, initial_vmodel=Vmodel, target_return=rmax-100)

    # Exibe um gráfico episódios x retornos (não descontados)
    plot_result(returns, rmax, window=50)

    # Executa alguns episódios de forma NÃO-determinística e imprime um sumário
    test_policy(ENV, policy, False, 5, render=True)
