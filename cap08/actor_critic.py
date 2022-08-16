
################
# Algoritmo mais simples da família ACTOR-CRITIC (subfamília da policy-gradient).
# Baseado em códigos de "Lazy Programmer" do curso do Udemy e codigos do livro de M. Lapan.
################

import gym
import numpy as np

from models_torch_pg import PolicyModelPG, ValueModel, test_policy
from util_plot import plot_result


# Algoritmo actor-critic básico
def run_basic_actor_critic(env, total_steps, gamma, initial_policy=None, initial_vmodel=None, target_return=None):
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
    episodes = 0
    steps = 0

    next_state = env.reset()
    ep_return = 0.0

    #while episodes < total_episodes:
    while steps < total_steps:
        state = next_state

        # 1. Faz 1 passo
        action = policy_model.sample_action(state)
        next_state, r, done, info = env.step(action)
        ep_return += r
        steps += 1

        # 2. Treina a política
        G_est = r + gamma * Vmodel.predict(next_state)
        advantage = G_est - Vmodel.predict(state)
        policy_model.partial_fit([state], [action], [advantage])

        # 3. Treina o modelo de V(.),
        Vmodel.partial_fit([state], [G_est])
        
        if done:
            all_returns.append(ep_return)
            episodes += 1 
            reward_m = np.mean(all_returns[-50:])

            if target_return is not None and reward_m >= target_return:
                print("-> target reached!")
                break

            print("step %d / ep %d: reward=%.2f, reward_mean=%.2f" % (steps, episodes, ep_return, reward_m))

            next_state = env.reset()
            ep_return = 0.0
    
    print("step %d / ep %d: return_mean_50=%.2f - end of training!" % (steps, episodes, reward_m))
    return all_returns, policy_model


if __name__ == "__main__":
    ENV_NAME, rmax = "CartPole-v1", 300
    #ENV_NAME, rmax = "Acrobot-v1", 0
    ENV = gym.make(ENV_NAME)

    # ATENÇÃO para a mudança: agora, o critério de parada é pela quantidade de passos
    # e não pela quantidade de episódios (estamos seguindo o padrão da área)
    NUM_STEPS = 10000
    GAMMA    = 0.99
    
    inputs = ENV.observation_space.shape[0]
    outputs = ENV.action_space.n

    policy_model = PolicyModelPG(inputs, [256,256], outputs, lr=1e-5)
    Vmodel = ValueModel(inputs, [256,32], lr=2e-4)

    returns, policy = run_basic_actor_critic(ENV, NUM_STEPS, GAMMA, initial_policy=policy_model, initial_vmodel=Vmodel, target_return=rmax-100)

    # Exibe um gráfico episódios x retornos (não descontados)
    plot_result(returns, rmax, window=50)

    # Executa alguns episódios de forma NÃO-determinística e imprime um sumário
    test_policy(ENV, policy, False, 5, render=True)
