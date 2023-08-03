
################
# Algoritmo mais simples da família ACTOR-CRITIC (subfamília da policy-gradient).
# Baseado em códigos de "Lazy Programmer" do curso do Udemy e codigos do livro de M. Lapan.
################

import gym
import numpy as np

import sys
from os import path
sys.path.append( path.dirname( path.dirname( path.abspath(__file__) ) ) )

import cap08.models_torch_pg as models


# Algoritmo actor-critic básico
def run_basic_actor_critic(env, max_steps, gamma, initial_policy=None, initial_v_model=None, verbose=True):
    obs_size = env.observation_space.shape[0]
    n_actions = env.action_space.n

    if initial_policy is None:
        policy_model = models.PolicyModelPG(obs_size, [256], n_actions, lr=0.0001)
    else:
        policy_model = initial_policy.clone()

    if initial_v_model is None:
        Vmodel = models.ValueModel(obs_size, [128], lr=0.008)
    else:
        Vmodel = initial_v_model.clone()

    all_returns = []
    episodes = 0
    steps = 0

    next_state = env.reset()
    ep_return = 0.0

    while steps < max_steps:
        state = next_state

        # 1. Faz 1 passo
        action = policy_model.sample_action(state)
        next_state, r, done, _ = env.step(action)
        ep_return += r
        steps += 1

        # 2. Treina a política
        G_estimate = r + gamma * Vmodel.predict(next_state)
        advantage = G_estimate - Vmodel.predict(state)
        policy_model.partial_fit([state], [action], [advantage])

        # 3. Treina o modelo de V(.),
        Vmodel.partial_fit([state], [G_estimate])
        
        if done:
            all_returns.append((steps-1, ep_return))
            episodes += 1 
            print("step %d / ep %d: return=%.2f" % (steps, episodes, ep_return))
            
            next_state = env.reset()
            ep_return = 0.0
    
    if not done:
        all_returns.append((steps-1, ep_return))

    if verbose:
        print("step %d / ep %d: return=%.2f - end of training!" % (steps, episodes, ep_return))
    
    return all_returns, policy_model


if __name__ == "__main__":
    from cap08.models_torch_pg import test_policy
    from util.plot import plot_result

    ENV_NAME, rmax = "CartPole-v1", 500
    #ENV_NAME, rmax = "Acrobot-v1", 0

    # ATENÇÃO para a mudança: agora, o critério de parada é pela quantidade de passos
    # e não pela quantidade de episódios (estamos seguindo o padrão usado hoje em dia)
    NUM_STEPS = 10000
    GAMMA    = 0.99
    
    env = gym.make(ENV_NAME)
    inputs = env.observation_space.shape[0]
    outputs = env.action_space.n

    policy_model = models.PolicyModelPG(inputs, [256,256], outputs, lr=1e-5)
    v_model = models.ValueModel(inputs, [256,32], lr=2e-4)

    returns, policy = run_basic_actor_critic(env, NUM_STEPS, GAMMA, initial_policy=policy_model, initial_v_model=v_model)

    # Exibe um gráfico episódios x retornos (não descontados)
    plot_result(returns, rmax, window=50, return_type='steps')

    # Executa alguns episódios de forma NÃO-determinística e imprime um sumário
    test_policy(env, policy, False, 5, render=True)
