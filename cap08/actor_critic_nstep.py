
################
# Algoritmo simples da família ACTOR-CRITIC, acrescido da técnica "n-step". 
# Também usamos uma política com fator de exploração na loss function, similar ao A2C.
# Baseado no código anterior e na implementação do "n-step SARSA".
################

from collections import deque
import gym
import numpy as np

import sys
from os import path
sys.path.append( path.dirname( path.dirname( path.abspath(__file__) ) ) )

import cap08.models_torch_pg as models


# Algoritmo actor-critic com parâmetro nsteps
def run_vanilla_actor_critic_nstep(env, max_steps, gamma, nstep=2, initial_policy=None, initial_v_model=None, verbose=True):
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

    gamma_array = np.array([ gamma**i for i in range(0,nstep)])
    gamma_power_nstep = gamma**nstep

    all_returns = []
    episodes = 0
    steps = 0

    next_state = env.reset()
    ep_return = 0.0
    
    # históricos de: estados, ações e recompensas
    hist_s = deque(maxlen=nstep)
    hist_a = deque(maxlen=nstep)
    hist_r = deque(maxlen=nstep)

    while steps < max_steps:
        state = next_state

        # 1. Faz 1 passo
        action = policy_model.sample_action(state)
        next_state, r, done, _ = env.step(action)
        ep_return += r
        steps += 1

        # 2. Adiciona no histórico
        hist_s.append(state)
        hist_a.append(action)
        hist_r.append(r)

        # se o histórico estiver completo, faz uma atualização nos modelos
        if len(hist_s) == nstep:
            G_estimate = sum(gamma_array*hist_r) + gamma_power_nstep*Vmodel.predict(next_state)

            # 3. Treina a política
            advantage = G_estimate - Vmodel.predict(hist_s[0])
            policy_model.partial_fit([hist_s[0]], [hist_a[0]], [advantage])

            # 4. Treina o modelo de V(.),
            Vmodel.partial_fit([hist_s[0]], [G_estimate])
        
        if done:
            all_returns.append((steps, ep_return))
            episodes += 1 

            # ao fim do episódio, atualiza o modelo para os estados que restaram no histórico
            # trata de forma especial o caso em que o tamanho episódio é inferior ao "nstep"
            if len(hist_s) == nstep:
                hist_s.popleft()
                hist_a.popleft()
                hist_r.popleft()
                laststeps = nstep - 1
            else:
                laststeps = len(hist_s) 
            
            for j in range(laststeps,0,-1):
                G_estimate = ( sum(gamma_array[0:j]*hist_r) + 0 )
                advantage = G_estimate - Vmodel.predict(state)
                policy_model.partial_fit([hist_s[0]], [hist_a[0]], [advantage])
                Vmodel.partial_fit([hist_s[0]], [G_estimate])
                hist_s.popleft()
                hist_a.popleft()
                hist_r.popleft()

            if verbose:
                print("step %d / ep %d: return=%.2f" % (steps, episodes, ep_return))

            next_state = env.reset()
            ep_return = 0.0
    
    if not done:
        all_returns.append((steps, ep_return))

    if verbose:
        print("step %d / ep %d: return=%.2f - end of training!" % (steps, episodes, ep_return))
    
    return all_returns, policy_model


if __name__ == "__main__":
    from cap08.models_torch_pg import test_policy
    from util.plot import plot_result

    ENV_NAME, rmax = "CartPole-v1", 500
    #ENV_NAME, rmax = "Acrobot-v1", 0

    # ATENÇÃO para a mudança: agora, o critério de parada é pela quantidade de passos
    # e não pela quantidade de episódios (agora estamos seguindo o padrão da área)
    NUM_STEPS = 20000
    GAMMA     = 0.99
    NSTEP     = 32
    EXPLORATION_FACTOR = 0.05  # no CartPole, funciona bem com 0.0
    
    env = gym.make(ENV_NAME)
    inputs = env.observation_space.shape[0]
    outputs = env.action_space.n

    #policy_model = models.PolicyModelPGWithExploration(inputs, [256, 256], outputs, exploration_factor=EXPLORATION_FACTOR, lr=3e-5)
    policy_model = models.PolicyModelPG(inputs, [256, 256], outputs, lr=4e-5) #5e-5
    v_model = models.ValueModel(inputs, [256,32], lr=8e-5) #1e-4

    returns, policy = run_vanilla_actor_critic_nstep(env, NUM_STEPS, GAMMA, nstep=NSTEP, initial_policy=policy_model, initial_v_model=v_model)
    
    # Exibe um gráfico passos x retornos (não descontados)
    plot_result(returns, rmax, window=1, x_axis='steps')

    # Executa alguns episódios de forma determinística e imprime um sumário
    test_policy(env, policy, True, 5, render=True)
