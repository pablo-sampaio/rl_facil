
################
# Algoritmo simples da família ACTOR-CRITIC, acrescido da técnica "n-step". 
# Também usamos uma política com fator de exploração na loss function, similar ao A2C.
# Baseado no código anterior e na implementação do "n-step SARSA".
################

from collections import deque
import numpy as np

import sys
from os import path
sys.path.append( path.dirname( path.dirname( path.abspath(__file__) ) ) )

import cap09.models_torch_pg as models


# Algoritmo actor-critic com parâmetro nsteps
# A opção "exclusive_steps" indica se o histórico deve ser limpo a cada atualização, assim,
# cada atualização é feita com um grupo de "nsteps" passos distintos (sem sobreposição).
def run_vanilla_actor_critic_nstep(env, max_steps, gamma=0.95, nsteps=2, exclusive_steps=False,
                                   initial_policy=None, initial_v_model=None, p_lr=1e-4, relative_v_lr=5.0,
                                   verbose=True):
    obs_size = env.observation_space.shape[0]
    n_actions = env.action_space.n

    if initial_policy is None:
        policy_model = models.PolicyModelPG(obs_size, [128, 256], n_actions, lr=p_lr)
    else:
        policy_model = initial_policy.clone()

    if initial_v_model is None:
        v_lr = relative_v_lr * policy_model.lr
        Vmodel = models.ValueModel(obs_size, [128, 256], lr=v_lr)
    else:
        Vmodel = initial_v_model.clone()

    gamma_array = np.array([ gamma**i for i in range(0,nsteps)])
    gamma_power_nstep = gamma**nsteps

    all_returns = []
    episodes = 0
    steps = 0

    next_state, _ = env.reset()
    ep_return = 0.0
    
    # históricos de: estados, ações e recompensas
    hist_s = deque(maxlen=nsteps)
    hist_a = deque(maxlen=nsteps)
    hist_r = deque(maxlen=nsteps)

    while steps < max_steps:
        state = next_state

        # 1. Escolhe a ação (de forma não-determinística)
        action = policy_model.sample_action(state)

        # 2. Faz 1 passo
        next_state, r, terminated, trunc, _ = env.step(action)
        done = terminated or trunc
        ep_return += r
        steps += 1

        # 3. Adiciona no histórico
        hist_s.append(state)
        hist_a.append(action)
        hist_r.append(r)

        # 4. Se o histórico estiver completo, faz uma atualização nos modelos
        if len(hist_s) == nsteps:
            if terminated:
                V_next_state = 0
            else:
                V_next_state = Vmodel.predict(next_state)
            
            G_estimate = sum(gamma_array*hist_r) + gamma_power_nstep * V_next_state

            # 4.1 - Treina a política
            advantage = G_estimate - Vmodel.predict(hist_s[0])
            policy_model.update_weights([hist_s[0]], [hist_a[0]], [advantage])

            # 4.2 - Treina o modelo de V(.),
            Vmodel.update_weights([hist_s[0]], [G_estimate])

            if exclusive_steps:
                hist_s.clear()
                hist_a.clear()
                hist_r.clear()
        
        if done:
            all_returns.append((steps, ep_return))
            episodes += 1 

            # 5. Ao fim do episódio, atualiza o modelo para os estados que restaram no histórico
            
            if len(hist_s) == nsteps:
                # neste caso, o episódio terminou logo após uma atualização nos pesos, sem "exclusive steps"
                # logo, vai ser feita uma atualização desconsiderando o estado mais antigo, usando nsteps-1  
                # obs.: este caso nunca vai acontecer com "exclusive steps", que limpa os históricos a cada atualização
                hist_s.popleft()
                hist_a.popleft()
                hist_r.popleft()
                laststeps = nsteps - 1
            else:
                # pode acontecer com "exclusive steps", antes de encher o histórico com exatos "nsteps" passos (pela x-ésima vez qualquer)
                # ou sem "exclusive steps", se o episódio terminar antes de preencher o histórico (pela 1a vez)
                laststeps = len(hist_s)
            
            if terminated:
                V_next_state = 0
            else:
                V_next_state = Vmodel.predict(next_state)

            for j in range(laststeps,0,-1):
                G_estimate = ( sum(gamma_array[0:j]*hist_r) + V_next_state )
                advantage = G_estimate - Vmodel.predict(hist_s[0])
                policy_model.update_weights([hist_s[0]], [hist_a[0]], [advantage])
                Vmodel.update_weights([hist_s[0]], [G_estimate])
                hist_s.popleft()
                hist_a.popleft()
                hist_r.popleft()

            if verbose:
                print("step %d / ep %d: return=%.2f" % (steps, episodes, ep_return))

            next_state, _ = env.reset()
            ep_return = 0.0
    
    if not done:
        all_returns.append((steps, ep_return))

    if verbose:
        print("step %d / ep %d: return=%.2f - end of training!" % (steps, episodes, ep_return))
    
    return all_returns, policy_model


if __name__ == "__main__":
    import gymnasium as gym
    from cap09.models_torch_pg import test_policy
    from util.plot import plot_result, plot_single_result

    #ENV_NAME, rmax = "CartPole-v1", 500
    #ENV_NAME, rmax = "Acrobot-v1", 0       # demora a dar resultados
    ENV_NAME, rmax = "LunarLander-v2", 200  # demora a dar resultados (mais de 100k passos)

    # ATENÇÃO para a mudança: agora, o critério de parada é pela quantidade de passos
    # e não pela quantidade de episódios (agora estamos seguindo o padrão da área)
    NUM_STEPS = 100_000
    NSTEP     = 8
    POLICY_LR = 15e-5
    #EXPLORATION_FACTOR = 0.01  # no CartPole, funciona bem com 0.0
    
    env = gym.make(ENV_NAME)
    inputs = env.observation_space.shape[0]
    outputs = env.action_space.n

    #policy_model = models.PolicyModelPGWithExploration(inputs, [256, 256], outputs, exploration_factor=EXPLORATION_FACTOR, lr=POLICY_LR)   
    policy_model = models.PolicyModelPG(inputs, [256, 256], outputs, lr=POLICY_LR)
    v_model = models.ValueModel(inputs, [256, 256], lr=5*POLICY_LR)

    returns, policy = run_vanilla_actor_critic_nstep(env, NUM_STEPS, nsteps=NSTEP, exclusive_steps=False, 
                                                     initial_policy=policy_model, initial_v_model=v_model)
 
    # Exibe um gráficos passos x retornos (não descontados)
    plot_result(returns, rmax, x_axis='step')
    plot_result(returns, rmax, x_axis='step', cumulative='avg')

    # Executa alguns episódios de forma NÃO-determinística e imprime um sumário
    eval_env = gym.make(ENV_NAME, render_mode="human")
    test_policy(eval_env, policy, False, 5)
    eval_env.close()
