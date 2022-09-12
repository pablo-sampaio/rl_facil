
################
# Algoritmo simples da família ACTOR-CRITIC, acrescido da técnica "n-step". 
# Também usamos uma política com fator de exploração na loss function, similar ao A2C.
# Baseado no código anterior e na implementação do "n-step SARSA".
################

from collections import deque
import gym
import numpy as np

from models_torch_pg import PolicyModelPG, PolicyModelPGWithExploration, ValueModel, test_policy
from util_plot import plot_result


# Algoritmo actor-critic com parâmetro nsteps
def run_actor_critic_nstep(env, max_steps, gamma, nstep=2, initial_policy=None, initial_vmodel=None, target_return=None, verbose=True):
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
        next_state, r, done, info = env.step(action)
        ep_return += r
        steps += 1

        # 2. Adiciona no histórico
        hist_s.append(state)
        hist_a.append(action)
        hist_r.append(r)

        # se o histórico estiver completo, faz uma atualização nos modelos
        if len(hist_s) == nstep:
            G_est = sum(gamma_array*hist_r) + gamma_power_nstep*Vmodel.predict(next_state)

            # 3. Treina a política
            advantage = G_est - Vmodel.predict(hist_s[0])
            policy_model.partial_fit([hist_s[0]], [hist_a[0]], [advantage])

            # 4. Treina o modelo de V(.),
            Vmodel.partial_fit([hist_s[0]], [G_est])
        
        if done:
            all_returns.append((steps-1, ep_return))
            episodes += 1 
            reward_m = np.mean( [ ret for (st,ret) in all_returns[-50:] ] )

            if target_return is not None and reward_m >= target_return:
                if verbose:
                    print("-> target reached!")
                break

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
                G_est = ( sum(gamma_array[0:j]*hist_r) + 0 )
                advantage = G_est - Vmodel.predict(state)
                policy_model.partial_fit([hist_s[0]], [hist_a[0]], [advantage])
                Vmodel.partial_fit([hist_s[0]], [G_est])
                hist_s.popleft()
                hist_a.popleft()
                hist_r.popleft()

            if verbose:
                print("step %d / ep %d: reward=%.2f, reward_mean=%.2f" % (steps, episodes, ep_return, reward_m))

            next_state = env.reset()
            ep_return = 0.0
    
    if not done:
        all_returns.append((max_steps-1, ep_return))
    
    if verbose:
        print("step %d / ep %d: return_mean=%.2f, end of training!" % (steps, episodes, reward_m))
    
    return all_returns, policy_model


if __name__ == "__main__":
    ENV_NAME, rmax = "CartPole-v1", 300
    #ENV_NAME, rmax = "Acrobot-v1", 0
    ENV = gym.make(ENV_NAME)

    # ATENÇÃO para a mudança: agora, o critério de parada é pela quantidade de passos
    # e não pela quantidade de episódios (agora estamos seguindo o padrão da área)
    NUM_STEPS = 10000
    GAMMA     = 0.99
    NSTEP     = 32
    EXPLORATION_FACTOR = 0.05  # no CartPole, funciona bem com 0.0
    
    inputs = ENV.observation_space.shape[0]
    outputs = ENV.action_space.n

    #policy_model = PolicyModelPGWithExploration(inputs, [256, 256], outputs, exploration_factor=EXPLORATION_FACTOR, lr=3e-5)
    policy_model = PolicyModelPG(inputs, [256, 256], outputs, lr=5e-5)
    Vmodel = ValueModel(inputs, [256,32], lr=1e-4)

    returns, policy = run_actor_critic_nstep(ENV, NUM_STEPS, GAMMA, nstep=NSTEP, initial_policy=policy_model, initial_vmodel=Vmodel, target_return=rmax-100)
    #print(returns)
    
    # Exibe um gráfico passos x retornos (não descontados)
    plot_result(returns, rmax, window=1, return_type='steps')

    # Executa alguns episódios de forma determinística e imprime um sumário
    test_policy(ENV, policy, True, 5, render=True)
