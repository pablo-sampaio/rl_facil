################
# Algoritmo "REINFORCE", da familia policy-gradient, acrescido da técnica de "Advantage"
# Referências: curso Udemy (e códigos) de "Lazy Programmer" e livro de Maxim Lapan.
################

import gym

import sys
from os import path
sys.path.append( path.dirname( path.dirname( path.abspath(__file__) ) ) )

import cap08.models_torch_pg as models


# Algoritmo REINFORCE usando "advantage" como ténica de baseline para reduzir a variância
def run_reinforce_with_adv(env, total_episodes, gamma, initial_policy=None, initial_v_model=None, render=False):
    obs_size = env.observation_space.shape[0]
    n_actions = env.action_space.n

    if initial_policy is None:
        policy_model = models.PolicyModelPG(obs_size, [256], n_actions, lr=0.001)
    else:
        policy_model = initial_policy.clone()

    if initial_v_model is None:
        Vmodel = models.ValueModel(obs_size, [128], lr=0.005)
    else:
        Vmodel = initial_v_model.clone()

    all_returns = []
    total_steps = 0

    # loop principal
    for i in range(total_episodes):
        done = False
        ep_return = 0
        reward = 0
        ep_trajectory = []
        
        state = env.reset()
    
        # PARTE 1: Executa um episódio completo
        while not done:
            # exibe/renderiza os passos no ambiente, a cada 500 episódios
            if render and (i+1) % 500 == 0:
                env.render()

            # escolhe a próxima ação
            action = policy_model.sample_action(state)
        
            # realiza a ação, ou seja, dá um passo no ambiente
            next_state, reward, done, _ = env.step(action)
            
            # adiciona a tripla que representa este passo
            ep_trajectory.append( (state, action, reward) )
            
            ep_return += reward
            state = next_state
        
        all_returns.append(ep_return)

        # PARTE 2: Calcula listas separadas de estados, ações e retornos parciais
        states = []
        actions = []
        partial_returns = []
        advantages = []
        
        Gt = 0
        for (s, a, r) in reversed(ep_trajectory):
            Gt = r + gamma*Gt
            states.append(s)
            actions.append(a)
            partial_returns.append(Gt)
            advantages.append(Gt - Vmodel.predict(s))

        # PARTE 3: Atualiza a política usando os trios (s, a, At), 
        #          onde  's' é entrada da rede, 'a' é o índice da saída, e o 'At' é o "advantage" usado no cálculo da loss function
        loss_p = policy_model.partial_fit(states, actions, advantages)
        
        # PARTE 4: Atualiza o modelo de V(.), usando o par (s, Gt), onde  's' é entrada da rede, e 'Gt' é o retorno parcial cuja esperança deve ser dada como saída da rede
        loss_v = Vmodel.partial_fit(states, partial_returns)

        if (i+1) % 200 == 0:
            print("- episode %d (step %d): losses[v|p]=%.4f|%.4f, ep_return=%.2f" % (i+1, total_steps, loss_p, loss_v, ep_return))
 
    return all_returns, policy_model


if __name__ == "__main__":
    from cap08.models_torch_pg import test_policy
    from util.plot import plot_result

    ENV_NAME, rmax = "CartPole-v1", 500
    #ENV_NAME, rmax = "Acrobot-v1", 0
    #ENV_NAME, rmax = "LunarLander-v2", 150
    #ENV_NAME, rmax = "MountainCar-v0", -20

    EPISODES = 1000
    GAMMA    = 0.95

    env = gym.make(ENV_NAME)
    inputs = env.observation_space.shape[0]
    outputs = env.action_space.n
    policy = models.PolicyModelPG(inputs, [128, 512], outputs, lr=0.001)

    returns, policy = run_reinforce_with_adv(env, EPISODES, GAMMA, initial_policy=policy)

    # Exibe um gráfico episódios x retornos (não descontados)
    plot_result(returns, rmax, window=50)

    # Executa alguns episódios de forma NÃO-determinística e imprime um sumário
    test_policy(env, policy, False, 5, render=True)
