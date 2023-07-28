################
# Algoritmo "REINFORCE", da familia policy-gradient (ou Vanila Police Gradient - VPG).
# Referências: curso Udemy (e códigos) de "Lazy Programmer" e livro de Maxim Lapan.
################

import gym
from collections import namedtuple, deque
import numpy as np

EpisodeStep = namedtuple('EpisodeStep', field_names=['state', 'action', 'reward', 'next_state'])

def run_episodes(env, policy_net, batch_size=1):
    batch_trajectories = []
    batch_returns = []
    num_steps = 0
    for i in range(0,batch_size):
        sum_rewards = 0.0
        trajectory = []
        obs = env.reset()
        is_done = False
        while not is_done:
            action = policy_net.sample_action(obs)
            next_obs, reward, is_done, _ = env.step(action)
            sum_rewards += reward
            trajectory.append(EpisodeStep(state=obs, action=action, reward=reward, next_state=next_obs))
            obs = next_obs
        batch_trajectories.append(trajectory)
        batch_returns.append(sum_rewards)
        num_steps += len(trajectory)
    return batch_trajectories, batch_returns, num_steps


def extract_states_actions_returns(episodes, gamma):
    train_states, train_actions, train_state_rets = deque([]), deque([]), deque([])
    for epi in episodes:
        G = 0  # retorno (soma descontada) do último estado
        for step in reversed(epi):
            G = step.reward + gamma*G
            train_states.appendleft(step.state)  # insere à esquerda, para desfazer a inversão da lista
            train_actions.appendleft(step.action)
            train_state_rets.appendleft(G)
    return train_states, train_actions, train_state_rets


# Algoritmo REINFORCE original (também chamado de "Vanilla Policy Gradient")
def run_reinforce(env, total_episodes, gamma, initial_policy=None, target_return=None):
    obs_size = env.observation_space.shape[0]
    n_actions = env.action_space.n

    if initial_policy is None:
        policy_model = PolicyModelPG(obs_size, [256], n_actions, lr=0.001)
    else:
        policy_model = initial_policy.clone()

    all_returns = []
    total_steps = 0

    episodes = 0
    while episodes < total_episodes:
        # 1. Roda um episódio
        trajectories, returns, steps = run_episodes(env, policy_model, 1)
        all_returns.extend(returns)
        episodes += 1 
        total_steps += steps
        ep_return = float(np.mean(returns))

        if target_return is not None and np.mean(all_returns[-50:]) >= target_return:
            print("- episode %d (step %d): return_mean_50=%.2f, target reached!" % (episodes, total_steps, np.mean(all_returns[-50:])))
            break

        # 2. Retorna listas separadas com estados, ações e retornos futuros G_i (a partir do estado)
        states, actions, state_returns = extract_states_actions_returns(trajectories, gamma)

        # 3. Treina a política usando os trios (s, a, Gi), onde  's' é entrada da rede, 'a' é saída, e o 'G' é usado no cálculo da loss function
        loss_p = policy_model.partial_fit(states, actions, state_returns)
        
        print("- episode %d (step %d): loss_p=%.5f, ep_return=%.2f" % (episodes, total_steps, loss_p, ep_return))
 
    return all_returns, policy_model



if __name__ == "__main__":
    from models_torch_pg import PolicyModelPG, test_policy
    from util.plot import plot_result

    ENV_NAME, rmax = "CartPole-v1", 500
    #ENV_NAME, rmax = "Acrobot-v1", 0
    #ENV_NAME, rmax = "LunarLander-v2", 150
    #ENV_NAME, rmax = "MountainCar-v0", -20
    ENV = gym.make(ENV_NAME)

    EPISODES = 2000
    GAMMA    = 0.95
    
    inputs = ENV.observation_space.shape[0]
    outputs = ENV.action_space.n
    policy = PolicyModelPG(inputs, [128, 512], outputs, lr=0.0005)

    returns, policy = run_reinforce(ENV, EPISODES, GAMMA, initial_policy=policy, target_return=200.0)

    # Exibe um gráfico episódios x retornos (não descontados)
    plot_result(returns, rmax, window=50)

    # Executa alguns episódios de forma NÃO-determinística e imprime um sumário
    test_policy(ENV, policy, False, 5, render=True)
