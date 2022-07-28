################
#
# Adaptação do exemplo 1 do Cap. 4 do livro de M. Lapan.
# 
################


import gym
from collections import namedtuple
import numpy as np

from models_torch import PolicyModelCrossentropy

EpisodeStep = namedtuple('EpisodeStep', field_names=['state', 'action'])


def run_episodes(env, policy_net, batch_size):
    all_trajectories = []
    all_returns = []
    for i in range(0,batch_size):
        sum_rewards = 0.0
        ep_trajectory = []
        obs = env.reset()
        is_done = False
        while not is_done:
            # faz uma amostragem da ação, ou seja, 
            # gera uma ação de acordo com as probabilidades retornadas pela rede
            action = policy_net.sample_action(obs)
            next_obs, reward, is_done, _ = env.step(action)
            sum_rewards += reward
            ep_trajectory.append(EpisodeStep(state=obs, action=action))
            obs = next_obs
        all_trajectories.append(ep_trajectory)
        all_returns.append(sum_rewards)
    return all_trajectories, all_returns


def filter_batch(ep_trajectories, ep_returns, percent_of_best):
    return_limit = np.percentile(ep_returns, 100-percent_of_best)
    return_mean = float(np.mean(ep_returns))

    train_states = []
    train_actions = []
    for i in range(len(ep_trajectories)):
        if ep_returns[i] >= return_limit:
            train_states.extend(map(lambda step: step.state, ep_trajectories[i])) # extrai apenas o estado da lista de passos e insere na lista de treinamento
            train_actions.extend(map(lambda step: step.action, ep_trajectories[i]))   # extrai apenas o acao da lista de passos e insere na lista de treinamento

    return train_states, train_actions, return_limit, return_mean


def train_crossentropy(env, total_episodes, ep_batch_size=10, ep_selected_proportion=0.3, policy_hidden_layers=[64]):
    obs_size = env.observation_space.shape[0]
    n_actions = env.action_space.n

    policy_model = PolicyModelCrossentropy(obs_size, policy_hidden_layers, n_actions, lr=0.01)
    
    ep_returns = [] # all episodes' returns

    for iter_no in range(1,total_episodes+1, ep_batch_size):
        trajectories, returns = run_episodes(env, policy_model, ep_batch_size)
        ep_returns.extend(returns)

        states, actions, reward_bound, reward_mean = filter_batch(trajectories, returns, ep_selected_proportion)

        if reward_mean > 350.0:
            print("%d: reward_mean=%.2f, reward_bound=%.2f" % (iter_no, reward_mean, reward_bound))
            print("Solved!")
            break

        # treina para reforcar o mapeamento estado-acao das politicas selecionadas
        p_loss = policy_model.partial_train(states, actions)
 
        print("%d: loss=%.3f, reward_mean=%.2f, reward_bound=%.2f" % (iter_no, p_loss, reward_mean, reward_bound))
    
    return ep_returns, policy_model

if __name__ == "__main__":
    ENV_NAME = "CartPole-v1"
    env = gym.make(ENV_NAME)

    HIDDEN_LAYERS = [64]   # mais rapido com 128 / tambem funciona com 16, mas demora mais
    EPISODES_BATCH_SIZE = 16
    PERCENT_BEST = 0.3     # percentage of the best episodes to be selected

    returns, policy = train_crossentropy(env, 1000, EPISODES_BATCH_SIZE, PERCENT_BEST, HIDDEN_LAYERS)

    # play a final round to show
    print("Final policy demonstration...")
    obs = env.reset()
    done = False
    reward = 0.0
    steps = 0
    while not done:
        env.render()
        action = policy.best_action(obs) # faz so a acao de maior probabilidade
        obs, r, done, _ = env.step(action)
        reward += r
        steps += 1
    env.render()
    print("Total steps:", steps)
    print("Final reward:", reward)

