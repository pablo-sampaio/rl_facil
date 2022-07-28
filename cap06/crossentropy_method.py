import time
import gym
from collections import namedtuple
import numpy as np

from models_torch import PolicyModelCrossentropy

################
#
# Adaptação do exemplo 1 do Cap. 4 do livro de M. Lapan.
# 
# Para monitorar com Tensorboard, rodar a partir do diretorio deste projeto:
# cd Dropbox/Estudos/Reinforcement\ Learning/Grupo\ Estudos/exemplos
# tensorboard --logdir runs --host localhost
#
################


Episode = namedtuple('Episode', field_names=['sum_rewards', 'steps'])
EpisodeStep = namedtuple('EpisodeStep', field_names=['state', 'action'])


def run_episodes(env, policy_net, batch_size):
    batch = []
    for i in range(0,batch_size):
        sum_rewards = 0.0
        epi_steps = []
        obs = env.reset()
        is_done = False
        while not is_done:
            action = policy_net.sample_action(obs)
            next_obs, reward, is_done, _ = env.step(action)
            sum_rewards += reward
            epi_steps.append(EpisodeStep(state=obs, action=action))
            obs = next_obs
        batch.append(Episode(sum_rewards=sum_rewards, steps=epi_steps))
    return batch


def filter_batch(batch, percent_of_best):
    ep_returns = list(map(lambda epi: epi.sum_rewards, batch))
    return_limit = np.percentile(ep_returns, 100-percent_of_best)
    return_mean = float(np.mean(ep_returns))

    train_states = []
    train_actions = []
    for example in batch:
        if example.reward >= return_limit:
            train_states.extend(map(lambda step: step.state, example.steps)) # extrai apenas o estado da lista de passos e insere na lista de treinamento
            train_actions.extend(map(lambda step: step.action, example.steps))   # extrai apenas o acao da lista de passos e insere na lista de treinamento

    return train_states, train_actions, return_limit, return_mean


if __name__ == "__main__":
    ENV_NAME = "CartPole-v1"
    env = gym.make(ENV_NAME)
    #env = gym.wrappers.Monitor(env, directory="mon", force=True)
    HIDDEN_NODES = 64   # mais rapido com 128 / tambem funciona com 16, mas demora mais
    EPISODES_TO_RUN = 16
    PERCENT_BEST = 30   # percentage of the best episodes to be selected

    obs_size = env.observation_space.shape[0]
    n_actions = env.action_space.n

    policy_model = PolicyModelCrossentropy(obs_size, [HIDDEN_NODES], n_actions, lr=0.01)

    for iter_no in range(1,1001):
        batch = run_episodes(env, policy_model, EPISODES_TO_RUN)
        states, actions, reward_bound, reward_mean = filter_batch(batch, PERCENT_BEST)

        #writer.add_scalar("reward_mean", reward_mean, iter_no)
        #writer.add_scalar("reward_bound", reward_bound, iter_no)

        if reward_mean > 350.0:
            print("%d: reward_mean=%.2f, reward_bound=%.2f" % (iter_no, reward_mean, reward_bound))
            print("Solved!")
            break

        # treina para reforcar o mapeamento estado-acao das politicas selecionadas
        p_loss = policy_model.partial_train(states, actions)
 
        writer.add_scalar("policy_loss", p_loss, iter_no)
        print("%d: loss=%.3f, reward_mean=%.2f, reward_bound=%.2f" % (iter_no, p_loss, reward_mean, reward_bound))

    writer.close()

    # play a final round to show
    print("Final policy demonstration...")
    obs = env.reset()
    done = False
    reward = 0.0
    steps = 0
    while not done:
        env.render()
        action = policy_model.best_action(obs) # faz so a acao de maior probabilidade
        obs, r, done, _ = env.step(action)
        reward += r
        steps += 1
    env.render()
    print("Total steps:", steps)
    print("Final reward:", reward)

