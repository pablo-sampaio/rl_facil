################
#
# Adaptação do exemplo 1 do Cap. 4 do livro de M. Lapan.
# 
################
from collections import namedtuple

import gym
import numpy as np

from models_torch import PolicyModelCrossentropy, test_policy
from util_plot import plot_result

EpisodeStep = namedtuple('EpisodeStep', field_names=['state', 'action'])


def run_episodes(env, policy_net, batch_size, render_last):
    all_trajectories = []
    all_returns = []
    render = False
    for i in range(0,batch_size):
        sum_rewards = 0.0
        trajectory = []
        obs = env.reset()
        is_done = False
        if render_last and i == batch_size-1:
            render = True
            env.render()
        while not is_done:
            # faz uma amostragem da ação, ou seja, 
            # gera uma ação de acordo com as probabilidades retornadas pela rede
            action = policy_net.sample_action(obs)
            next_obs, reward, is_done, _ = env.step(action)
            sum_rewards += reward
            trajectory.append(EpisodeStep(state=obs, action=action))
            if render:
                env.render()
            obs = next_obs
        all_trajectories.append(trajectory)
        all_returns.append(sum_rewards)
    return all_trajectories, all_returns


def run_crossentropy_method_x(env, total_episodes, ep_batch_size=10, ep_selected_proportion=0.2, policy_model=None, render=False):
    obs_size = env.observation_space.shape[0]
    n_actions = env.action_space.n

    if policy_model is None:
        policy_model = PolicyModelCrossentropy(obs_size, [128], n_actions, lr=0.005)
    
    all_returns = []

    episodes_to_select = round(ep_batch_size * ep_selected_proportion)
    elite_trajectories = [ (float("-inf"), []) for i in range(episodes_to_select)]

    episodes = 0
    while episodes < total_episodes:

        # 1. Roda alguns episódios
        episodes += ep_batch_size
        render_last = render and (episodes % 100 == 0)
        trajectories, returns = run_episodes(env, policy_model, ep_batch_size, render_last)
        all_returns.extend(returns)

        # 2. Define o retorno de corte para os melhores episódios
        return_limit = np.quantile(returns, 1.0-ep_selected_proportion)
        return_mean = float(np.mean(returns))

        # 3.1. Extrai os estados e ações dos melhores episódios, e seleciona para a "elite"
        states = []
        actions = []
        ep_selected = 0
        min_selected_return = float("inf")
        for i in range(len(trajectories)):
            if returns[i] >= return_limit:
                if returns[i] > elite_trajectories[0][0]:
                    index = np.random.randint(0, len(elite_trajectories))
                    elite_trajectories[index] = (returns[i], trajectories[i])
                    elite_trajectories.sort(key = (lambda x : x[0]))
                if ep_selected < episodes_to_select or returns[i] > min_selected_return:
                    ep_selected += 1
                    states.extend(map(lambda step: step.state, trajectories[i])) # extrai apenas o estado da lista de passos e insere na lista de treinamento
                    actions.extend(map(lambda step: step.action, trajectories[i]))   # extrai apenas o acao da lista de passos e insere na lista de treinamento
                    if returns[i] < min_selected_return:
                        min_selected_return = returns[i]

        # 3.2. Extrai os estados e ações dos episódios elite
        for i in range(len(elite_trajectories)):
            states.extend(map(lambda step: step.state, elite_trajectories[i][1]))
            actions.extend(map(lambda step: step.action, elite_trajectories[i][1]))

        # 4. Treina o modelo para reforcar o mapeamento estado-ação
        p_loss = policy_model.partial_fit(states, actions)
 
        print("- episode %d (selected %d): loss=%.3f, return_mean=%.2f, return_limit=%.2f" % (episodes, ep_selected, p_loss, return_mean, return_limit))
        #print("- elite:", list(map(lambda ep: ep[0], elite_trajectories)) )
    
    return all_returns, policy_model


if __name__ == "__main__":
    #ENV_NAME, rmax = "CartPole-v1", 500
    ENV_NAME, rmax = "Acrobot-v1", 0
    #ENV_NAME, rmax = "LunarLander-v2", 300
    #ENV_NAME, rmax = "MountainCar-v0", 0  # resultados ruins
    ENV = gym.make(ENV_NAME)

    EPISODES = 700         # total de episódios
    BATCH_SIZE = 10        # quantidade de episódios executados por época de treinamento
    PERCENT_BEST = 0.2     # percentual dos episódios (do batch) que serão selecionados

    returns, policy = run_crossentropy_method_x(ENV, EPISODES, BATCH_SIZE, PERCENT_BEST)

    print("Últimos resultados: media =", np.mean(returns[-20:]), ", desvio padrao =", np.std(returns[-20:]))

    # Exibe um gráfico episódios x retornos (não descontados)
    plot_result(returns, rmax, None)

    # Executa alguns episódios de forma NÃO-determinística e imprime um sumário
    test_policy(ENV, policy, False, 5, render=True)

    # Expandindo aqui a execução de alguns episódios de forma DETERMINÍSTICA, para fins didáticos
    for i in range(5):
        print(f"TEST EPISODE {i+1}")
        obs = ENV.reset()
        done = False
        reward = 0.0
        steps = 0
        while not done:
            ENV.render()
            action = policy.best_action(obs) # faz so a acao de maior probabilidade
            obs, r, done, _ = ENV.step(action)
            reward += r
            steps += 1
        ENV.render()
        print("- steps:", steps)
        print("- return:", reward)

    ENV.close()
