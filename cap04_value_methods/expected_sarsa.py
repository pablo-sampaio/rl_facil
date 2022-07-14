import gym
import numpy as np

from util_plot import save_rewards_plot
from util_experiments import test_greedy_Q_policy


def softmax(x):
    x = x - np.max(x)
    x = np.exp(x)
    x = x / np.sum(x)
    return x

# choose action from a Q-table, with a softmax strategy
# attention: state must be converted (discretized) to "single bin"
def softmax_policy(Q, state_num):
    probs = softmax(Q[state_num])
    return np.random.choice(len(probs), p=probs)

# choose action from a Q-table, with a variant of epsilon-greedy strategy
# attention: state must be converted (discretized) to "single bin"
def epsilon_greedy_probs(Q, state_num, num_actions, epsilon):
    q_max = np.max(Q[state_num, :])
    greedy_actions = 0
    for i in range(num_actions):
        if Q[state_num][i] == q_max:
            greedy_actions += 1
    
    non_greedy_action_probability = epsilon / num_actions
    greedy_action_probability = ((1 - epsilon) / greedy_actions) + non_greedy_action_probability

    probs = []
    for i in range(num_actions):
        if Q[state_num][i] == q_max:
            probs.append(greedy_action_probability)
        else:
            probs.append(non_greedy_action_probability)
    return probs

def epsilon_greedy_policy(Q, state_num, epsilon=0.1):
    num_actions = len(Q[state_num])
    probs = epsilon_greedy_probs(Q, state_num, num_actions, epsilon)
    return np.random.choice(num_actions, p=probs)


# Expected-SARSA algorithm
# Atenção: os espaços de estados e de ações precisam ser discretos, dados por valores inteiros
def run_expected_sarsa(env, episodes, lr=0.1, gamma=0.95, epsilon=0.1, render=False):
    assert isinstance(env.observation_space, gym.spaces.Discrete)
    assert isinstance(env.action_space, gym.spaces.Discrete)

    num_actions = env.action_space.n
    
    # inicializa a tabela Q com valores aleatórios de -1.0 a 0.0
    # usar o estado como índice das linhas e a ação como índice das colunas
    Q = np.random.uniform(low = -1.0, high = 0.0, 
                          size = (env.observation_space.n, num_actions))

    # para cada episódio, guarda sua soma de recompensas (retorno não-discontado)
    sum_rewards_per_ep = []
     
    # loop principal
    for i in range(episodes):
        done = False
        sum_rewards, reward = 0, 0
        
        state = env.reset()  # resets and gets the initial state
    
        # This loop runs an entire episode
        while done != True:
            # Render environment for the final 5 episodes, and on each 1000 episodes
            if render and (i >= (episodes - 5) or (i+1) % 1000 == 0):
                env.render()
            
            # Determine next action
            action = epsilon_greedy_policy(Q, state, epsilon)
            #action = softmax_policy(Q, state)  # bad results!

            # Do a step, then get next state and reward
            next_state, reward, done, _ = env.step(action)

            if done: 
                # para estados terminais
                V_next_state = 0 
            else:
                # para estados não-terminais -- valor esperado
                p_next_actions = epsilon_greedy_probs(Q, next_state, num_actions, epsilon)
                #p_next_actions = softmax(Q[next_state_num])
                V_next_state = np.sum( p_next_actions * Q[next_state] ) 

            # atualiza a Q-table
            # delta = estimativa usando a nova recompensa - valor antigo
            delta = (reward + gamma * V_next_state) - Q[state,action]
            Q[state,action] = Q[state,action] + lr * delta
            
            sum_rewards += reward
            state = next_state

        # salva o retorno do episódio que encerrou
        sum_rewards_per_ep.append(sum_rewards)
        
        # a cada 100 episódios, imprime informação sobre o progresso 
        if (i+1) % 100 == 0:
            avg_reward = np.mean(sum_rewards_per_ep[-100:])
            print(f"Episode {i+1} Average Reward (last 100): {avg_reward:.3f}")

    return sum_rewards_per_ep, Q



if __name__ == "__main__":
    ENV_NAME = "Taxi-v3"
    r_max_plot = 10

    EPISODES = 30000
    LR = 0.01
    GAMMA = 0.95
    EPSILON = 0.1

    env = gym.make(ENV_NAME)
    
    # Roda o algoritmo Expected-SARSA
    rewards, Qtable = run_expected_sarsa(env, EPISODES, LR, GAMMA, EPSILON, render=False)
    print("Últimos resultados: media =", np.mean(rewards[-20:]), ", desvio padrao =", np.std(rewards[-20:]))

    # Salva um arquivo com o gráfico de episódios x retornos (não descontados)
    filename = f"results/expected_sarsa-{ENV_NAME.lower()[0:8]}-ep{EPISODES}-lr{LR}.png"
    save_rewards_plot(rewards, r_max_plot, filename)

    test_greedy_Q_policy(env, Qtable, 10, True)
    env.close()
