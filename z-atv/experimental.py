import gymnasium as gym
import numpy as np
import abc
import torch

# This is the abstract class for all policies 
class Policy(abc.ABC):
    @abc.abstractmethod
    def get_action(self, state):
        pass


# class that represents an epsilon-greedy from Q-Tables
class EpsilonGreedyPolicy(Policy):
    def __init__(self, q_table, epsilon):
        self.q_table = q_table
        self.epsilon = epsilon

    def get_action(self, state):
        if np.random.random() < self.epsilon:
            return np.random.randint(len(self.q_table[state]))
        else:
            return np.argmax(self.q_table[state])


class GreedyPolicy(Policy):
    def __init__(self, q_table, epsilon):
        self.q_table = q_table
        self.epsilon = epsilon

    def get_action(self, state):
        return np.argmax(self.q_table[state])


def epsilon_greedy(qtable, state, epsilon):
    '''Function that receives a Q-table and chooses an action using epsilon-greedy strategy'''
    if np.random.random() < epsilon:
        return np.random.randint(len(qtable[state]))
    else:
        return np.argmax(qtable[state])


def sarsa_trainning(env, alpha, gamma, epsilon, max_steps=100):
    '''Function that trains a Q-table using SARSA algorithm'''
    num_states = env.observation_space.n
    num_actions = env.action_space.n

    q_table = np.zeros((num_states, num_actions))

    state, _ = env.reset()
    action = epsilon_greedy(q_table, state, epsilon)
    steps = 0

    while steps < max_steps:        
        next_state, reward, truncated, terminated, _ = env.step(action)
        done = terminated or truncated
        next_action = epsilon_greedy(q_table, next_state, epsilon)

        if terminated:
            q_target = reward
        else:
            q_target = reward + gamma * q_table[next_state, next_action]
        
        q_table[state, action] = q_table[state, action] + alpha * (q_target - q_table[state, action])
        
        if done:
            state, _ = env.reset()
            action = epsilon_greedy(q_table, state, epsilon)
        else:
            state = next_state
        
        action = next_action
        steps += 1

    return EpsilonGreedyPolicy(q_table, epsilon)
    

def q_learning(env, alpha, gamma, epsilon, max_steps):
    q_table = np.zeros([env.observation_space.n, env.action_space.n])
    state = env.reset()
    steps = 0

    while steps < max_steps:
        action = epsilon_greedy(q_table, state, epsilon)
        next_state, reward, truncated, terminated, _ = env.step(action)

        if terminated:
            q_target = reward
        else:
            q_target = reward + gamma * np.max(q_table[next_state])
        
        q_table[state, action] = q_table[state, action] + alpha * (q_target - q_table[state, action])
        
        state = next_state
        steps += 1

    return EpsilonGreedyPolicy(q_table, epsilon)


class GreedyQNetPolicy(Policy):
    def __init__(self, qnet):
        self.qnet = qnet

    def get_action(self, state_tensor):
        with torch.no_grad():
            q_values = self.qnet(state_tensor)
            q_values = q_values.squeeze()
            action = torch.argmax(q_values)
        return action.item()


class EpsilonGreedyQNetPolicy(Policy):
    def __init__(self, qnet, epsilon):
        self.qnet = qnet
        self.epsilon = epsilon

    def get_action(self, state_tensor):
        with torch.no_grad():
            q_values = self.qnet(state_tensor)
            q_values = q_values.squeeze()
            if np.random.rand() < self.epsilon:
                action = np.random.randint(len(q_values))
            else:
                action = torch.argmax(q_values)
        return action.item()


def run_semigradient_qlearning(env, num_episodes=1000, learning_rate=0.001, discount_factor=0.99, epsilon=0.1):
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    qnet = TorchMultiLayerNetwork(state_dim, [32, 128], action_dim)
    optimizer = optim.Adam(qnet.parameters(), lr=learning_rate)

    for episode in range(num_episodes):
        state, _ = env.reset()
        state_tensor = torch.tensor(state, dtype=torch.float32) # turns the next_state into a Pytorch tensor
        state_tensor = state_tensor.unsqueeze(0)                # adds a dimension of size 1 in axis 0 (e.g. from shape (4,) to shape (1,4))

        total_reward = 0
        done = False

        while not done:
            
            # epsilon-greedy action selection
            if np.random.rand() < epsilon:
                action = env.action_space.sample()
            else:
                action = policy.get_action(state_tensor)

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            next_state_tensor = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)

            # Calculate target Q-value
            if terminated:
                target_q = reward
            else:
                V_next_state = max_qvalue(qnet, next_state_tensor)
                target_q = reward + discount_factor * V_next_state

            # Calculate current Q-value and update the network
            current_q = qnet(state_tensor)[0, action]
            
            #loss = nn.MSELoss()(current_q, target_q)
            loss = (current_q - target_q) ** 2

            optimizer.zero_grad() # Resets gradients from previous iteration
            loss.backward()       # Backward pass, to calculte the gradients of the loss function with respect to the weights
            optimizer.step()      # Updates the weights

            total_reward += reward
            state_tensor = next_state_tensor

        # Print the total reward obtained in this episode
        print(f"Episode {episode + 1}/{num_episodes}, Total Reward: {total_reward}")

    return GreedyQNetPolicy(qnet)


def record_video_policy(env_name, policy, length=500, folder='videos/', prefix='rl-video'):
    """
    Records a video using a given policy.
    - env_name: A string of the environment registered in the gym or an instance of the class. At the end, the environment is closed (function `close()`).
    - policy: An instance of a policy class.
    - length: Number of environment steps used in the video.
    - prefix: Prefix of the video file names.
    - folder: Folder where the video files will be saved.
    """
    if isinstance(env_name, str):
        env = gym.make(env_name, render_mode="rgb_array")
    else:
        env = env_name
    rec_env = gym.wrappers.RecordVideo(env, folder, episode_trigger=lambda i : True, video_length=length, name_prefix=prefix)
    num_steps = 0
    while num_steps < length:
        state, _ = rec_env.reset()
        num_steps += 1
        done = False
        while (not done) and (num_steps < length):
            action = policy.get_action(state)
            state, r, termi, trunc, _ = rec_env.step(action)
            done = termi or trunc
            num_steps += 1
    rec_env.close()
    env.close()
