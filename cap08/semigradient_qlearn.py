import numpy as np
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim

from dqn_models import MLP

import sys
from os import path
sys.path.append( path.dirname( path.dirname( path.abspath(__file__) ) ) )

from cap09.models_torch_pg import TorchMultiLayerNetwork


def greedy_action(qnet, state_tensor):
    with torch.no_grad():
        q_values = qnet(state_tensor)
        q_values = q_values.squeeze()
        action = torch.argmax(q_values)
    return action.item()

def max_qvalue(qnet, state_tensor):
    with torch.no_grad():
        q_values = qnet(state_tensor)
        q_values = q_values.squeeze()
        value = torch.max(q_values).item()
    return value


# Semi-gradient Q-learning function
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
                action = greedy_action(qnet, state_tensor)

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

    return qnet


# Function to play episodes using the trained QNetwork
def play_episodes(q_network, env, num_episodes=5):
    for episode in range(num_episodes):
        state, _ = env.reset()
        total_reward = 0
        done = False

        while not done:
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            action = greedy_action(q_network, state_tensor)

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
            state = next_state

        print(f"Episode {episode + 1}/{num_episodes}, Total Reward: {total_reward}")



if __name__ == "__main__":
    env_name = "CartPole-v1"   #or "MountainCar-v0", "Acrobot-v1"
    env = gym.make(env_name)
    
    q = run_semigradient_qlearning(env, num_episodes=400, learning_rate=0.001, epsilon=0.1)
    env.close()

    test_env = gym.make(env_name, render_mode="human")
    play_episodes(q, test_env, 5)
    test_env.close()

