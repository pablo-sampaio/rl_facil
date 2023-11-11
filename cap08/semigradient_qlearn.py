import numpy as np
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim


# Q-Network class
# Rede com camadas: estado x 32 x 128 x ações
class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(QNetwork, self).__init__()
        self.dense1 = nn.Linear(state_dim, 32)
        self.dense2 = nn.Linear(32, 128)
        self.output_layer = nn.Linear(128, action_dim)

    def forward(self, state):
        x = torch.relu(self.dense1(state))
        x = torch.relu(self.dense2(x))
        return self.output_layer(x)

    def greedy_action(self, state_tensor):
        with torch.no_grad():
            q_values = self(state_tensor)
            action = torch.argmax(q_values)
        return action.item()

    def max_value(self, state_tensor):
        with torch.no_grad():
            q_values = self(state_tensor)
            value = torch.max(q_values).item()
        return value


# Semi-gradient Q-learning function
def run_semigradient_qlearning(env, num_episodes=1000, learning_rate=0.001, discount_factor=0.99, epsilon=0.1):
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    q_network = QNetwork(state_dim, action_dim)
    optimizer = optim.Adam(q_network.parameters(), lr=learning_rate)

    for episode in range(num_episodes):
        state, _ = env.reset()
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)

        total_reward = 0
        done = False

        while not done:
            
            # Epsilon-greedy action selection
            if np.random.rand() < epsilon:
                action = env.action_space.sample()
            else:
                action = q_network.greedy_action(state_tensor)

            next_state, reward, terminated, truncated, _ = env.step(action)

            next_state_tensor = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)
            done = terminated or truncated

            # Calculate target Q-value
            target_q = torch.tensor(reward, dtype=torch.float32)
            if not terminated:
                next_state_value = q_network.max_value(next_state_tensor)
                target_q += discount_factor * next_state_value

            # Calculate current Q-value and update the network
            current_q = q_network(state_tensor)[0, action]
            loss = nn.MSELoss()(current_q, target_q)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_reward += reward
            #state = next_state
            state_tensor = next_state_tensor

        # Print the total reward obtained in this episode
        print(f"Episode {episode + 1}/{num_episodes}, Total Reward: {total_reward}")

    return q_network


# Function to play episodes using the trained QNetwork
def play_episodes(q_network, env, num_episodes=5):
    for episode in range(num_episodes):
        state, _ = env.reset()
        total_reward = 0
        done = False

        while not done:
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                action = q_network.greedy_action(state_tensor)

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
            state = next_state

        print(f"Episode {episode + 1}/{num_episodes}, Total Reward: {total_reward}")


# Example usage
if __name__ == "__main__":
    env_name = "CartPole-v1"   #"MountainCar-v0"
    env = gym.make(env_name)
    
    q = run_semigradient_qlearning(env, num_episodes=300, learning_rate=0.001, epsilon=0.1)
    play_episodes(q, env)

    env.close()
