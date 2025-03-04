# differential_nstep_sarsa.py
import gymnasium as gym
import numpy as np
import sys
from os import path
from collections import deque

sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

from util.qtable_helper import epsilon_greedy

def run_differential_nstep_sarsa(env, total_steps, n_steps=3, lr=0.1, lr_mean=0.01, epsilon=0.1, verbose=False):
    """
    Differential n-step SARSA for average reward formulation (step-based)
    """
    assert isinstance(env.observation_space, gym.spaces.Discrete)
    assert isinstance(env.action_space, gym.spaces.Discrete)

    num_actions = env.action_space.n
    
    # Initialize Q-table with small random values
    Q = np.random.uniform(low=-0.01, high=0.01, size=(env.observation_space.n, num_actions))
    mean_reward = 0.0
    rewards_per_step = []
    
    state, _ = env.reset()
    action = epsilon_greedy(Q, state, epsilon)
    
    # históricos de: estados, ações e recompensas
    hs = deque(maxlen=n_steps)
    ha = deque(maxlen=n_steps)
    hr = deque(maxlen=n_steps)
    
    step = 0
    
    while step < total_steps:
        # Collect experience
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        rewards_per_step.append(reward)

        assert not done, "This algorithm is for continuing tasks!"
        
        next_action = epsilon_greedy(Q, next_state, epsilon)
        
        hs.append(state)
        ha.append(action)
        hr.append(reward)

        # Update when buffer has enough steps
        if len(hs) >= n_steps:
            assert len(hs) == n_steps
            update_state = hs[0]
            update_action = ha[0]
            
            # Calculate n-step return using rewards from history
            G = sum((r - mean_reward) for r in hr)
            G += Q[next_state][next_action]
            
            # Calculate TD error and update
            delta = G - Q[update_state][update_action]
            Q[update_state][update_action] += lr * delta
            mean_reward += lr_mean * delta

        state = next_state
        action = next_action
        step += 1

        if verbose and (step + 1) % 1000 == 0:
            avg_reward = np.mean(rewards_per_step[-100:])
            print(f"Step {step+1} | Avg Reward (last 100): {avg_reward:.2f}")
    
    return rewards_per_step, Q, mean_reward


if __name__ == "__main__":
    from envs import TwoChoiceEnv
    from util.qtable_helper import evaluate_qtable_policy
    from util.plot import plot_result
    
    env = TwoChoiceEnv()
    rewards, Q, avg_reward = run_differential_nstep_sarsa(
        env,
        total_steps=10_000,
        n_steps=3,
        lr=0.2,
        lr_mean=0.05,
        epsilon=0.10,
        verbose=True
    )
    
    print(f"\nFinal average reward estimate: {avg_reward:.4f}")
    print(f"Actual average reward (last 200 steps): {np.mean(rewards[-200:]):.4f}")
    
    # Evaluate final policy
    mean_reward, _ = evaluate_qtable_policy(env, Q, num_episodes=1)
    print(f"Policy mean reward: {mean_reward:.2f}")
    
    # Plot results
    plot_result(rewards, cumulative='avg')