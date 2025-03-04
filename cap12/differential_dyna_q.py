# differential_dyna_q.py
import gymnasium as gym
import numpy as np
import random as rand

import sys
from os import path
sys.path.append( path.dirname( path.dirname( path.abspath(__file__) ) ) )

from util.qtable_helper import epsilon_greedy


def get_decayed_epsilon(step, total_steps, decay_steps=None, min_epsilon=0.0):
    """
    Calculate linearly decayed epsilon value
    
    Args:
        step: Current step number
        total_steps: Total number of steps in training
        decay_steps: Number of steps to reach min_epsilon (if None, uses total_steps)
        min_epsilon: Minimum value for epsilon
    """
    if decay_steps is None:
        decay_steps = total_steps
    else:
        decay_steps = min(decay_steps, total_steps)
        
    if step >= decay_steps:
        return min_epsilon
        
    return 1.0 - (1.0 - min_epsilon) * (step / decay_steps)


def planning_step(model, planning_steps, Q, lr, visit_counts, weighted_sampling=True):
    """
    Planning phase that can use either uniform sampling or sampling inversely proportional to visit counts
    
    Args:
        model: Dictionary containing (s,a) -> (r, next_s, mean_r) mappings
        planning_steps: Number of planning updates to perform
        Q: Q-value table
        lr: Learning rate
        visit_counts: Array tracking number of updates for each state-action pair
        weighted_sampling: If True, samples inversely proportional to visit counts. If False, uses uniform sampling.
    """
    all_s_a = list(model.keys())
    if len(all_s_a) == 0:
        return
        
    if weighted_sampling:
        # Calculate sampling weights as inverse of visit counts
        weights = [1.0 / visit_counts[s,a] for s,a in all_s_a]
        total_weight = sum(weights)
        if total_weight > 0:
            weights = [w/total_weight for w in weights]
        samples = rand.choices(all_s_a, weights=weights, k=min(planning_steps, len(all_s_a)))
    else:
        # Uniform sampling
        samples = rand.choices(all_s_a, k=min(planning_steps, len(all_s_a)))

    for s, a in samples:
        r, next_s, mean_r = model[(s,a)]
        V_next_s = np.max(Q[next_s])
        delta = (r - mean_r + V_next_s) - Q[s,a]
        Q[s,a] = Q[s,a] + lr * delta
        visit_counts[s,a] += 1


def run_differential_dynaq(env, total_steps, lr=0.1, lr_mean=0.1, epsilon_decay_steps=None, min_epsilon=0.0,
                          planning_steps=5, weighted_sampling=True, verbose=False):
    """
    Dyna-Q implementation for average reward MDPs with flexible planning sampling strategy
    
    Args:
        env: OpenAI Gym environment
        total_steps: Total number of steps to run
        lr: Learning rate for Q-value updates
        lr_mean: Learning rate for average reward updates
        epsilon_decay_steps: Steps to decay epsilon from 1.0 to min_epsilon (if None, uses total_steps)
        min_epsilon: Minimum value for epsilon after decay
        planning_steps: Number of planning updates per real step
        weighted_sampling: If True, uses inverse-count-based sampling for planning. If False, uses uniform sampling.
        verbose: If True, prints progress information
    """
    assert isinstance(env.observation_space, gym.spaces.Discrete)
    assert isinstance(env.action_space, gym.spaces.Discrete)

    num_states = env.observation_space.n
    num_actions = env.action_space.n

    # Initialize Q-table with small random values
    Q = np.random.uniform(low=-0.01, high=0.01, size=(num_states, num_actions))
    
    # Initialize visit counts for state-action pairs
    visit_counts = np.zeros_like(Q, dtype=int)
    
    # Initialize model dictionary for planning
    model = dict()
    
    # Initialize statistics
    rewards_per_step = []
    mean_reward = 0.0

    state, _ = env.reset()

    # Main loop
    for i in range(total_steps):
        # Get current epsilon value
        current_epsilon = get_decayed_epsilon(i, total_steps, epsilon_decay_steps, min_epsilon)
        
        # Choose action using epsilon-greedy
        action = epsilon_greedy(Q, state, current_epsilon)

        # Take action in environment
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        assert not done, "This algorithm is for infinite tasks!"

        # Value of next state
        V_next_state = np.max(Q[next_state])

        # Update Q-value using average reward formulation
        delta = (reward - mean_reward + V_next_state) - Q[state,action]
        Q[state,action] = Q[state,action] + lr * delta

        # Update average reward estimate
        mean_reward += lr_mean * delta

        # Update model and visit counts
        model[state,action] = (reward, next_state, mean_reward)
        visit_counts[state,action] += 1

        # Planning phase
        planning_step(model, planning_steps, Q, lr, visit_counts, weighted_sampling)

        rewards_per_step.append(reward)
        state = next_state

        # Print progress if requested
        if verbose and ((i+1) % 1000 == 0):
            avg_reward = np.mean(rewards_per_step[-100:])
            print(f"Step {i+1} Average Reward (last 100): {avg_reward:.3f} (epsilon: {current_epsilon:.3f})")
    
    if verbose:
        print("\nState-action visit counts:")
        print(visit_counts)

    return rewards_per_step, Q, mean_reward


if __name__ == "__main__":
    from envs import TwoChoiceEnv
    from util.qtable_helper import evaluate_qtable_policy
    from util.plot import plot_result

    # Create and run with TwoChoiceEnv
    env = TwoChoiceEnv()
    rewards, Q, avg_reward = run_differential_dynaq(
        env, 
        total_steps=100_000,
        lr=0.15,
        lr_mean=0.1,
        epsilon_decay_steps=80_000,  # Decay epsilon over first 50k steps
        min_epsilon=0.005,           # Don't go completely to zero
        planning_steps=10,
        weighted_sampling=False,
        verbose=True
    )
    
    print(f"\nFinal average reward estimate: {avg_reward:.4f}")
    print(f"Actual average reward (last 200 steps): {np.mean(rewards[-200:]):.4f}")
    
    # Evaluate final policy
    mean_reward, _ = evaluate_qtable_policy(env, Q, num_episodes=1)
    print(f"Policy mean reward: {mean_reward:.2f}")
    
    # Plot results
    plot_result(rewards, cumulative='avg')

    print("\nQ-table (state-action values):")
    for i in range(Q.shape[0]):
        print(f"state {i}:", Q[i])