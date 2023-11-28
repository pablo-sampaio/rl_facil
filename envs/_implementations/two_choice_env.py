
import gymnasium as gym
from gymnasium import spaces


class TwoChoiceEnv(gym.Env):
    def __init__(self, render_mode=None):
        super().__init__()
        # Define the action and observation spaces
        self.action_space = spaces.Discrete(2)      # Two discrete actions: 0 (left) and 1 (right)
        self.observation_space = spaces.Discrete(9) # Nine discrete states: 0 to 8

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        # Reset the environment to the initial state
        self.current_state = 0
        if self.render_mode == "human":
            self.render()
        return self.current_state, dict()

    def step(self, action):
        # Perform the specified action and transition to the next state
        if action != 0 and action != 1:
            raise ValueError("Invalid action!")

        reward = 0.0

        if self.current_state == 0:
            # left
            if action == 0:
                reward = 1.0
                self.current_state = 1
            # right
            else:
                reward = 0.0
                self.current_state = 5
        elif self.current_state == 4:
            reward = 0.0
            self.current_state = 0
        elif self.current_state == 8:
            reward = 2.0
            self.current_state = 0
        else:
            reward = 0.0
            self.current_state += 1

        if self.render_mode == "human":
            self.render()

        return self.current_state, reward, False, False, None

    def render(self):
        # Display the current state (optional)
        print("Current state:", self.current_state)
