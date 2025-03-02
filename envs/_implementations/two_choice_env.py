
import gymnasium as gym
from gymnasium import spaces


class TwoChoiceEnv(gym.Env):
    def __init__(self, render_mode=None, coherent_action=False):
        super().__init__()
        # Define the action and observation spaces
        self.action_space = spaces.Discrete(2)      # Two discrete actions: 0 (left) and 1 (right)
        self.observation_space = spaces.Discrete(9) # Nine discrete states: 0 to 8
        if coherent_action:
            # actions x states -> next_state
            # in this version, action 0 (left) is demanded in all states of the left cycle 
            # (in order to progress) and action 1 is demanded on the right cycle
            self.transitions = [[1, 2, 3, 4, 0, 5, 6, 7, 8],
                                [5, 1, 2, 3, 4, 6, 7, 8, 0]]
        else:
            # actions x states -> next_state
            # this is the original version
            self.transitions = [[1, 2, 3, 4, 0, 6, 7, 8, 0],
                                [5, 2, 3, 4, 0, 6, 7, 8, 0]]

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

        next_state = self.transitions[action][self.current_state]

        if self.current_state == 0 and next_state == 1:
            reward = 1.0
        elif self.current_state == 8 and next_state == 0:
            reward = 2.0
        else:
            reward = 0.0

        if self.render_mode == "human":
            self.render()

        self.current_state = next_state
        return self.current_state, reward, False, False, None

    def render(self):
        # Display the current state (optional)
        print("Current state:", self.current_state)


if __name__ == "__main__":
    env = TwoChoiceEnv()
    env.reset()
    for _ in range(20):
        action = env.action_space.sample()
        state, reward, done, truncated, info = env.step(action)
        print(f"{action=}")
        env.render()
