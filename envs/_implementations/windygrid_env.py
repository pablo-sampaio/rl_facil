import numpy as np
import gymnasium as gym
from gymnasium import spaces
import pygame


class WindyGridworldEnv(gym.Env):
    """
    A Gridworld environment with wind dynamics, inspired by Sutton and Barto's Reinforcement Learning book.

    The agent starts at a fixed position and must navigate to a goal position. Wind pushes the agent upwards in
    certain columns, according to wind strengths given per column. 

    The basic agent actions set includes: Left, Right, Up, Down. Additional diagonal actions can be added with the
    "kings" action set. A "kings+still" action set includes no movement as well.

    The wind_scheme can be set to stochastic, in which case the wind strength is randomly varied by +/- 1 with 50%
    probability (and 50% probability for the base wind strength).
    """

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 10
    }

    def __init__(self, action_set="basic", wind_scheme="fixed", render_mode="rgb_array", 
                 grid_size=(7, 10), start=(3, 0), goal=(3, 7), 
                 wind_strengths=[0, 0, 0, 1, 1, 1, 2, 2, 1, 0]):
        """
        Args:
            action_set (str): Action set: "basic", "kings", or "kings+still".
            wind_scheme (str): Wind scheme: "fixed" or "stochastic".
            render_mode (str): Rendering mode: "human" or "rgb_array".
            grid_size (tuple): Size of the grid (rows, columns).
            start (tuple): Starting position (row, column).
            goal (tuple): Goal position (row, column).
            wind_strengths (list): List of wind strengths for each column (length = grid_size[1]).
        """
        super().__init__()
        self.grid_size = grid_size
        self.start = start
        self.goal = goal
        self.wind_scheme = wind_scheme

        # Define action sets
        self.action_sets = {
            "basic": [(0, -1), (0, 1), (-1, 0), (1, 0)],  # Left, Right, Up, Down
            "kings": [(0, -1), (0, 1), (-1, 0), (1, 0), (-1, -1), (-1, 1), (1, -1), (1, 1)],  # Add diagonals
            "kings+still": [(0, -1), (0, 1), (-1, 0), (1, 0), (-1, -1), (-1, 1), (1, -1), (1, 1), (0, 0)]  # Add no movement
        }
        self.actions = self.action_sets[action_set]
        self.action_space = spaces.Discrete(len(self.actions))

        # Observation space
        self.observation_space = spaces.Tuple((
            spaces.Discrete(grid_size[0]),  # Row
            spaces.Discrete(grid_size[1])   # Column
        ))

        # Wind strengths
        assert len(wind_strengths) == grid_size[1], "Wind strengths must match the grid columns."
        self.wind_strengths = wind_strengths

        # Rendering setup
        self.render_mode = render_mode
        self.window = None
        self.cell_size = 50
        self.window_size = (self.grid_size[1] * self.cell_size, (self.grid_size[0] + 1) * self.cell_size)  # Extra row for wind display

        # State initialization
        self.reset()

    def reset(self):
        """
        Resets the environment to its initial state, with the agent placed in the 'start' position.

        Returns:
            observation (numpy array):
                The starting position of the agent as a pair (row, column).

            info (dict):
                Additional information (empty in this implementation).
        """
        self.agent_pos = list(self.start)
        if self.render_mode == "human":
            self._render_human()
        return tuple(self.agent_pos), {}

    def step(self, action):
        """
        Performs an action in the environment.

        Args:
            action (int):
                The index of the action to be performed.

        Returns:
            observation (tuple):
                The new position of the agent as (row, column).

            reward (int):
                The reward for the action. -1 for each step, 0 upon reaching the goal.

            done (bool):
                Indicates if the goal has been reached.

            truncated (bool):
                Always False; included for API compatibility.

            info (dict):
                Additional information (empty in this implementation).
        """
        move = self.actions[action]

        # Apply action
        new_pos = [self.agent_pos[0] + move[0], self.agent_pos[1] + move[1]]

        # Apply wind
        wind = self.wind_strengths[new_pos[1]] if 0 <= new_pos[1] < self.grid_size[1] else 0
        if self.wind_scheme == "stochastic" and wind != 0:
            wind += np.random.choice([-1, 0, 0, 1])  # Stochastic variation - 50% chance of being the same as the base wind strength

        new_pos[0] -= wind

        # Ensure agent stays within bounds
        new_pos[0] = max(0, min(self.grid_size[0] - 1, new_pos[0]))
        new_pos[1] = max(0, min(self.grid_size[1] - 1, new_pos[1]))

        # Update agent position
        self.agent_pos = new_pos

        # Check if goal is reached
        terminated = (self.agent_pos == list(self.goal))
        reward = 0 if terminated else -1

        if self.render_mode == "human":
            self.render()

        return tuple(self.agent_pos), reward, terminated, False, {}

    def render_text(self):
        grid = np.full(self.grid_size, fill_value=".")
        grid[self.goal[0], self.goal[1]] = "G"
        grid[self.agent_pos[0], self.agent_pos[1]] = "A"
        print("\n".join(["".join(row) for row in grid]))
        print()

    def render(self):
        if self.render_mode == "human":
            self._render_human()
        elif self.render_mode == "rgb_array":
            return self._render_rgb_array()

    def _render_human(self):
        if self.window is None:
            pygame.init()
            self.window = pygame.display.set_mode(self.window_size)
            pygame.display.set_caption("Windy Gridworld")
            self.clock = pygame.time.Clock()
        
        self.window.fill((255, 255, 255))  # White background
        font = pygame.font.SysFont(None, 24)

        # Draw grid and wind strengths
        for col in range(self.grid_size[1]):
            for row in range(self.grid_size[0]):
                rect = pygame.Rect(col * self.cell_size, row * self.cell_size, self.cell_size, self.cell_size)
                pygame.draw.rect(self.window, (0, 0, 0), rect, 1)  # Draw cell border

            # Display wind strength
            wind_text = font.render(str(self.wind_strengths[col]), True, (0, 0, 0))
            text_rect = wind_text.get_rect(center=(col * self.cell_size + self.cell_size // 2, self.grid_size[0] * self.cell_size + self.cell_size // 2))
            self.window.blit(wind_text, text_rect)

        # Draw goal
        goal_rect = pygame.Rect(
            self.goal[1] * self.cell_size, self.goal[0] * self.cell_size, self.cell_size, self.cell_size
        )
        pygame.draw.rect(self.window, (0, 255, 0), goal_rect)  # Green for goal

        # Draw agent
        agent_rect = pygame.Rect(
            self.agent_pos[1] * self.cell_size, self.agent_pos[0] * self.cell_size, self.cell_size, self.cell_size
        )
        pygame.draw.ellipse(self.window, (255, 0, 0), agent_rect)  # Red circle for agent

        pygame.event.pump()
        self.clock.tick(self.metadata["render_fps"])
        pygame.display.flip()

    def _render_rgb_array(self):
        surface = pygame.Surface(self.window_size)
        surface.fill((255, 255, 255))  # White background
        font = pygame.font.SysFont(None, 24)

        # Draw grid and wind strengths
        for col in range(self.grid_size[1]):
            for row in range(self.grid_size[0]):
                rect = pygame.Rect(col * self.cell_size, row * self.cell_size, self.cell_size, self.cell_size)
                pygame.draw.rect(surface, (0, 0, 0), rect, 1)  # Draw cell border

            # Display wind strength
            wind_text = font.render(str(self.wind_strengths[col]), True, (0, 0, 0))
            text_rect = wind_text.get_rect(center=(col * self.cell_size + self.cell_size // 2, self.grid_size[0] * self.cell_size + self.cell_size // 2))
            surface.blit(wind_text, text_rect)

        # Draw goal
        goal_rect = pygame.Rect(
            self.goal[1] * self.cell_size, self.goal[0] * self.cell_size, self.cell_size, self.cell_size
        )
        pygame.draw.rect(surface, (0, 255, 0), goal_rect)  # Green for goal

        # Draw agent
        agent_rect = pygame.Rect(
            self.agent_pos[1] * self.cell_size, self.agent_pos[0] * self.cell_size, self.cell_size, self.cell_size
        )
        pygame.draw.ellipse(surface, (255, 0, 0), agent_rect)  # Red circle for agent

        return np.array(pygame.surfarray.array3d(surface))

    def close(self):
        if self.window is not None:
            pygame.quit()
            self.window = None


if __name__ == "__main__":
    # Example usage
    env = WindyGridworldEnv(action_set="kings", wind_scheme="stochastic", render_mode="human")
    #env = WindyGridworldEnv()

    obs, _ = env.reset()
    print("Initial Observation:", obs)

    #for _ in range(100):
    terminated = False
    steps = 0
    while not terminated:
        action = env.action_space.sample()
        obs, reward, terminated, _, _ = env.step(action)
        print(obs)
        steps += 1

    if terminated:
        print(f"Goal reached in {steps=}!")
