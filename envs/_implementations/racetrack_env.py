import numpy as np
import gymnasium as gym
from gymnasium import spaces
import pygame

import sys
from os import path
sys.path.append( path.dirname( path.dirname( path.dirname( path.abspath(__file__) ) ) ) )

from envs.wrappers import convert_to_flattened_index


def find_positions_with_char(track, character):
    positions = []
    for y, row in enumerate(track):
        for x, ch in enumerate(row):
            if ch == character:
                positions.append((x, y))
    return positions


class RacetrackEnv(gym.Env):
    '''
    A simple environment for a car racing track, based on an environment proposed in the book of Sutton and Barto (2018). 
    
    # Actions
    In each dimension (x and y), the car can accelerate by -1, +1 or 0 (no acceleration).
    The actions represent the different combinations of acceleration in the x and y directions:
    * action 0 - (dx=-1, dy=-1) 
    * action 1: (dx=0, dy=-1) 
    * action 2: (dx=+1, dy=-1) 
    * ... 
    * action 8: (dx=+1, dy=+1)
    
    # Observations
    The observation is the actual state, given by a tuple (x, y, vx, vy), where:
    * x and y are the current position of the car
    * vx and vy are the current velocity in the x and y directions

    # Rewards
    * -1 for each time step
    * 0 when the car reaches the goal
    '''
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, render_mode="rgb_array", collision_restarts=False, observation_as_tuple=False):
        self.track = [
            "XXXXXXXXXXXXXXXXXX",
            "GG___XXXXXXXXXXXXX",
            "GG______XXXXXXXXXX",
            "GG________XXXXXXXX",
            "GG__________XXXXXX",
            "GG___________XXXXX",
            "GG____________XXXX",
            "XXXX___________XXX",
            "XXXXXXX_________XX",
            "XXXXXXXXX________X",
            "XXXXXXXXXX_______X",
            "XXXXXXXXXX_______X",
            "XXXXXXXXX_______XX",
            "XXXXXXXX________XX",
            "XXXXXXX________XXX",
            "XXXXXX_________XXX",
            "XXXXXX________XXXX",
            "XXXXX________XXXXX",
            "XXXXX_______XXXXXX",
            "XXXXX_______XXXXXX",
            "XXXXX________XXXXX",
            "XXXXXX_______XXXXX",
            "XXXXXX________XXXX",
            "XXXXXXX_______XXXX",
            "XXXXXXXSSSSSSSXXXX",
        ]        
        self.collision_restarts = collision_restarts
        self.render_mode = render_mode
        self.observation_as_tuple = observation_as_tuple
        
        self.action_space = spaces.Discrete(9)  # 9 possible actions (0-8)
        
        self.vel_limit = 3
        
        # Dimensions for x position / y position / x velocity / y velocity
        self.obs_dimensions = [len(self.track[0]), len(self.track), 2*self.vel_limit+1, 2*self.vel_limit+1]

        if observation_as_tuple:
            self.observation_space = spaces.Tuple((
                spaces.Discrete(self.obs_dimensions[0]),  # x position
                spaces.Discrete(self.obs_dimensions[1]),  # y position
                spaces.Discrete(self.obs_dimensions[2]),  # x velocity
                spaces.Discrete(self.obs_dimensions[3])   # y velocity
            ))
        else:
            self.observation_space = spaces.Discrete(np.prod(self.obs_dimensions))
        
        self.start_positions = find_positions_with_char(self.track, 'S')

        # para renderização
        self.screen = None
        self.clock = None

        self.colors = {
            'X': (0, 150, 0),     # Color for walls
            '_': (255, 255, 255), # Open tracks
            'G': (0, 0, 0),       # Goals
            'S': (180, 180, 180), # Start positions
            'A': (0, 0, 255),     # Blue for current position
        }
        self.square_size = 20  # Scale factor for rendering
        self.isopen = True

        self.reset()

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        idx = self.np_random.choice(len(self.start_positions))
        start_pos = self.start_positions[idx]
        self.current_state = (*start_pos, self.vel_limit, self.vel_limit)  # o valor de self.vel_limit representa a velocidade zero
        if self.render_mode == "human":
            self.render()
        if self.observation_as_tuple:
            return self.current_state, {}
        else:
            return convert_to_flattened_index(self.current_state, self.obs_dimensions), dict(track=list(self.track))
    
    def step(self, action):
        x, y, vx, vy = self.current_state
        vx = vx - self.vel_limit
        vy = vy - self.vel_limit
        
        # Map action to velocity changes
        # i.e. action 0: (dx=-1, dy=-1) / action 1: (dx=0, dy=-1) / action 2: (dx=+1, dy=-1) / ... / action 8: (dx=+1, dy=+1)
        dx = action % 3 - 1
        dy = action // 3 - 1
        
        # Update velocities with acceleration
        vx_new = vx + dx
        vy_new = vy + dy
        
        # Limit velocities
        vx_new = np.clip(vx_new, -self.vel_limit, self.vel_limit)
        vy_new = np.clip(vy_new, -self.vel_limit, self.vel_limit)
        
        # Update position
        x_new = x + vx_new
        y_new = y + vy_new
        
        # Handle track boundaries and wall colisions
        if x_new < 0 or x_new >= len(self.track[0]) \
                or y_new < 0 or y_new >= len(self.track) \
                or self.track[y_new][x_new] == 'X':
            if self.collision_restarts:
                # go to a random start position
                idx = self.np_random.choice(len(self.start_positions))
                #idx = np.random.choice(len(self.start_positions))
                x_new, y_new = self.start_positions[idx]
                vx_new, vy_new = (0, 0)
            else:
                # stop in current position
                x_new, y_new, vx_new, vy_new = x, y, 0, 0 
        
        # Velocity is internally represented summed to self.vel_limit, just to make it non-negative
        self.current_state = (x_new, y_new, vx_new + self.vel_limit, vy_new + self.vel_limit)
        
        if self.track[y_new][x_new] == 'G':
            reward = 0   # Reached the goal
            terminated = True
        else:
            reward = -1  # Time step penalty
            terminated = False
        
        if self.render_mode == "human":
            self.render()

        if self.observation_as_tuple:
            obs = self.current_state
        else:
            obs = convert_to_flattened_index(self.current_state, self.obs_dimensions)

        return obs, reward, terminated, False, {}

    def render_text(self):
        track_copy = self.track.copy()  # Create a copy of the track
        x, y, _, _ = self.current_state
        track_copy[y] = track_copy[y][:x] + 'A' + track_copy[y][x+1:]  # Mark the current position with 'A'
        for row in track_copy:
            print(row)

    def render(self):
        if self.screen is None:
            pygame.init()
            pygame.display.init()
            width = self.square_size * len(self.track[0])
            height = self.square_size * len(self.track)
            self.screen = pygame.display.set_mode((width, height))
            self.font = pygame.font.SysFont(None, 30)
        if self.clock is None:
            self.clock = pygame.time.Clock()

        self.screen.fill((255, 255, 255))  # Fill the screen with white

        for y, row in enumerate(self.track):
            for x, ch in enumerate(row):
                color = self.colors[ch]
                pygame.draw.rect(self.screen, color, (x * self.square_size, y * self.square_size, self.square_size, self.square_size))
        
        x, y, _, _ = self.current_state
        offset = 2
        pygame.draw.rect(self.screen, self.colors['A'], (x*self.square_size + offset, y*self.square_size + offset, self.square_size - 2*offset, self.square_size - 2*offset))

        if self.render_mode == "human":
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            pygame.display.flip()

        if self.render_mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
            )
        else:
            return self.isopen

    def close(self):
        if self.screen is not None:
            pygame.display.quit()
            pygame.quit()
            self.screen = False
            self.isopen = False


gym.envs.registration.register(
    id="RaceTrack-v0",
    entry_point="envs:RacetrackEnv",  # Caminho para a classe
    max_episode_steps=400,
)


if __name__=='__main__':
    import time
    #env = RacetrackEnv(collision_restarts=False, observation_as_tuple=True)
    env = gym.make("RaceTrack-v0", render_mode="human", collision_restarts=False, observation_as_tuple=True)
    
    state, _ = env.reset()

    terminated = truncated = False
    while not (terminated or truncated):
        action = env.action_space.sample()
        next_state, reward, terminated, truncated, _ = env.step(action)

        time.sleep(0.15)
        print("Action:", action)
        print("Next State:", next_state)
        print("Reward:", reward)
        print("Termi/Trunc:", terminated, truncated)
        print()
    
    env.close()
