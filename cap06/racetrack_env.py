import numpy as np
import gym
from gym import spaces
import pygame

import sys
from os import path
sys.path.append( path.dirname( path.dirname( path.abspath(__file__) ) ) )

from util.wrappers import convert_to_flattened_index


def get_positions(track, character):
    positions = []
    for y, row in enumerate(track):
        for x, ch in enumerate(row):
            if ch == character:
                positions.append((x, y))
    return positions


class RacetrackEnv(gym.Env):
    def __init__(self, book_collision=False):
        self.track = [
            "XXXXXXXXXXXXXXXXXX",
            "GG___XXXXXXXXXXXXX",
            "GG______XXXXXXXXXX",
            "GG________XXXXXXXX",
            "GG__________XXXXXX",
            "GG___________XXXXX",
            "GG____________XXXX",
            #"GG_____________XXX",
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
        self.book_collision = book_collision
        self.action_space = spaces.Discrete(9)  # 9 possible actions (0-8)
        
        self.vel_limit = 4
        # Dimensions for x position / y position / x velocity / y velocity
        self.obs_dimensions = [len(self.track[0]), len(self.track), 2*self.vel_limit+1, 2*self.vel_limit+1]
        self.observation_space = spaces.Discrete(np.prod(self.obs_dimensions))
        '''self.observation_space = spaces.Tuple((
            spaces.Discrete(self.obs_dimensions[0]),  # x position
            spaces.Discrete(self.obs_dimensions[1]),  # y position
            spaces.Discrete(self.obs_dimensions[2]),  # x velocity
            spaces.Discrete(self.obs_dimensions[3])  # y velocity
        ))'''
        
        self.start_positions = get_positions(self.track, 'S')
        self.reset()

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

    def reset(self):
        idx = np.random.choice(len(self.start_positions))
        start_pos = self.start_positions[idx]
        self.current_position = (*start_pos, self.vel_limit, self.vel_limit)  # o valor de self.vel_limit representa a velocidade zero
        #return self.current_position
        return convert_to_flattened_index(self.current_position, self.obs_dimensions)
    
    def step(self, action):
        x, y, vx, vy = self.current_position
        vx = vx- self.vel_limit
        vy = vy- self.vel_limit
        
        # Map action to velocity changes
        dx = action % 3 - 1
        dy = action // 3 - 1
        
        # Update velocities with acceleration
        vx_new = vx + dx
        vy_new = vy + dy
        
        # Limit velocities to [-4, 4]
        vx_new = np.clip(vx_new, -4, 4)
        vy_new = np.clip(vy_new, -4, 4)
        
        # Update position
        x_new = x + vx_new
        y_new = y + vy_new
        
        # Handle track boundaries and wall colisions
        if x_new < 0 or x_new >= len(self.track[0]) \
                or y_new < 0 or y_new >= len(self.track) \
                or self.track[y_new][x_new] == 'X':
            if self.book_collision:
                # go to a random start position
                idx = np.random.choice(len(self.start_positions))
                x_new, y_new = self.start_positions[idx]
                vx_new, vy_new = (0, 0)
            else:
                # stop in current position
                x_new, y_new, vx_new, vy_new = x, y, 0, 0 
        
        self.current_position = (x_new, y_new, vx_new + self.vel_limit, vy_new + self.vel_limit)
        
        if self.track[y_new][x_new] == 'G':
            reward = 0   # Reached the goal
            done = True
        else:
            reward = -1  # Time step penalty
            done = False
        
        obs = convert_to_flattened_index(self.current_position, self.obs_dimensions)
        return obs, reward, done, {}

    def render_text(self):
        track_copy = self.track.copy()  # Create a copy of the track
        x, y, _, _ = self.current_position
        track_copy[y] = track_copy[y][:x] + 'A' + track_copy[y][x+1:]  # Mark the current position with 'A'
        for row in track_copy:
            print(row)

    def render(self, mode="human"):
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
        
        x, y, _, _ = self.current_position
        offset = 2
        pygame.draw.rect(self.screen, self.colors['A'], (x*self.square_size + offset, y*self.square_size + offset, self.square_size - 2*offset, self.square_size - 2*offset))

        #pygame.display.flip()
        #self.clock.tick(30)  # Limit the frame rate
        #pygame.event.pump()  # Process events

        if mode == "human":
            pygame.event.pump()
            self.clock.tick(60)
            pygame.display.flip()

        if mode == "rgb_array":
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


if __name__=='__main__':
    import time
    env = RacetrackEnv()
    state = env.reset()
    env.render()

    done = False
    while not done:
        action = env.action_space.sample()
        next_state, reward, done, _ = env.step(action)

        time.sleep(0.1)
        env.render()
        #print("Action:", action)
        #print("Next State:", next_state)
        #print("Reward:", reward)
        #print("Done:", done)
        #print()
    
    env.close()


if __name__=='__main':
    from cap05.nstep_sarsa import run_nstep_sarsa
    from util.plot import plot_result
    from util.experiments import test_greedy_Q_policy

    EPISODES = 10_000
    LR = 0.1
    GAMMA = 0.95
    EPSILON = 0.1
    NSTEPS = 3

    env = RacetrackEnv()
    
    # Roda o algoritmo "n-step SARSA"
    rewards, qtable = run_nstep_sarsa(env, EPISODES, NSTEPS, LR, GAMMA, EPSILON, render=True)
    print("Últimos resultados: media =", np.mean(rewards[-20:]), ", desvio padrao =", np.std(rewards[-20:]))

    # Exibe um gráfico episódios x retornos (não descontados)
    plot_result(rewards, window=50)

    test_greedy_Q_policy(env, qtable, 10, True, render_wait=0.1)
    env.close()
