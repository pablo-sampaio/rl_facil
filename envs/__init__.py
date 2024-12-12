
from ._implementations.racetrack_env import RacetrackEnv
from ._implementations.robot_env import SimulatedRobotEnv
from ._implementations.access_control_env import AccessControlEnv
from ._implementations.two_choice_env import TwoChoiceEnv

import gymnasium as gym

gym.envs.registration.register(
    id="RaceTrack-v0",
    entry_point="envs:RacetrackEnv",  # Caminho para a classe
    max_episode_steps=100,
)
