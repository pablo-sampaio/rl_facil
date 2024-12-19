
from ._implementations.racetrack_env import RacetrackEnv
from ._implementations.robot_env import SimulatedRobotEnv
from ._implementations.access_control_env import AccessControlEnv
from ._implementations.two_choice_env import TwoChoiceEnv
from ._implementations.windygrid_env import WindyGridworldEnv

import gymnasium as gym

gym.envs.registration.register(
    id="RaceTrack-v0",
    #entry_point="envs:RacetrackEnv",  # Caminho para a classe
    entry_point="envs._implementations.racetrack_env:create_wrapped_racetrack_env",
    max_episode_steps=150,
)

gym.envs.registration.register(
    id="WindyGrid-v0",
    entry_point="envs._implementations.windygrid_env:create_wrapped_windy_grid_env"
)

gym.envs.registration.register(
    id="WindyGrid-v1",
    entry_point="envs._implementations.windygrid_env:create_wrapped_windy_grid_env",
    kwargs=dict(action_set="kings", wind_scheme="fixed"),
)

gym.envs.registration.register(
    id="WindyGrid-v2",
    entry_point="envs._implementations.windygrid_env:create_wrapped_windy_grid_env",
    kwargs=dict(action_set="kings+still", wind_scheme="stochastic"),
)
