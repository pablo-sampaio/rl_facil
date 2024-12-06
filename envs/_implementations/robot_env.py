
import gymnasium as gym

from enum import Enum, unique
from itertools import product

@unique
class Direction(Enum):
    # Important: numbered in clocwise order !
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3

@unique
class Action(Enum):
    FRONT = 0
    TURN_CW = 1
    TURN_COUNTER_CW = 2


## TODO: ainda não está compliant com o gymnasium
##  - definir action e observation spaces
##  - renderização?

class SimulatedRobotEnv(gym.Env):

    def __init__(self, count_visits=False, use_real_state=False, reward_option='goal', allow_all_actions=True):
        # codes used in the bi-dimensional map:
        # - 0 is corridor; 1 is wall; 2 is goal (terminal state); 
        # - 3 is start position; 
        # - 4 is a hole (terminal state); 
        self.map = [ 
            [ 0, 0, 0, 0, 4, 0, 2],
            [ 0, 0, 0, 0, 0, 0, 0],
            [ 0, 1, 1, 1, 1, 1, 1],
            [ 0, 0, 0, 0, 0, 0, 0],
            [ 0, 0, 0, 3, 0, 0, 0],
        ]
        # 'transition' reflects the absolute view of the map
        # while "observation" is the view of the agent
        self.initial_state = (4, 3, Direction.UP)  # row, column, orientation
        self.goal_position = (0, 6)
        assert self.map[self.initial_state[0]][self.initial_state[1]] == 3, "Initial position in the map should have value 3"
        assert self.map[self.goal_position[0]][self.goal_position[1]] == 2, "Goal position in the map should have value 2"
        self.state = None
        self.observation = None
        self.use_real_state = use_real_state
        
        assert reward_option in ['goal', 'step_cost']
        self.reward_option = reward_option
        if reward_option == 'goal':
            self.STEP_REWARD = 0.0
            self.GOAL_REWARD = 1.0
            self.HOLE_REWARD = -1.0
        else:
            self.STEP_REWARD = -1.0
            self.GOAL_REWARD = 0.0
            self.HOLE_REWARD = -50.0

        if count_visits:
            self.visits = [[0 for x in range(len(self.map[0]))] for x in range(len(self.map))]
        else:
            self.visits = None
        self.count_visits = count_visits
        
        self.actionset = tuple(Action) # immuttable
        actionset_no_front = list(Action)
        actionset_no_front.remove(Action.FRONT)
        self.actionset_no_front = tuple(actionset_no_front)


    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.state = self.initial_state 
        if self.use_real_state:
            self.observation = self.state
        else:
            self.observation = (0, 0, Direction.UP)
        if self.visits is not None:
            self.visits[self.state[0]][self.state[1]] += 1
        return self.observation

    def _internal_apply_action(self, obs, action):
        row, col, direction = obs
        if action == Action.FRONT:
            if direction == Direction.UP:
                row -= 1
            elif direction == Direction.DOWN:
                row += 1
            elif direction == Direction.RIGHT:
                col += 1
            elif direction == Direction.LEFT:
                col -= 1
            else:
                raise Exception()
        elif action == Action.TURN_CW:
            direction = Direction( (direction.value + 1) % 4 )
        elif action == Action.TURN_COUNTER_CW:
            direction = Direction( (direction.value - 1) % 4)
        else:
            raise Exception("Invalid action")
        return (row, col, direction)

    def reset_visits(self):
        old_visits = self.visits
        if self.count_visits:
            self.visits = [[0 for x in range(len(self.map[0]))] for x in range(len(self.map))]
        return old_visits

    def step(self, action):
        assert self.state is not None, "Environment must be reset"

        new_state = self._internal_apply_action(self.state, action)
        
        if 0 <= new_state[0] < len(self.map) \
                and 0 <= new_state[1] < len(self.map[0]) \
                and self.map[new_state[0]][new_state[1]] != 1:
            self.state = new_state
            self.observation = self._internal_apply_action(self.observation, action)
            if self.visits is not None and action == Action.FRONT:
                self.visits[new_state[0]][new_state[1]] += 1
        else:
            # invalid moves: don't change state
            new_state = self.state
        
        is_terminal = False
        if self.map[new_state[0]][new_state[1]] == 2:  # goal
            is_terminal = True
            reward = self.GOAL_REWARD
        elif self.map[new_state[0]][new_state[1]] == 4:  # hole
            is_terminal = True
            reward = self.HOLE_REWARD
        else:
            is_terminal = False
            reward = self.STEP_REWARD
        
        if is_terminal:
            self.state = None  # indicates that a reset is needed (but it is not returned in this step)

        return self.observation, reward, is_terminal, False, None
