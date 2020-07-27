''' Class for Maze environment. '''

import numpy as np
import matplotlib.pyplot as plt
from IPython import display
from random import choice
import time
from copy import deepcopy
from itertools import product

# Some colours
LIGHT_RED = '#FFC4CC'
LIGHT_GREEN = '#95FD99'
BLACK = '#000000'
WHITE = '#FFFFFF'
LIGHT_PURPLE = '#E8D0FF'
LIGHT_ORANGE = '#FAE0C3'


# Implemented methods
methods = ['DynProg', 'ValIter']

class Maze:

    # Actions
    STAY       = 0
    MOVE_LEFT  = 1
    MOVE_RIGHT = 2
    MOVE_UP    = 3
    MOVE_DOWN  = 4

    # Give names to actions
    actions_names = {
        STAY: "stay",
        MOVE_LEFT: "move left",
        MOVE_RIGHT: "move right",
        MOVE_UP: "move up",
        MOVE_DOWN: "move down"
    }

    # Reward values
    STEP_REWARD = 0
    GOAL_REWARD = 1
    IMPOSSIBLE_REWARD = -100
    DEAD_REWARD = - 200


    def __init__(self, maze, weights=None, random_rewards=False, enemy=None, e_can_stay=False):
        """ Constructor of the environment Maze.
        """
        self.enemy             = False
        self.win               = False
        self.maze              = maze
        self.coords            = [(x, y) for x, y in np.ndindex(self.maze.shape)]
        self.goal              = [coord for coord in self.coords if self.maze[coord] == 2]
        self.actions           = self.__actions()
        self.n_actions         = len(self.actions)
        self.id2states, self.states2id = self.__states()
        self.n_states          = len(self.id2states)

        if enemy:
            self.enemy      = True
            self.e_start    = enemy
            self.e_actions  = self.__enemyActions(e_can_stay)
            self.e_id2states, self.e_states2id = self.__enemyStates()
            self.e_n_states = len(self.e_id2states)
            # Joint states
            self.id2jointStates, self.jointStates2id = self.__jointStates()
            self.n_jointStates = len(self.id2jointStates)
            self.dyingStates   = self.__dyingStates()
            self.winningStates = self.__winningStates()
            # self.e_c, self.e_b = self.__bordersAndCorners()

        self.T, self.R = self.__transitionsAndRewards(
            weights=weights, random_rewards=random_rewards)

    def __actions(self):
        actions = dict()
        actions[self.STAY]       = (0, 0)
        actions[self.MOVE_LEFT]  = (0,-1)
        actions[self.MOVE_RIGHT] = (0, 1)
        actions[self.MOVE_UP]    = (-1,0)
        actions[self.MOVE_DOWN]  = (1,0)
        return actions

    def __states(self):
        coords    = [(i, (x, y)) for i, (x, y) in enumerate(np.ndindex(self.maze.shape)) \
            if self.maze[x,y] != 1]
        # Mapping from index state to x,y coordinates
        id2states = dict([(i, s) for i, s in (coords)])
        # Invert mapping from coordinates to index
        states2id = {v: k for k, v in id2states.items()}
        return id2states, states2id
    
    def __enemyStates(self):
        id2states = dict([(i, s) for i, s in enumerate(self.coords)])
        states2id = {v: k for k, v in id2states.items()}
        return id2states, states2id
    
    def __enemyActions(self, e_can_stay):
        e_actions = deepcopy(self.actions)
        # Enemy not allowed to stay
        if not e_can_stay:
            del e_actions[0]
        return e_actions

    def __e_possibleMoves(self, position):
        possible_act = []
        for idx, action in self.e_actions.items():
            row = position[0] + action[0]
            col = position[1] + action[1]
            hitting_maze_walls = (row == -1) or (row == self.maze.shape[0]) or \
                                (col == -1) or (col == self.maze.shape[1])
            if not hitting_maze_walls:
                possible_act.append(idx)
        return possible_act

    def __jointStates(self):
        id2jointStates = dict([(i, s) for i, s in \
            enumerate(product(self.states2id.keys(), self.e_states2id.keys()))])
        jointStates2id = {v: k for k, v in id2jointStates.items()}
        # [(a,b) for a, b in product(self.id2states.keys(), self.e_id2states.keys())]
        return id2jointStates, jointStates2id

    def __dyingStates(self):
        # Get indexes of states that are at the same position
        idx = [i for i, A in enumerate(self.id2jointStates.values()) if A[0] == A[1]]
        return idx

    def __winningStates(self):
        # Get indexes of states that are at the same position
        idx = [i for i, A in enumerate(self.id2jointStates.values()) if A[0] in self.goal]
        return idx

    def __move(self, state, action):
        """ Makes a step in the maze, given a current position and an action.
            If the action STAY or an inadmissible action is used, the agent stays in place.
            :return tuple next_cell: Position (x,y) on the maze that agent transitions to.
        """
        # Compute the future position given current (state, action)
        row = state[0] + self.actions[action][0]
        col = state[1] + self.actions[action][1]
        # Is the future position an impossible one ?
        hitting_maze_walls =  (row == -1) or (row == self.maze.shape[0]) or \
                              (col == -1) or (col == self.maze.shape[1]) or \
                              (self.maze[row,col] == 1)
        # Return both (x,y) and index state
        if hitting_maze_walls:
            return state, self.states2id[state]
        else:
            return (row, col), self.states2id[(row, col)]

    def __e_move(self, state, action=None):
        """ Makes a step in the maze, given a current position and an action.
            If the action STAY or an inadmissible action is used, the agent stays in place.
            :return tuple next_cell: Position (x,y) on the maze that agent transitions to.
        """
        # Random action
        if action == None:
            action  = choice(list(self.e_actions.keys()))
        # Compute the future position given current (state, action)
        row = state[0] + self.e_actions[action][0]
        col = state[1] + self.e_actions[action][1]
        # Is the future position an impossible one ?
        outside_maze = (row == -1) or (row == self.maze.shape[0]) or \
            (col == -1) or (col == self.maze.shape[1])
        # Based on the impossiblity check return the next state.
        while outside_maze:
            # The enemy cannot choose a wrong position. Try another one.
            action  = choice(list(self.e_actions.keys()))
            row = state[0] + self.e_actions[action][0]
            col = state[1] + self.e_actions[action][1]
            outside_maze = (row == -1) or (row == self.maze.shape[0]) or \
                (col == -1) or (col == self.maze.shape[1])
        # return np.ravel_multi_index((row, col), dims=self.maze.shape)
        return (row, col), self.e_states2id[(row, col)]

    def __transitionsAndRewards(self, weights=None, random_rewards=None):
        """ Computes the transition probabilities for every state action pair.
            :return numpy.tensor transition probabilities: tensor of transition
            probabilities of dimension S*S*A
        """
        # Initialize the transition probailities tensor (S,S,A)
        dimensions = (self.n_jointStates,self.n_jointStates,self.n_actions)
        T = np.zeros(dimensions)
        R = np.zeros((self.n_jointStates, self.n_actions))

        # Compute the transition probabilities. Note that the transitions
        # are deterministic.
        for i, jState in self.id2jointStates.items(): # in range(self.n_states):
            # jState[0] = Player state / jState[1] = enemy state
            # print('Index, joint state : ', (i, jState))
            s, e       = jState
            if s == e and s not in self.winningStates:
                # print('Dead state')
                T[:, i, :] = 0
                R[   i, :] = self.DEAD_REWARD
                continue

            # For each action of the enemy
            e_possible_actions = self.__e_possibleMoves(e)
            p = 1./len(e_possible_actions)

            for e_action in e_possible_actions:
                # Move the enemy
                next_e, _ = self.__e_move(e, e_action)
                # For each action of the player

                for action in self.actions:
                    next_s, _ = self.__move(s, action)
                    # Take the index of the new joint state
                    i_next    = self.jointStates2id[(next_s, next_e)]
                    # Update transition probability from i-th to i_next-th joint state
                    T[i_next, i, action] = p

                    if weights is None:
                        # Reward for reaching the exit
                        if self.maze[next_s] == 2:
                            R[i, 1:] += self.STEP_REWARD
                            # Stay is the best action to take
                            R[i, 0] += self.GOAL_REWARD
                        # Dead state
                        elif i_next in self.dyingStates:
                            R[i, action] += self.DEAD_REWARD
                        # Reward for impossible action
                        elif s == next_s and action != self.STAY:
                            R[i, action] += self.IMPOSSIBLE_REWARD
                        # Reward for taking a step to an empty cell that is not the exit
                        else:
                            R[i, action] += self.STEP_REWARD
                        # If there exists trapped cells with probability 0.5
                        if random_rewards and self.maze[next_s] < 0:
                            row, col = next_s
                            # With probability 0.5 the reward is
                            r1 = (1 + abs(self.maze[row, col])) * R[s, action]
                            # With probability 0.5 the reward is
                            r2 = R[s, action]
                            # The average reward
                            R[s, action] = 0.5*r1 + 0.5*r2
                    # If the weights are described by a weight matrix
                    else:
                        i, j = next_s
                        # Simply put the reward as the weights o the next state.
                        R[s, action] = weights[i][j]

        return T, R

    def simulate(self, start, policy, method):
        if method not in methods:
            error = 'ERROR: the argument method must be in {}'.format(methods)
            raise NameError(error)

        path, path_enemy = [], []

        if method == 'DynProg':
            # Deduce the horizon from the policy shape
            horizon = policy.shape[1]
            # Initialize current state and time
            t = 0
            s, _   = start, self.states2id[start]
            e_s, _ = self.e_start, self.e_states2id[self.e_start]
            # Add the starting position in the maze to the path
            path.append(s)
            path_enemy.append(e_s)

            while t < horizon-1 and not self.win:

                i = self.jointStates2id[(s, e_s)]
                # Move to next state given the policy and the current state
                next_s, _ = self.__move(s, policy[i,t])
                path.append(next_s)
                # Move enemy
                next_e_s, _ = self.__e_move(e_s)
                path_enemy.append(next_e_s)

                if self.maze[next_s] == 2:
                    self.win = True
                    path.append(next_s)
                    path_enemy.append(next_e_s)
                # Update time and state for next iteration
                t +=1
                s      = next_s
                e_s    = next_e_s

        if method == 'ValIter':
            # Initialize current state, next state and time
            t = 1
            s = start
            # Add the starting position in the maze to the path
            path.append(start)
            # Move to next state given the policy and the current state
            next_s, _ = self.__move(s, policy[s])
            # Add the position in the maze corresponding to the next state
            # to the path
            path.append(next_s)
            # Loop while state is not the goal state
            while s != next_s:
                # Update state
                s = next_s
                # Move to next state given the policy and the current state
                next_s, next_s_id = self.__move(s, policy[s])
                # Add the position in the maze corresponding to the next state
                # to the path
                path.append(next_s)
                # Update time and state for next iteration
                t +=1

        self.win = False
        return path, path_enemy

    def show(self):
        print('The states are :')
        print(self.id2states)
        print('The actions are:')
        print(self.actions)
        print('The mapping of the states:')
        print(self.states2id)
        print('The rewards:')
        print(self.R)


def draw_maze(maze):

    # Mapp a color to each cell in the maze
    col_mapp = {0: WHITE, 1: BLACK,
                2: LIGHT_GREEN, -6: LIGHT_RED, -1: LIGHT_RED}

    # Give a color to each cell
    rows, cols = maze.shape
    colored_maze = [[col_mapp[maze[j, i]]
                     for i in range(cols)] for j in range(rows)]

    # Create figure of the size of the maze
    fig = plt.figure(1, figsize=(cols, rows))

    # Remove the axis ticks and add title title
    ax = plt.gca()
    ax.set_title('The Maze')
    ax.set_xticks([])
    ax.set_yticks([])

    # Create a table to color
    grid = plt.table(cellText=None,
                     cellColours=colored_maze,
                     cellLoc='center',
                     loc=(0, 0),
                     edges='closed')
    # Modify the hight and width of the cells in the table
    tc = grid.properties()['child_artists']
    for cell in tc:
        cell.set_height(1.0/rows)
        cell.set_width(1.0/cols)


def animate_solution(maze, path, path_enemy=None):

    # Mapp a color to each cell in the maze
    col_mapp = {0: WHITE, 1: BLACK,
                2: LIGHT_GREEN, -6: LIGHT_RED, -1: LIGHT_RED}

    # Size of the maze
    rows, cols = maze.shape

    # Create figure of the size of the maze
    fig = plt.figure(1, figsize=(cols, rows))

    # Remove the axis ticks and add title title
    ax = plt.gca()
    ax.set_title('Policy simulation')
    ax.set_xticks([])
    ax.set_yticks([])

    # Give a color to each cell
    colored_maze = [[col_mapp[maze[j, i]]
                     for i in range(cols)] for j in range(rows)]

    # Create a table to color
    grid = plt.table(cellText=None,
                     cellColours=colored_maze,
                     cellLoc='center',
                     loc=(0, 0),
                     edges='closed')

    # Modify the hight and width of the cells in the table
    tc = grid.properties()['child_artists']
    for cell in tc:
        cell.set_height(1.0/rows)
        cell.set_width(1.0/cols)

    # Update the color at each frame
    for i in range(len(path)):
        grid.get_celld()[(path[i])].set_facecolor(LIGHT_ORANGE)
        grid.get_celld()[(path[i])].get_text().set_text('Player')
        if path_enemy:
            grid.get_celld()[(path_enemy[i])].set_facecolor(LIGHT_RED)
            grid.get_celld()[(path_enemy[i])].get_text().set_text('Enemy')
        if i > 0:
            if path[i] == path[i-1] and i != len(path)-1:
                grid.get_celld()[(path[i])].set_facecolor(LIGHT_PURPLE)
                grid.get_celld()[(path[i])].get_text().set_text(
                    'Player is waiting')

            if path[i] == path_enemy[i]:
                grid.get_celld()[(path[i])].set_facecolor(LIGHT_RED)
                grid.get_celld()[(path[i])].get_text().set_text('Game Over')
                break

            elif path[i] == path[i-1]:
                grid.get_celld()[(path[i])].set_facecolor(LIGHT_GREEN)
                grid.get_celld()[(path[i])].get_text().set_text(
                    'Player is out')
            elif path[i-1] != path_enemy[i]:
                grid.get_celld()[(path[i-1])
                                 ].set_facecolor(col_mapp[maze[path[i-1]]])
                grid.get_celld()[(path[i-1])].get_text().set_text('')
            if path_enemy[i-1] != path[i] and path_enemy[i] != path_enemy[i-1]:
                grid.get_celld()[(path_enemy[i - 1])
                                 ].set_facecolor(col_mapp[maze[path_enemy[i - 1]]])
                grid.get_celld()[(path_enemy[i - 1])].get_text().set_text('')

        display.display(fig)
        display.clear_output(wait=True)
        time.sleep(1)
