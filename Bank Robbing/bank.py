''' Class for Bank environment. '''

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


class Bank:

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
    IN_BANK_REWARD    = 1
    IMPOSSIBLE_REWARD = -100
    CAUGHT_REWARD     = -10


    def __init__(self, grid):
        """ Constructor of the environment Bank (and town).
        """
        # Town attributes
        self.grid              = grid
        self.start             = tuple(np.argwhere(grid == 1).reshape(1,-1)[0])
        self.money             = tuple(np.argwhere(grid == 3).reshape(1, -1)[0])
        self.coords            = [(x, y) for x, y in np.ndindex(self.grid.shape)]
        # Robber attributes
        self.caught            = False
        self.actions           = self.__actions()
        self.n_actions         = len(self.actions)
        self.id2states, self.states2id = self.__states()
        self.n_states          = len(self.id2states)
        # Police attributes
        self.e_start           = tuple(np.argwhere(grid == 2).reshape(1, -1)[0])
        self.e_actions         = self.__enemyActions()
        self.e_id2states, self.e_states2id = self.__enemyStates()
        self.e_n_states        = len(self.e_id2states)
        # Joint states
        self.id2jointStates, self.jointStates2id = self.__jointStates()
        self.n_jointStates     = len(self.id2jointStates)
        # We are oblivious to the good / bad states at first
        # Losing states are those when player pos == police pos
        self.losingStates      = self.__losingStates()
        self.winningStates     = []

    def __actions(self):
        actions = dict()
        actions[self.STAY]       = (0, 0)
        actions[self.MOVE_LEFT]  = (0,-1)
        actions[self.MOVE_RIGHT] = (0, 1)
        actions[self.MOVE_UP]    = (-1,0)
        actions[self.MOVE_DOWN]  = (1,0)
        return actions

    def __states(self):
        coords    = [(i, (x, y)) for i, (x, y) in enumerate(np.ndindex(self.grid.shape))]
        # Mapping from index state to x,y coordinates
        id2states = dict([(i, s) for i, s in (coords)])
        # Invert mapping from coordinates to index
        states2id = {v: k for k, v in id2states.items()}
        return id2states, states2id

    def __losingStates(self):
        # Get indexes of states that are at the same position
        idx = set([i for i, A in enumerate(
            self.id2jointStates.values()) if A[0] == A[1]])
        return idx

    def __enemyStates(self):
        id2states = dict([(i, s) for i, s in enumerate(self.coords)])
        states2id = {v: k for k, v in id2states.items()}
        return id2states, states2id
    
    def __enemyActions(self):
        e_actions = deepcopy(self.actions)
        # Enemy not allowed to stay
        del e_actions[0]
        return e_actions

    def __e_possibleMoves(self, position):
        possible_act = []
        for idx, action in self.e_actions.items():
            row = position[0] + action[0]
            col = position[1] + action[1]
            hitting_maze_walls = (row == -1) or (row == self.grid.shape[0]) or \
                                (col == -1) or (col == self.grid.shape[1])
            if not hitting_maze_walls:
                possible_act.append(idx)
        return possible_act

    def __jointStates(self):
        id2jointStates = dict([(i, s) for i, s in \
            enumerate(product(self.states2id.keys(), self.e_states2id.keys()))])
        jointStates2id = {v: k for k, v in id2jointStates.items()}
        # [(a,b) for a, b in product(self.id2states.keys(), self.e_id2states.keys())]
        return id2jointStates, jointStates2id

    def __move(self, state, action=None):
        """ Makes a step in the grid, given a current position and an action.
            If the action STAY or an inadmissible action is used, the agent stays in place.
            :return tuple next_cell: Position (x,y) on the grid that agent transitions to.
        """
        # Random action
        if action == None:
            action  = choice(list(self.actions.keys()))
        # Compute the future position given current (state, action)
        row = state[0] + self.actions[action][0]
        col = state[1] + self.actions[action][1]
        # Is the future position an impossible one ?
        hitting_maze_walls =  (row == -1) or (row == self.grid.shape[0]) or \
                              (col == -1) or (col == self.grid.shape[1])
        # Return both (x,y) and index state
        if hitting_maze_walls:
            return state, self.states2id[state]
        else:
            return (row, col), self.states2id[(row, col)]

    def __e_move(self, state, action=None):
        """ Makes a step in the grid, given a current position and an action.
            If the action STAY or an inadmissible action is used, the agent stays in place.
            :return tuple next_cell: Position (x,y) on the grid that agent transitions to.
        """
        # Random action
        if action == None:
            action  = choice(list(self.e_actions.keys()))
        # Compute the future position given current (state, action)
        row = state[0] + self.e_actions[action][0]
        col = state[1] + self.e_actions[action][1]
        # Is the future position an impossible one ?
        outside_maze = (row == -1) or (row == self.grid.shape[0]) or \
            (col == -1) or (col == self.grid.shape[1])
        # Based on the impossiblity check return the next state.
        while outside_maze:
            # The enemy cannot choose a wrong position. Try another one.
            action  = choice(list(self.e_actions.keys()))
            row = state[0] + self.e_actions[action][0]
            col = state[1] + self.e_actions[action][1]
            outside_maze = (row == -1) or (row == self.grid.shape[0]) or \
                (col == -1) or (col == self.grid.shape[1])
        # return np.ravel_multi_index((row, col), dims=self.grid.shape)
        return (row, col), self.e_states2id[(row, col)]


    def simulate(self, policy, tlimit, player=None, police=None):

        path, path_enemy = [], []
        t = 0

        if player == None:
            s, _   = self.start, self.states2id[self.start]
        else:
            s, _ = player, self.states2id[player]
        if police == None:
            e_s, _ = self.e_start, self.e_states2id[self.e_start]
        else:
            e_s, _ = police, self.e_states2id[police]

        # Add the starting position in the grid to the path
        path.append(s)
        path_enemy.append(e_s)

        while t < tlimit-1 and not self.caught:

            i = self.jointStates2id[(s, e_s)]
            # Move to next state given the policy and the current state
            next_s, _ = self.__move(s, policy[i])
            path.append(next_s)
            # Move enemy
            next_e_s, _ = self.__e_move(e_s)
            path_enemy.append(next_e_s)

            if next_s == next_e_s:
                self.caught = True
                # Add again paths for the vizualization function
                path.append(next_s)
                path_enemy.append(next_e_s)

            # Update time and state for next iteration
            t  +=1
            s   = next_s
            e_s = next_e_s

        return path, path_enemy

    def show(self):
        print('The states are :')
        print(self.id2states)
        print('The actions are:')
        print(self.actions)
        print('The mapping of the states:')
        print(self.states2id)
        # print('The rewards:')
        # print(self.R)


def draw_maze(grid):

    # Mapp a color to each cell in the grid
    col_mapp = {0: WHITE, 1: BLACK,
                2: LIGHT_GREEN, 3: LIGHT_ORANGE, -1: LIGHT_RED}

    # Give a color to each cell
    rows, cols = grid.shape
    colored_maze = [[col_mapp[grid[j, i]]
                     for i in range(cols)] for j in range(rows)]

    # Create figure of the size of the grid
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


def animate_solution(grid, path, path_enemy=None):

    # Mapp a color to each cell in the grid
    col_mapp = {0: WHITE, 1: BLACK,
                2: LIGHT_GREEN, 3: LIGHT_ORANGE, -1: LIGHT_RED}

    # Size of the grid
    rows, cols = grid.shape

    # Create figure of the size of the grid
    fig = plt.figure(1, figsize=(cols, rows))

    # Remove the axis ticks and add title title
    ax = plt.gca()
    ax.set_title('Policy simulation')
    ax.set_xticks([])
    ax.set_yticks([])

    # Give a color to each cell
    colored_maze = [[col_mapp[grid[j, i]]
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
        grid.get_celld()[(path[i])].set_facecolor(LIGHT_GREEN)
        grid.get_celld()[(path[i])].get_text().set_text('Robber')
        grid.get_celld()[(path_enemy[i])].set_facecolor(LIGHT_RED)
        grid.get_celld()[(path_enemy[i])].get_text().set_text('Police')
        if i > 0:
            # Re-initialize previous cells
            grid.get_celld()[(path[i-1])].set_facecolor(WHITE)
            grid.get_celld()[(path[i-1])].get_text().set_text('')
            grid.get_celld()[(path_enemy[i-1])].set_facecolor(WHITE)
            grid.get_celld()[(path_enemy[i-1])].get_text().set_text('')

            # if waiting but no the end of the path
            if path[i] == path[i-1]:
                if i != len(path)-1:
                    grid.get_celld()[(path[i])].set_facecolor(LIGHT_PURPLE)
                    grid.get_celld()[(path[i])].get_text().set_text(
                        'Stealing')
                else:
                    grid.get_celld()[(path[i])].set_facecolor(LIGHT_GREEN)
                    grid.get_celld()[(path[i])].get_text().set_text(
                        'Win !')
                    break

            if path[i] == path_enemy[i]:
                grid.get_celld()[(path[i])].set_facecolor(LIGHT_RED)
                grid.get_celld()[(path[i])].get_text().set_text('Caught !')
                break

        display.display(fig)
        display.clear_output(wait=True)
        time.sleep(1)
