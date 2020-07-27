''' Main for solving a Maze object from maze.py. '''

import numpy as np
import matplotlib.pyplot as plt
import time

import maze as mz


# Implemented methods
methods = ['DynProg', 'ValIter']


def dynamic_programming(env, horizon):
    """ Solves the shortest path problem using dynamic programming
        :input Maze env           : The maze environment in which we seek to
                                    find the shortest path.
        :input int horizon        : The time T up to which we solve the problem.
        :return numpy.array V     : Optimal values for every state at every
                                    time, dimension S*T
        :return numpy.array policy: Optimal time-varying policy at every state,
                                    dimension S*T
    """
    # The dynamic prgramming requires the knowledge of :
    # - Transition probabilities
    # - Rewards
    # - State space
    # - Action space
    # - The finite horizon
    p = env.T
    r = env.R
    n_states = env.n_states if not env.enemy else env.n_jointStates
    n_actions = env.n_actions
    T = horizon

    # The variables involved in the dynamic programming backwards recursions
    V         = np.zeros((n_states, T+1))
    policy    = np.zeros((n_states, T+1))
    Q         = np.zeros((n_states, n_actions))

    # Initialization
    Q = np.copy(r)
    V[:, T] = np.max(Q, 1)
    policy[:, T] = np.argmax(Q, 1)

    # The dynamic programming bakwards recursion
    for t in range(T-1, -1, -1):
        # Update the value function acccording to the bellman equation
        # for i, _ in env.id2jointStates.items():
            # for s in range(n_states):
            # Avoid only action loop:
            # Q[i, :] = r[i, :] + np.dot(p[:, i, :].T, V[:, t+1])
            # Both loops:
            # for a in range(n_actions):
                # Update of the temporary Q values
                # Q[i, a] = r[i, a] + np.dot(p[:, i, a], V[:, t+1])

        # No loops
        Q = r + np.transpose(np.dot(p.T, V[:, t+1]))
        # Update by taking the maximum Q value w.r.t the action a
        V[:, t] = np.max(Q, 1)
        # The optimal action is the one that maximizes the Q function
        policy[:, t] = np.argmax(Q, 1)

    return V, policy


def value_iteration(env, gamma, epsilon):
    """ Solves the shortest path problem using value iteration
        :input Maze env           : The maze environment in which we seek to
                                    find the shortest path.
        :input float gamma        : The discount factor.
        :input float epsilon      : accuracy of the value iteration procedure.
        :return numpy.array V     : Optimal values for every state at every
                                    time, dimension S*T
        :return numpy.array policy: Optimal time-varying policy at every state,
                                    dimension S*T
    """
    # The value itearation algorithm requires the knowledge of :
    # - Transition probabilities
    # - Rewards
    # - State space
    # - Action space
    # - The finite horizon
    p = env.T
    r = env.R
    n_states = env.n_states
    n_actions = env.n_actions

    # Required variables and temporary ones for the VI to run
    V = np.zeros(n_states)
    Q = np.zeros((n_states, n_actions))
    BV = np.zeros(n_states)
    # Iteration counter
    n = 0
    # Tolerance error
    tol = (1 - gamma) * epsilon/gamma

    # Initialization of the VI
    for s in range(n_states):
        for a in range(n_actions):
            Q[s, a] = r[s, a] + gamma*np.dot(p[:, s, a], V)
    BV = np.max(Q, 1)

    # Iterate until convergence
    while np.linalg.norm(V - BV) >= tol and n < 200:
        # Increment by one the numbers of iteration
        n += 1
        # Update the value function
        V = np.copy(BV)
        # Compute the new BV
        for s in range(n_states):
            for a in range(n_actions):
                Q[s, a] = r[s, a] + gamma*np.dot(p[:, s, a], V)
        BV = np.max(Q, 1)
        # Show error
        #print(np.linalg.norm(V - BV))

    # Compute policy
    policy = np.argmax(Q, 1)
    # Return the obtained policy
    return V, policy


if __name__ == "__main__":
    # Description of the maze as a numpy array
    maze = np.array([
         [0, 0, 1, 0, 0, 0, 0, 0],
         [0, 0, 1, 0, 0, 1, 0, 0],
         [0, 0, 1, 0, 0, 1, 1, 1],
         [0, 0, 1, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0],
         [0, 1, 1, 1, 1, 1, 1, 0],
         [0, 0, 0, 0, 1, 2, 0, 0],
    ])

    # with the convention
    # 0 = empty cell
    # 1 = obstacle
    # 2 = exit of the Maze
    env = mz.Maze(maze, enemy=(5, 6))
    # # Finite horizon
    horizon = 20
    # # Solve the MDP problem with dynamic programming
    V, policy = dynamic_programming(env, horizon)

    # method = 'DynProg'
    # start  = (0,0)
    # path, path_enemy = env.simulate(start, policy, method)