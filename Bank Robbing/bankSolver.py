''' Main for solving Bank problem with Q-learning. '''

import numpy as np
import matplotlib.pyplot as plt
from random import choice, random
from tqdm import tqdm
import time

import bank as bk


def Q_learning(env, lmbda=0.8, T=1000000):
    ''' Q-function learning algorithm, 
    @param step  : learning rate
    @param lmbda : discount factor
    @param T     : time limit
    @return Q, P : Q function and policy P
    '''

    # Initialization of Q, dimension SxA
    dim = (env.n_jointStates, env.n_actions)
    Q   = np.zeros(dim)
    # State-action counter
    c   =  np.zeros(dim)
    # Positions and joint state
    s   = env.start
    p   = env.e_start
    i   = env.jointStates2id[(s,p)]

    # Q-function improvement
    t = 0
    for t in tqdm(range(T)):
        a          = choice(list(env.actions.keys()))
        next_s, _  = env._Bank__move(s, a)
        next_p, _  = env._Bank__e_move(p)
        next_i     = env.jointStates2id[(next_s, next_p)]
        c[i, a]   += 1
        alpha      = 1./(c[i,a]**(2./3))

        if next_i in env.losingStates:
            r = env.CAUGHT_REWARD
        elif next_s == env.money:
            r = env.IN_BANK_REWARD
        else:
            r = 0

        maxQ    = np.max(Q[next_i])
        Q[i, a] = Q[i, a] + alpha * (r + (lmbda * maxQ) - Q[i, a])

        s = next_s
        p = next_p
        i = next_i
        t += 1

    # Optimal policy for the best actions (axis 1) given each state (axis 0)
    P = np.argmax(Q, axis=1)
    return Q, P


def sarsa(env, lmbda=0.8, eps=0.1, alpha=0.05, T=10000000):
    ''' SARSA algorithm for Q-function learning, 
    @param alpha : learning rate
    @param lmbda : discount factor
    @param eps   : proba epsilon for exploration
    @param T     : time limit
    @return Q, P : Q function and policy P
    '''
    # Epsilon-greedy policy
    # With proba (1-eps) I follow my best policy
    # With proba eps     I explore
    def chooseAction(eps,Qi):
        if random() < eps:
            # Explore by choising random action
            a = choice(list(env.actions.keys()))
        else:
            # Follow the policy: argmax_b(Q(s, b))
            a = np.argmax(Qi) if np.count_nonzero(Qi) != 0 else \
                choice(list(env.actions.keys()))
        return a

    # Initialization of Q, dimension SxA
    dim = (env.n_jointStates, env.n_actions)
    Q = np.zeros(dim)
    # State-action counter
    c = np.zeros(dim)
    # Positions and joint state
    s = env.start
    p = env.e_start
    i = env.jointStates2id[(s, p)]

    # Q-function improvement
    t = 0
    for t in tqdm(range(T)):
        # Greedy or non-greedy action
        a         = chooseAction(eps, Q[i])
        next_s, _ = env._Bank__move(s, a)
        next_p, _ = env._Bank__e_move(p)
        next_i    = env.jointStates2id[(next_s, next_p)]
        c[i, a]  += 1

        # Choose next action (greedily or not)
        next_a = chooseAction(eps, Q[next_i])

        # Constant step-size or not ? works with constant
        # alpha     = 1./(c[i, a]**(2./3))

        if next_i in env.losingStates:
            r = env.CAUGHT_REWARD
        elif next_s == env.money:
            r = env.IN_BANK_REWARD
        else:
            r = 0

        Q[i, a] = Q[i, a] + alpha * (r + lmbda * Q[next_i, next_a] - Q[i, a])

        s   = next_s
        p   = next_p
        i   = next_i
        t   += 1

        # Epsilon can change over time too
        # eps += 0.5/t

    # Optimal policy for the best actions (axis 1) given each state (axis 0)
    P = np.argmax(Q, axis=1)
    return Q, P

if __name__ == "__main__":
    # Description of the town as a numpy array
    # 1 is the robber starting position
    # 2 is the police starting position
    # 3 is the bank position
    town = np.array([
        [1, 0, 0, 0],
        [0, 3, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 2]
    ])
    # Bank instance
    bank = bk.Bank(town)
    # Q, P = Q_learning(bank)
    Q, P = sarsa(bank)
    path, path_enemy = bank.simulate(P, 10)
