# The Maze and the Random Minotaur
Consider the maze in Figure 1. You enter the maze in A and at the same time, the minotaur
enters in B. The minotaur follows a random walk while staying within the limits of the maze. The
minotaur’s walk goes through walls (which obviously you cannot do). At each step, you observe the
position of the minotaur, and decide on a one-step move (up, down, right or left) or not to move.
If the minotaur catches you, he will eat you. Your objective is to identify a strategy maximizing
the probability of exiting the maze (reaching B) before time T.

Note 1: Neither you nor the minotaur can walk diagonally.
Note 2: The minotaur catches you, if and only if, you are located at the same position, at the
same time.

![Figure 1: The minotaur’s maze.](../minotaur%27s%20maze.png)

(a) Formulate the problem as an MDP.
(b) Solve the problem, and illustrate an optimal policy for $T = 20^2$ Plot the maximal probability
of exiting the maze as a function of T. Is there a difference if the minotaur is allowed to
stand still? If so, why?
(c) Assume now that your life is geometrically distributed with mean 30. Modify the problem
so as to derive a policy minimizing the expected time to exit the maze. Motivate your new
problem formulation (model). Estimate the probability of getting out alive using this policy
by simulating 10 000 games.

![Best policy example](minotaur%27s_maze_policy_example.gif)
