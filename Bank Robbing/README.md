# Bank Robbing
You are a bank robber trying to heist the bank of an unknown town. You enter the town at position A (see Figure 1), the police starts from the opposite corner, and the bank is at position B. For each round spent in the bank, you receive a reward of 1 SEK. The police walks randomly (that is, uniformly at random up, down, left or right) across the grid and whenever you are caught (you are in the same cell as the police) you lose 10 SEK.

You are new in town, and hence oblivious to the value of the rewards, the position of the bank, the starting point and the movement strategy of the police. Before you take an action (move up, down, left, right or stand still), you can observe both your position and that of the police. Your task is to develop an algorithm learning the policy that maximizes your total discounted reward for a discount factor λ = 0.8 .

<p align="center">
  <img src="../images/town.png" width="40%" height="40%"/>
</p>
<p align="center">
  Figure 1: The unknown town.
</p>



**(a)** Solve the problem by implementing the Q-learning algorithm exploring actions uniformly at random. Create a plot of the value function over time (in particular, for the initial state),
showing the convergence of the algorithm. Note: Expect the value function to converge after roughly 10 000 000 iterations (for step size <img src="https://render.githubusercontent.com/render/math?math=T={1/(n(s,a)^{\frac{2}{3}})}">, where n(s, a) is the number of updates of Q(s, a)).

**(b)** Solve the problem by implementing the SARSA algorithm using "-greedy exploration (initially " = 0:1). Show the convergence for different values of ".
