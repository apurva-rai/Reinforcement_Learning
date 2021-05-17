# Reinforcement Learning

Reinforcement Learning is a category of machine learning algorithms where an agent learns to behave in an unknown environment by performing actions and learning from the results. The agent's goal is to reinforce its maximal paths for long-term rewards. I have implemented two popular algorithms - Q-learning and SARSA (state-action-reward-state-action). Both algorithms have a similar approach where the model is updated with time with an optimal policy towards the final goal.

## Q-Learning

Q-learning is a model-free reinforcement learning algorithm to learn the value of an action in a particular state. It does not require a model of the environment (hence "model-free"), and it can handle problems with stochastic transitions and rewards without requiring adaptations.

## SARSA

This is an algorithm for learning a Markov decision process policy. This name simply reflects the fact that the main function for updating the Q-value depends on the current state of the agent *S1*, the action the agent chooses *A1*, the reward *R* the agent gets for choosing this action, the state *S2* that the agent enters after taking that action, and finally the next action *A2* the agent chooses in its new state.

## Execution

The code is simple and only uses the standard extra dependancy of Numpy. Execute the program by traversing to the appropriate directory in shell and running `py main.py`

## Functionality

This code uses a simple and straightforward treasure system where the path to the *treasure* is rewarded. In my example I used a grid of 9 *states* or rooms. We start at *0* and the treasure is at *8* with the according weights. One can go into code and change the starting location for different paths as desired as well. For the sake of this test case the starting point is *0*.

![alt text](https://github.com/apurva-rai/Reinforcement_Learning/tree/main/images/graphNew.png)

![alt text](https://github.com/apurva-rai/Reinforcement_Learning/tree/main/images/results.png)

## References

https://programming.vip/docs/q-learning-sarsa-2d-treasure-hunt.html

https://towardsdatascience.com/introduction-to-q-learning-88d1c4f2b49c

https://en.wikipedia.org/wiki/Q-learning

https://en.wikipedia.org/wiki/State%E2%80%93action%E2%80%93reward%E2%80%93state%E2%80%93action

https://tcnguyen.github.io/reinforcement_learning/sarsa_vs_q_learning.html

